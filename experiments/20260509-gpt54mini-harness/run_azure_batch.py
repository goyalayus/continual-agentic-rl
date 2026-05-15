#!/usr/bin/env python3
"""Run Azure GPT-5.4-mini Tau3 harness tasks and pack one trace artifact."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tau2.runner.batch import _cleanup_thread_event_loop, _init_thread_event_loop
from tau3_custom_harness.run_banking import run_banking_task

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
TASKS_JSON = REPO_ROOT / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks.json"


@dataclass(frozen=True)
class RepeatSpec:
    bench_run_id: str
    seed: int


@dataclass(frozen=True)
class WorkItem:
    task_id: str
    bench_run_id: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task ids to run. Defaults to every banking task.",
    )
    parser.add_argument("--model", default="azure/gpt-5.4-mini")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Per-simulation wallclock timeout in seconds. Use 0 for no timeout.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subagent-delegation",
        choices=["single", "batch"],
        default="batch",
        help="Knowledge delegation mode for the custom harness.",
    )
    parser.add_argument("--batch-name", default=None)
    parser.add_argument(
        "--repeat",
        action="append",
        default=None,
        metavar="BATCH_NAME:SEED",
        help=(
            "Run multiple batch names inside this Python process. Repeat this "
            "flag with values like baseline_r1_s849558:849558. When set, "
            "--batch-name and --seed are ignored for the batch matrix."
        ),
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Skip completed task folders for this batch name and rerun only "
            "missing or failed tasks."
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Number of task worker threads to run at once. 0 means all selected tasks.",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.environ.get("TAU3_BENCH_S3_URI"),
        help=(
            "Optional S3 destination for the single .jsonl.gz bench artifact. "
            "If this is a prefix, the artifact filename is appended."
        ),
    )
    parser.add_argument(
        "--s3-strict",
        action="store_true",
        help="Fail if the final single-file S3 upload fails.",
    )
    parser.add_argument(
        "--allow-bm25-only",
        action="store_true",
        help="Allow calibration runs without OPENROUTER_API_KEY query embeddings.",
    )
    parser.add_argument(
        "--provider-error-retries",
        type=int,
        default=4,
        help=(
            "Extra whole-attempt retries for transient provider transport/server "
            "errors. Content-policy, auth, quota, and validation errors are not "
            "retried here."
        ),
    )
    return parser.parse_args()


def llm_args_for_run(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {"max_tokens": args.max_tokens}
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    elif args.reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    return payload


def normalized_timeout_seconds(raw_timeout: int | None) -> int | None:
    if raw_timeout is None or raw_timeout <= 0:
        return None
    return raw_timeout


def is_retryable_provider_error(exc: BaseException, traceback_text: str) -> bool:
    text = f"{type(exc).__name__}: {exc}\n{traceback_text}".lower()
    non_retryable_patterns = (
        "authenticationerror",
        "api_key",
        "contentpolicyviolation",
        "content policy",
        "cyber_policy",
        "quota",
        "credit",
        "billing",
        "invalid message",
        "badrequesterror",
    )
    if any(pattern in text for pattern in non_retryable_patterns):
        return False
    retryable_patterns = (
        "ratelimiterror",
        "too many requests",
        "too_many_requests",
        "status code: 429",
        "status_code=429",
        "bad file descriptor",
        "server disconnected without sending a response",
        "server_error",
        "httpx.readerror",
        "httpcore.readerror",
        "apierror",
    )
    return any(pattern in text for pattern in retryable_patterns)


def provider_retry_sleep_seconds(exc: BaseException, attempt_index: int) -> float:
    text = f"{type(exc).__name__}: {exc}".lower()
    if (
        "ratelimiterror" in text
        or "too many requests" in text
        or "too_many_requests" in text
        or "429" in text
    ):
        return min(90.0, 15.0 * attempt_index)
    return min(30.0, 2**attempt_index)


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.replace("export ", "").strip()
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_all_task_ids() -> list[str]:
    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    return [task["id"] for task in tasks]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "bench_run"


def parse_repeat_specs(args: argparse.Namespace) -> list[RepeatSpec]:
    specs = []
    for raw_spec in args.repeat or []:
        if ":" not in raw_spec:
            raise SystemExit(
                "--repeat must use BATCH_NAME:SEED, got " + repr(raw_spec)
            )
        raw_name, raw_seed = raw_spec.rsplit(":", 1)
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise SystemExit(
                f"--repeat seed must be an integer, got {raw_seed!r}"
            ) from exc
        specs.append(RepeatSpec(bench_run_id=safe_name(raw_name), seed=seed))

    duplicate_names = [
        spec.bench_run_id
        for spec in specs
        if sum(other.bench_run_id == spec.bench_run_id for other in specs) > 1
    ]
    if duplicate_names:
        raise SystemExit(
            "--repeat batch names must be unique after sanitization: "
            + ", ".join(sorted(set(duplicate_names)))
        )
    return specs


def patch_azure_env(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    if env.get("AZURE_OPENAI_API_KEY") and not env.get("AZURE_API_KEY"):
        env["AZURE_API_KEY"] = env["AZURE_OPENAI_API_KEY"]
    if env.get("AZURE_OPENAI_ENDPOINT") and not env.get("AZURE_API_BASE"):
        env["AZURE_API_BASE"] = env["AZURE_OPENAI_ENDPOINT"]
    if env.get("AZURE_OPENAI_API_VERSION") and not env.get("AZURE_API_VERSION"):
        env["AZURE_API_VERSION"] = env["AZURE_OPENAI_API_VERSION"]
    return env


def require_azure_env(env: dict[str, str]) -> None:
    missing = [
        name
        for name in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION")
        if not env.get(name)
    ]
    if missing:
        raise SystemExit(
            "Missing Azure env vars after alias mapping: " + ", ".join(missing)
        )


def require_model_env(model: str, env: dict[str, str]) -> None:
    if model.startswith("azure/"):
        require_azure_env(env)
        return
    if model.startswith("openrouter/") and not env.get("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY for OpenRouter model run")


def require_hybrid_retrieval_env(env: dict[str, str], *, allow_bm25_only: bool) -> None:
    if allow_bm25_only or env.get("OPENROUTER_API_KEY"):
        if allow_bm25_only:
            return
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = env["OPENROUTER_API_KEY"]
        try:
            from tau3_custom_harness.retrieval import BankingHybridRetriever

            retriever = BankingHybridRetriever()
            if retriever._query_embedding("hybrid retrieval preflight") is not None:
                return
        finally:
            if original_key is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = original_key
        raise SystemExit(
            "OPENROUTER_API_KEY is set, but embedding preflight failed. The "
            "custom harness would silently degrade to BM25-only; pass "
            "--allow-bm25-only only for an explicit degraded-control run."
        )
    raise SystemExit(
        "Missing OPENROUTER_API_KEY. The custom harness is meant to run hybrid "
        "BM25 + embedding retrieval; pass --allow-bm25-only only for an explicit "
        "degraded-control run."
    )


def run_task(
    task_id: str,
    args: argparse.Namespace,
    env: dict[str, str],
    bench_run_id: str,
) -> dict[str, Any]:
    llm_args = json.dumps(llm_args_for_run(args))
    run_id = safe_name(f"{bench_run_id}_{task_id}")
    run_dir = RUNS_DIR / run_id
    if args.auto_resume:
        resumed = resume_row_for_task(task_id, bench_run_id)
        if resumed is not None:
            return resumed
    if run_dir.exists():
        shutil.rmtree(run_dir)
    started = time.time()
    retry_count = 0
    last_error: BaseException | None = None
    max_attempts = 1 + max(0, args.provider_error_retries)
    for attempt_index in range(max_attempts):
        if attempt_index > 0:
            retry_count = attempt_index
            if run_dir.exists():
                shutil.rmtree(run_dir)
            if last_error is not None:
                time.sleep(provider_retry_sleep_seconds(last_error, attempt_index))
        _init_thread_event_loop()
        try:
            result = run_banking_task(
                task_id=task_id,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                seed=args.seed,
                agent_model=args.model,
                user_model=args.model,
                subagent_model=args.model,
                subagent_delegation=args.subagent_delegation,
                temperature=args.temperature,
                agent_llm_args_json=llm_args,
                user_llm_args_json=llm_args,
                subagent_llm_args_json=llm_args,
                log_dir=RUNS_DIR,
                run_id=run_id,
                timeout=normalized_timeout_seconds(args.timeout_seconds),
            )
            returncode = 0
            output_tail = ""
            break
        except Exception as exc:
            last_error = exc
            result = read_child_result(run_dir)
            returncode = 1
            output_tail = traceback.format_exc()[-5000:]
            if attempt_index < max_attempts - 1 and is_retryable_provider_error(
                exc, output_tail
            ):
                print(
                    f"{task_id}\tretry_provider_error\t"
                    f"attempt={attempt_index + 1}/{max_attempts}\t"
                    f"error={str(exc).splitlines()[0][:240]}",
                    flush=True,
                )
                continue
            break
        finally:
            _cleanup_thread_event_loop()

    elapsed = time.time() - started
    return {
        "task_id": task_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "elapsed_seconds": round(elapsed, 2),
        "result": result,
        "output_tail": output_tail,
        "provider_error_retries": retry_count,
    }


def read_child_result(run_dir: Path) -> dict[str, Any] | None:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        result = read_json(result_path)
    except Exception:
        return None
    if isinstance(result, dict) and "run_id" in result and "task_id" in result:
        return result
    return None


def is_completed_run_dir(run_dir: Path, *, task_id: str, run_id: str) -> bool:
    if not run_dir.exists():
        return False
    if (run_dir / "run_error.json").exists():
        return False
    if not (run_dir / "simulation.json").exists():
        return False
    result = read_child_result(run_dir)
    if result is None:
        return False
    if result.get("task_id") != task_id:
        return False
    if result.get("run_id") != run_id:
        return False
    if "reward" not in result:
        return False
    return result.get("termination_reason") is not None


def resume_row_for_task(task_id: str, bench_run_id: str) -> dict[str, Any] | None:
    run_id = safe_name(f"{bench_run_id}_{task_id}")
    run_dir = RUNS_DIR / run_id
    if not is_completed_run_dir(run_dir, task_id=task_id, run_id=run_id):
        return None
    return {
        "task_id": task_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returncode": 0,
        "elapsed_seconds": None,
        "result": read_child_result(run_dir),
        "output_tail": "auto_resume: reused completed task artifacts",
        "resumed": True,
    }


def not_run_row_for_task(task_id: str, bench_run_id: str, reason: str) -> dict[str, Any]:
    run_id = safe_name(f"{bench_run_id}_{task_id}")
    run_dir = RUNS_DIR / run_id
    return {
        "task_id": task_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returncode": None,
        "elapsed_seconds": None,
        "result": None,
        "output_tail": f"not run: {reason}",
        "skipped_reason": reason,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def row_text_for_error_detection(row: dict[str, Any]) -> str:
    parts = [str(row.get("output_tail") or "")]
    run_dir_text = row.get("run_dir") or (row.get("result") or {}).get("run_dir")
    if run_dir_text:
        run_dir = Path(run_dir_text)
        for name in ("run_error.json", "events.jsonl"):
            path = run_dir / name
            if not path.exists():
                continue
            try:
                parts.append(path.read_text(encoding="utf-8", errors="ignore")[-8000:])
            except OSError:
                pass
    return "\n".join(parts).lower()


def is_provider_credit_error(row: dict[str, Any]) -> bool:
    if row.get("returncode") == 0:
        return False
    text = row_text_for_error_detection(row)
    credit_patterns = (
        "requires more credits",
        "insufficient credits",
        "insufficient credit",
        "insufficient quota",
        "quota exceeded",
        "out of credits",
        "credit balance",
        "payment required",
        "billing hard limit",
        "billing issue",
        "billing quota",
        "billing account",
    )
    return any(pattern in text for pattern in credit_patterns)


def artifact_status_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("returncode") == 0]
    resumed = [row for row in rows if row.get("resumed")]
    failed = [row for row in rows if row.get("returncode") not in (0, None)]
    not_run = [row for row in rows if row.get("skipped_reason")]
    credit_limited = [row for row in rows if is_provider_credit_error(row)]
    return {
        "completed_task_count": len(completed),
        "resumed_task_count": len(resumed),
        "rerun_task_count": len([row for row in completed if not row.get("resumed")]),
        "failed_task_count": len(failed),
        "not_run_task_count": len(not_run),
        "provider_credit_exhausted": bool(credit_limited),
        "provider_credit_task_ids": [row["task_id"] for row in credit_limited],
        "stop_reason": infer_stop_reason(rows),
    }


def infer_stop_reason(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if row.get("skipped_reason"):
            return str(row["skipped_reason"])
    if any(is_provider_credit_error(row) for row in rows):
        return "provider_credit_exhausted"
    return None


def load_run_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result") or {}
    run_dir_text = result.get("run_dir") or row.get("run_dir")
    if not run_dir_text:
        return {}

    run_dir = Path(run_dir_text)
    artifacts: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in ("result.json", "simulation.json", "kb_evidence.json", "run_error.json"):
        path = run_dir / name
        if path.exists():
            try:
                artifacts[name.removesuffix(".json")] = read_json(path)
            except Exception as exc:
                artifacts.setdefault("artifact_errors", []).append(
                    {"path": str(path), "error": str(exc)}
                )

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        try:
            artifacts["events"] = read_jsonl(events_path)
        except Exception as exc:
            artifacts.setdefault("artifact_errors", []).append(
                {"path": str(events_path), "error": str(exc)}
            )

    llm_calls = []
    for path in sorted((run_dir / "llm_calls").glob("*.json")):
        try:
            llm_calls.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "payload": read_json(path),
                }
            )
        except Exception as exc:
            artifacts.setdefault("artifact_errors", []).append(
                {"path": str(path), "error": str(exc)}
            )
    if llm_calls:
        artifacts["llm_calls"] = llm_calls
    return artifacts


def write_bench_artifact(
    bench_run_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    task_order: list[str],
    parallelism: int,
    started_at: str,
    completed_at: str,
) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    row_by_task = {row["task_id"]: row for row in rows}
    metadata = {
        "type": "bench_run",
        "schema_version": 1,
        "bench_run_id": bench_run_id,
        "created_at": started_at,
        "completed_at": completed_at,
        "task_count": len(task_order),
        "parallelism": parallelism,
        "model": args.model,
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
        "max_tokens": args.max_tokens,
        "timeout_seconds": normalized_timeout_seconds(args.timeout_seconds),
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_enabled": args.reasoning_enabled,
        "subagent_delegation": args.subagent_delegation,
        "retrieval_mode": "bm25_only" if args.allow_bm25_only else "hybrid",
        "auto_resume": args.auto_resume,
        "provider_error_retries": max(0, args.provider_error_retries),
        **artifact_status_counts(rows),
    }
    path = ARTIFACTS_DIR / f"{bench_run_id}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, default=str) + "\n")
        for task_index, task_id in enumerate(task_order):
            row = row_by_task[task_id]
            record = {
                "type": "task_trace",
                "bench_run_id": bench_run_id,
                "task_index": task_index,
                "task_id": task_id,
                "returncode": row.get("returncode"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "timeout": row.get("timeout", False),
                "provider_error_retries": row.get("provider_error_retries", 0),
                "result": row.get("result"),
                "stdout_tail": row.get("output_tail"),
                "artifacts": load_run_artifacts(row),
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


def s3_destination(s3_uri: str, artifact_path: Path) -> str:
    if s3_uri.endswith(".jsonl.gz"):
        return s3_uri
    return s3_uri.rstrip("/") + "/" + artifact_path.name


def upload_artifact_to_s3(artifact_path: Path, s3_uri: str, *, strict: bool) -> bool:
    destination = s3_destination(s3_uri, artifact_path)
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", str(artifact_path), destination],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        message = "AWS CLI executable not found; single-file S3 upload skipped."
        if strict:
            raise RuntimeError(message)
        print(message, file=sys.stderr, flush=True)
        return False
    if result.returncode != 0:
        message = (
            "Single-file S3 upload failed "
            f"(returncode={result.returncode}): {result.stderr[-1000:]}"
        )
        if strict:
            raise RuntimeError(message)
        print(message, file=sys.stderr, flush=True)
        return False
    print(f"s3_artifact={destination}", flush=True)
    return True


def print_table(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        result = row.get("result") or {}
        reward = result.get("reward")
        reason = result.get("termination_reason")
        run_dir = result.get("run_dir") or row.get("run_dir")
        if row.get("skipped_reason"):
            status = "not_run"
        elif row.get("resumed"):
            status = "resume"
        else:
            status = "ok" if row["returncode"] == 0 else "fail"
        print(
            f"{row['task_id']}\t{status}\treward={reward}\t"
            f"reason={reason}\tseconds={row['elapsed_seconds']}\trun_dir={run_dir}",
            flush=True,
        )


def run_batch(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.update(load_env_file(args.env_file))
    env = patch_azure_env(env)
    require_model_env(args.model, env)
    require_hybrid_retrieval_env(env, allow_bm25_only=args.allow_bm25_only)
    os.environ.update(env)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    task_ids = list(args.tasks) if args.tasks is not None else load_all_task_ids()
    if not task_ids:
        raise SystemExit("No tasks selected")

    repeat_specs = parse_repeat_specs(args)
    if not repeat_specs:
        raw_bench_run_id = args.batch_name or datetime.now(timezone.utc).strftime(
            "bench_run_%Y%m%d_%H%M%S"
        )
        repeat_specs = [RepeatSpec(safe_name(raw_bench_run_id), args.seed)]

    work_items = [
        WorkItem(task_id=task_id, bench_run_id=spec.bench_run_id, seed=spec.seed)
        for spec in repeat_specs
        for task_id in task_ids
    ]
    parallelism = args.parallelism if args.parallelism > 0 else len(work_items)
    parallelism = max(1, min(parallelism, len(work_items)))

    started_at = datetime.now(timezone.utc).isoformat()
    if len(repeat_specs) == 1:
        print(
            f"bench_run={repeat_specs[0].bench_run_id} tasks={len(task_ids)} "
            f"parallelism={parallelism}",
            flush=True,
        )
    else:
        print(
            "bench_runs="
            + ",".join(spec.bench_run_id for spec in repeat_specs)
            + f" tasks={len(task_ids)} attempts={len(work_items)} "
            + f"parallelism={parallelism}",
            flush=True,
        )

    rows_by_item: dict[tuple[str, str], dict[str, Any]] = {}
    pending_items: list[WorkItem] = []
    if args.auto_resume:
        for item in work_items:
            resumed = resume_row_for_task(item.task_id, item.bench_run_id)
            if resumed is None:
                pending_items.append(item)
            else:
                rows_by_item[(item.bench_run_id, item.task_id)] = resumed
                print_table([resumed])
    else:
        pending_items = list(work_items)

    pending_index = 0
    stop_reason: str | None = None

    def make_error_row(
        item: WorkItem, returncode: int, output_tail: str
    ) -> dict[str, Any]:
        run_id = safe_name(f"{item.bench_run_id}_{item.task_id}")
        return {
            "task_id": item.task_id,
            "run_id": run_id,
            "run_dir": str(RUNS_DIR / run_id),
            "returncode": returncode,
            "elapsed_seconds": None,
            "result": read_child_result(RUNS_DIR / run_id),
            "output_tail": output_tail[-5000:],
            "runner_error": True,
        }

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_item: dict[Any, WorkItem] = {}

        def submit_until_full() -> None:
            nonlocal pending_index
            while (
                stop_reason is None
                and len(future_to_item) < parallelism
                and pending_index < len(pending_items)
            ):
                item = pending_items[pending_index]
                pending_index += 1
                task_args = argparse.Namespace(**vars(args))
                task_args.seed = item.seed
                future = executor.submit(
                    run_task, item.task_id, task_args, env, item.bench_run_id
                )
                future_to_item[future] = item

        submit_until_full()
        while future_to_item:
            done, _ = wait(future_to_item, return_when=FIRST_COMPLETED)
            for future in done:
                item = future_to_item.pop(future)
                try:
                    row = future.result()
                except subprocess.TimeoutExpired as exc:
                    output = exc.stdout or ""
                    if isinstance(output, bytes):
                        output = output.decode("utf-8", errors="replace")
                    row = make_error_row(item, 124, output)
                    row["timeout"] = True
                except Exception:
                    row = make_error_row(item, 1, traceback.format_exc())
                rows_by_item[(item.bench_run_id, item.task_id)] = row
                print_table([row])
                if is_provider_credit_error(row):
                    stop_reason = "provider_credit_exhausted"
            submit_until_full()

    if stop_reason is not None:
        for item in pending_items[pending_index:]:
            row = not_run_row_for_task(item.task_id, item.bench_run_id, stop_reason)
            rows_by_item[(item.bench_run_id, item.task_id)] = row
            print_table([row])

    completed_at = datetime.now(timezone.utc).isoformat()
    all_rows: list[dict[str, Any]] = []
    for spec in repeat_specs:
        rows = [rows_by_item[(spec.bench_run_id, task_id)] for task_id in task_ids]
        all_rows.extend(rows)
        artifact_path = write_bench_artifact(
            spec.bench_run_id,
            rows,
            args,
            task_ids,
            parallelism,
            started_at,
            completed_at,
        )
        if args.s3_uri:
            upload_artifact_to_s3(artifact_path, args.s3_uri, strict=args.s3_strict)
        print(f"artifact={artifact_path}", flush=True)

    credit_limited = [
        row["run_id"] for row in all_rows if is_provider_credit_error(row)
    ]
    failed = [
        row["run_id"] for row in all_rows if row.get("returncode") not in (0, None)
    ]
    not_run = [row["run_id"] for row in all_rows if row.get("skipped_reason")]
    if credit_limited or not_run:
        print(
            "provider_credit_limit_detected="
            + ",".join(credit_limited)
            + "\nnot_run="
            + ",".join(not_run)
            + "\nAdd credits, then rerun the same command with --auto-resume.",
            file=sys.stderr,
            flush=True,
        )
        return 42
    if failed:
        print(
            "task_failures_detected="
            + ",".join(failed)
            + "\nRerun the same command with --auto-resume to retry failed tasks.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def main() -> int:
    return run_batch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
