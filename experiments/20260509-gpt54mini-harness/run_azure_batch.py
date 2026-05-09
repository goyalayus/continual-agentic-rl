#!/usr/bin/env python3
"""Run Azure GPT-5.4-mini Tau3 harness tasks and pack one trace artifact."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
TASKS_JSON = REPO_ROOT / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks.json"


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
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-name", default=None)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Number of task subprocesses to run at once. 0 means all selected tasks.",
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
    return parser.parse_args()


def load_all_task_ids() -> list[str]:
    tasks = json.loads(TASKS_JSON.read_text(encoding="utf-8"))
    return [task["id"] for task in tasks]


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "bench_run"


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
    llm_args = json.dumps({"max_tokens": args.max_tokens})
    run_id = safe_name(f"{bench_run_id}_{task_id}")
    run_dir = RUNS_DIR / run_id
    command = [
        "uv",
        "run",
        "python",
        "tau3_custom_harness/run_banking.py",
        "--task-id",
        task_id,
        "--agent-model",
        args.model,
        "--user-model",
        args.model,
        "--subagent-model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--agent-llm-args-json",
        llm_args,
        "--user-llm-args-json",
        llm_args,
        "--subagent-llm-args-json",
        llm_args,
        "--max-steps",
        str(args.max_steps),
        "--max-errors",
        str(args.max_errors),
        "--log-dir",
        str(RUNS_DIR),
        "--run-id",
        run_id,
    ]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_seconds,
        check=False,
    )
    elapsed = time.time() - started
    parsed = read_child_result(run_dir) or parse_result_from_output(completed.stdout)
    return {
        "task_id": task_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "returncode": completed.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "result": parsed,
        "output_tail": completed.stdout[-5000:],
    }


def parse_result_from_output(output: str) -> dict[str, Any] | None:
    start = output.rfind("{")
    if start < 0:
        return None
    try:
        candidate = json.loads(output[start:])
    except json.JSONDecodeError:
        return None
    if "run_id" in candidate and "task_id" in candidate:
        return candidate
    return None


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


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
        "temperature": args.temperature,
        "retrieval_mode": "bm25_only" if args.allow_bm25_only else "hybrid",
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
        run_dir = result.get("run_dir")
        status = "ok" if row["returncode"] == 0 else "fail"
        print(
            f"{row['task_id']}\t{status}\treward={reward}\t"
            f"reason={reason}\tseconds={row['elapsed_seconds']}\trun_dir={run_dir}",
            flush=True,
        )


def main() -> int:
    args = parse_args()
    env = patch_azure_env(os.environ)
    require_azure_env(env)
    require_hybrid_retrieval_env(env, allow_bm25_only=args.allow_bm25_only)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    task_ids = args.tasks if args.tasks is not None else load_all_task_ids()
    if not task_ids:
        raise SystemExit("No tasks selected")
    parallelism = args.parallelism if args.parallelism > 0 else len(task_ids)
    parallelism = max(1, min(parallelism, len(task_ids)))

    raw_bench_run_id = args.batch_name or datetime.now(timezone.utc).strftime(
        "bench_run_%Y%m%d_%H%M%S"
    )
    bench_run_id = safe_name(raw_bench_run_id)
    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"bench_run={bench_run_id} tasks={len(task_ids)} parallelism={parallelism}",
        flush=True,
    )

    rows_by_task: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        future_to_task = {
            executor.submit(run_task, task_id, args, env, bench_run_id): task_id
            for task_id in task_ids
        }
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                row = future.result()
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or ""
                if isinstance(output, bytes):
                    output = output.decode("utf-8", errors="replace")
                row = {
                    "task_id": task_id,
                    "run_id": safe_name(f"{bench_run_id}_{task_id}"),
                    "run_dir": str(RUNS_DIR / safe_name(f"{bench_run_id}_{task_id}")),
                    "returncode": 124,
                    "elapsed_seconds": args.timeout_seconds,
                    "result": None,
                    "output_tail": output[-5000:],
                    "timeout": True,
                }
            except Exception as exc:
                row = {
                    "task_id": task_id,
                    "run_id": safe_name(f"{bench_run_id}_{task_id}"),
                    "run_dir": str(RUNS_DIR / safe_name(f"{bench_run_id}_{task_id}")),
                    "returncode": 1,
                    "elapsed_seconds": None,
                    "result": None,
                    "output_tail": str(exc),
                    "runner_error": True,
                }
            rows_by_task[task_id] = row
            print_table([row])

    rows = [rows_by_task[task_id] for task_id in task_ids]
    completed_at = datetime.now(timezone.utc).isoformat()
    artifact_path = write_bench_artifact(
        bench_run_id,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
