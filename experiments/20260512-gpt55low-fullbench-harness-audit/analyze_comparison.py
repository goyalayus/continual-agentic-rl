#!/usr/bin/env python3
"""Read-only comparison helper for banking Tau runs.

The script only reads benchmark artifacts and writes summary files in this
experiment directory. It does not import or call the harness.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CUSTOM_ARTIFACTS = (
    REPO_ROOT / "experiments/20260509-gpt54mini-harness/artifacts"
)
DEFAULT_CUSTOM_RUNS = REPO_ROOT / "experiments/20260509-gpt54mini-harness/runs"
DEFAULT_SIMULATIONS = REPO_ROOT / "data/simulations"
DEFAULT_TASKS_DIR = REPO_ROOT / "data/tau2/domains/banking_knowledge/tasks"
DEFAULT_OUTPUT_JSON = EXPERIMENT_DIR / "comparison_summary.json"
DEFAULT_OUTPUT_CSV = EXPERIMENT_DIR / "comparison_summary.csv"
DEFAULT_CUSTOM_SOURCE_PREFIXES = ("baseline_custom_azure_gpt55low_",)
DEFAULT_DEFAULT_SOURCE_PREFIXES = ("baseline_default_tau_bm25_azure_gpt55low_",)
MESSAGE_EXCERPT_CHARS = 1200
TOOL_ARG_EXCERPT_CHARS = 700


Json = dict[str, Any]


@dataclass
class Attempt:
    harness: str
    task_id: str
    run_id: str
    reward: float | None
    passed: bool
    termination_reason: str | None = None
    source_label: str | None = None
    returncode: int | None = None
    error_type: str | None = None
    observed_tool_calls: list[Json] = field(default_factory=list)
    observed_messages: list[Json] = field(default_factory=list)
    event_counts: dict[str, int] = field(default_factory=dict)
    evidence_paths: list[str] = field(default_factory=list)
    run_contract: Json = field(default_factory=dict)

    def merge(self, other: "Attempt") -> None:
        self.reward = self.reward if self.reward is not None else other.reward
        self.passed = self.passed or other.passed
        self.termination_reason = self.termination_reason or other.termination_reason
        self.source_label = self.source_label or other.source_label
        self.returncode = self.returncode if self.returncode is not None else other.returncode
        self.error_type = self.error_type or other.error_type
        if not self.observed_tool_calls:
            self.observed_tool_calls = other.observed_tool_calls
        if not self.observed_messages:
            self.observed_messages = other.observed_messages
        if not self.event_counts:
            self.event_counts = other.event_counts
        if not self.run_contract:
            self.run_contract = other.run_contract
        self.evidence_paths = sorted(set(self.evidence_paths + other.evidence_paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize custom and default Tau banking results by task_id."
    )
    parser.add_argument(
        "--custom-artifact",
        action="append",
        type=Path,
        help="Custom artifact file, usually *.jsonl.gz. Can be passed more than once.",
    )
    parser.add_argument(
        "--custom-artifacts-dir",
        type=Path,
        default=DEFAULT_CUSTOM_ARTIFACTS,
        help="Directory scanned for custom *.jsonl.gz artifacts.",
    )
    parser.add_argument(
        "--custom-runs-dir",
        type=Path,
        default=DEFAULT_CUSTOM_RUNS,
        help="Directory scanned for custom run subdirectories.",
    )
    parser.add_argument(
        "--default-results",
        action="append",
        type=Path,
        help="Default Tau results.json path. Can be passed more than once.",
    )
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=DEFAULT_SIMULATIONS,
        help="Directory scanned recursively for default Tau results.json files.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Banking task-definition directory.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--no-default-scans",
        action="store_true",
        help="Use only explicitly passed artifact/result paths.",
    )
    parser.add_argument(
        "--custom-source-prefix",
        action="append",
        default=None,
        help=(
            "Only include auto-scanned custom artifacts/run dirs whose names start "
            "with this prefix. Can be passed more than once. Use '*' to include all."
        ),
    )
    parser.add_argument(
        "--default-source-prefix",
        action="append",
        default=None,
        help=(
            "Only include auto-scanned default result dirs whose names start with "
            "this prefix. Can be passed more than once. Use '*' to include all."
        ),
    )
    args = parser.parse_args()
    if args.custom_source_prefix is None:
        args.custom_source_prefix = list(DEFAULT_CUSTOM_SOURCE_PREFIXES)
    if args.default_source_prefix is None:
        args.default_source_prefix = list(DEFAULT_DEFAULT_SOURCE_PREFIXES)
    return args


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_pass(reward: float | None) -> bool:
    return reward == 1.0


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compact_text(value: Any, limit: int = MESSAGE_EXCERPT_CHARS) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [truncated {omitted} chars]"


def load_task_definitions(tasks_dir: Path) -> dict[str, Json]:
    tasks: dict[str, Json] = {}
    if not tasks_dir.exists():
        return tasks

    for path in sorted(tasks_dir.glob("task_*.json")):
        task = read_json(path)
        task_id = task.get("id") or path.stem
        tasks[task_id] = {
            "task_id": task_id,
            "gold_expected_actions": normalize_actions(task),
            "reward_basis": normalize_reward_basis(task),
            "required_documents": task.get("required_documents") or [],
            "definition_path": rel(path),
        }

    combined_path = tasks_dir.parent / "tasks.json"
    if combined_path.exists():
        combined = read_json(combined_path)
        if isinstance(combined, list):
            rows = combined
        elif isinstance(combined, dict):
            rows = combined.get("tasks") or combined.get("rows") or []
        else:
            rows = []
        for task in rows:
            if not isinstance(task, dict):
                continue
            task_id = task.get("id")
            if not task_id or task_id in tasks:
                continue
            tasks[task_id] = {
                "task_id": task_id,
                "gold_expected_actions": normalize_actions(task),
                "reward_basis": normalize_reward_basis(task),
                "required_documents": task.get("required_documents") or [],
                "definition_path": rel(combined_path),
            }

    return tasks


def normalize_actions(task: Json) -> list[Json]:
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    normalized = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        normalized.append(
            {
                "action_id": action.get("action_id"),
                "requestor": action.get("requestor"),
                "name": action.get("name"),
                "arguments": action.get("arguments") or {},
            }
        )
    return normalized


def normalize_reward_basis(task: Json) -> list[str]:
    criteria = task.get("evaluation_criteria") or {}
    return list(criteria.get("reward_basis") or [])


def extract_reward_info(simulation: Json | None) -> Json:
    if not isinstance(simulation, dict):
        return {}
    return simulation.get("reward_info") or {}


def extract_tool_calls(simulation: Json | None) -> list[Json]:
    if not isinstance(simulation, dict):
        return []
    calls = []
    for index, message in enumerate(simulation.get("messages") or []):
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            calls.append(
                {
                    "turn_idx": message.get("turn_idx", index),
                    "role": message.get("role"),
                    "requestor": call.get("requestor"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments") or {},
                    "id": call.get("id"),
                }
            )
    return calls


def extract_message_timeline(simulation: Json | None) -> list[Json]:
    if not isinstance(simulation, dict):
        return []
    timeline: list[Json] = []
    for index, message in enumerate(simulation.get("messages") or []):
        if not isinstance(message, dict):
            continue
        row: Json = {
            "index": index,
            "turn_idx": message.get("turn_idx", index),
            "role": message.get("role"),
        }
        requestor = message.get("requestor")
        if requestor:
            row["requestor"] = requestor
        content = compact_text(message.get("content"))
        if content:
            row["content"] = content
        tool_calls = compact_tool_calls(message.get("tool_calls") or [])
        if tool_calls:
            row["tool_calls"] = tool_calls
        if message.get("role") == "tool":
            row["tool_call_id"] = message.get("id")
            if message.get("error"):
                row["error"] = compact_text(message.get("error"), limit=400)
        timeline.append({key: value for key, value in row.items() if value not in (None, "", [])})
    return timeline


def compact_tool_calls(tool_calls: Any) -> list[Json]:
    if not isinstance(tool_calls, list):
        return []
    compacted = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        item: Json = {
            "id": call.get("id"),
            "name": call.get("name"),
            "requestor": call.get("requestor"),
        }
        arguments = call.get("arguments")
        if arguments not in (None, {}, []):
            item["arguments"] = compact_text(arguments, limit=TOOL_ARG_EXCERPT_CHARS)
        compacted.append(
            {key: value for key, value in item.items() if value not in (None, "", [])}
        )
    return compacted


def count_events(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["malformed_event"] += 1
            continue
        counts[str(row.get("event_type") or "unknown")] += 1
    return dict(sorted(counts.items()))


def embedded_event_counts(row: Json) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in ((row.get("artifacts") or {}).get("events") or []):
        if isinstance(event, dict):
            counts[str(event.get("event_type") or "unknown")] += 1
    return dict(sorted(counts.items()))


def custom_run_contract(metadata: Json) -> Json:
    keys = (
        "bench_run_id",
        "task_count",
        "parallelism",
        "model",
        "max_steps",
        "max_errors",
        "max_tokens",
        "temperature",
        "reasoning_effort",
        "reasoning_enabled",
        "subagent_delegation",
        "retrieval_mode",
        "auto_resume",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


def default_run_contract(info: Json, source_label: str) -> Json:
    user_info = info.get("user_info") or {}
    agent_info = info.get("agent_info") or {}
    environment_info = info.get("environment_info") or {}
    return {
        "save_to": source_label,
        "domain_name": environment_info.get("domain_name"),
        "retrieval_config": info.get("retrieval_config"),
        "retrieval_config_kwargs": info.get("retrieval_config_kwargs"),
        "num_trials": info.get("num_trials"),
        "max_steps": info.get("max_steps"),
        "max_errors": info.get("max_errors"),
        "seed": info.get("seed"),
        "agent_implementation": agent_info.get("implementation"),
        "agent_llm": agent_info.get("llm"),
        "agent_llm_args": agent_info.get("llm_args") or {},
        "user_implementation": user_info.get("implementation"),
        "user_llm": user_info.get("llm"),
        "user_llm_args": user_info.get("llm_args") or {},
    }


def classify_custom_error(row: Json) -> str | None:
    if row.get("skipped_reason"):
        return f"not_run:{row['skipped_reason']}"
    if row.get("returncode") in (None, 0):
        return None
    tail = f"{row.get('output_tail') or ''}\n{row.get('stdout_tail') or ''}".lower()
    if "rate limit" in tail or "429" in tail:
        return "provider_rate_limit"
    if "prompt tokens limit exceeded" in tail:
        return "provider_prompt_limit"
    if "unsupportedparamserror" in tail and "temperature" in tail:
        return "provider_temperature"
    if any(
        text in tail
        for text in (
            "requires more credits",
            "insufficient credits",
            "insufficient credit",
            "insufficient quota",
            "quota exceeded",
            "out of credits",
            "credit balance",
            "payment required",
        )
    ):
        return "provider_credit_limit"
    if "content_filter" in tail or "cyber_policy" in tail:
        return "provider_content_filter"
    return "harness_or_provider_error"


def custom_artifact_paths(args: argparse.Namespace) -> list[Path]:
    explicit = [path for path in args.custom_artifact or [] if path.exists()]
    if args.no_default_scans:
        return sorted(explicit)
    scanned = sorted(
        path
        for path in args.custom_artifacts_dir.glob("*.jsonl.gz")
        if matches_prefixes(path.name, args.custom_source_prefix)
    )
    return sorted(set(explicit + scanned))


def default_result_paths(args: argparse.Namespace) -> list[Path]:
    explicit = [path for path in args.default_results or [] if path.exists()]
    if args.no_default_scans:
        return sorted(explicit)
    scanned = sorted(
        path
        for path in args.simulations_dir.glob("*/results.json")
        if matches_prefixes(path.parent.name, args.default_source_prefix)
    )
    return sorted(set(explicit + scanned))


def matches_prefixes(name: str, prefixes: list[str] | tuple[str, ...]) -> bool:
    return "*" in prefixes or any(name.startswith(prefix) for prefix in prefixes)


def attempts_from_custom_artifact(path: Path) -> Iterable[Attempt]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    metadata = next((row for row in rows if row.get("type") == "bench_run"), {})
    source_label = metadata.get("bench_run_id") or path.stem
    run_contract = custom_run_contract(metadata)
    for row in rows:
        if row.get("type") != "task_trace":
            continue
        result = row.get("result") or {}
        artifacts = row.get("artifacts") or {}
        simulation = artifacts.get("simulation")
        run_id = result.get("run_id") or row.get("run_id") or f"{source_label}:{row.get('task_id')}"
        task_id = result.get("task_id") or row.get("task_id")
        if not task_id:
            continue
        reward = as_float(result.get("reward"))
        evidence = [rel(path)]
        run_dir_value = result.get("run_dir") or artifacts.get("run_dir")
        if run_dir_value:
            run_dir = Path(run_dir_value)
            evidence.extend(
                rel(candidate)
                for candidate in (
                    run_dir / "result.json",
                    run_dir / "simulation.json",
                    run_dir / "events.jsonl",
                    run_dir / "kb_evidence.json",
                    run_dir / "run_error.json",
                )
                if candidate.exists()
            )
        yield Attempt(
            harness="custom",
            task_id=task_id,
            run_id=str(run_id),
            reward=reward,
            passed=is_pass(reward),
            termination_reason=result.get("termination_reason"),
            source_label=str(source_label),
            returncode=row.get("returncode"),
            error_type=classify_custom_error(row),
            observed_tool_calls=extract_tool_calls(simulation),
            observed_messages=extract_message_timeline(simulation),
            event_counts=embedded_event_counts(row),
            evidence_paths=sorted(set(evidence)),
            run_contract=run_contract,
        )


def attempts_from_custom_runs(
    runs_dir: Path, source_prefixes: list[str] | tuple[str, ...]
) -> Iterable[Attempt]:
    if not runs_dir.exists():
        return
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if not matches_prefixes(run_dir.name, source_prefixes):
            continue
        result_path = run_dir / "result.json"
        error_path = run_dir / "run_error.json"
        simulation_path = run_dir / "simulation.json"
        events_path = run_dir / "events.jsonl"
        evidence_path = run_dir / "kb_evidence.json"

        result = read_json(result_path) if result_path.exists() else {}
        error = read_json(error_path) if error_path.exists() else {}
        simulation = read_json(simulation_path) if simulation_path.exists() else {}

        task_id = result.get("task_id") or simulation.get("task_id") or error.get("task_id")
        if not task_id:
            task_id = infer_task_id(run_dir.name)
        if not task_id:
            continue

        run_id = result.get("run_id") or simulation.get("id") or run_dir.name
        reward = as_float(result.get("reward"))
        if reward is None:
            reward = as_float(extract_reward_info(simulation).get("reward"))
        error_type = None
        if error_path.exists():
            error_type = str(error.get("error_type") or error.get("type") or "run_error")

        evidence = [
            rel(path)
            for path in (result_path, simulation_path, events_path, evidence_path, error_path)
            if path.exists()
        ]
        yield Attempt(
            harness="custom",
            task_id=str(task_id),
            run_id=str(run_id),
            reward=reward,
            passed=is_pass(reward),
            termination_reason=result.get("termination_reason")
            or simulation.get("termination_reason"),
            source_label=run_dir.name,
            error_type=error_type,
            observed_tool_calls=extract_tool_calls(simulation),
            observed_messages=extract_message_timeline(simulation),
            event_counts=count_events(events_path),
            evidence_paths=evidence,
        )


def infer_task_id(text: str) -> str | None:
    parts = text.replace("-", "_").split("_")
    for index, part in enumerate(parts):
        if part == "task" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return f"task_{int(parts[index + 1]):03d}"
        if part.startswith("task") and part[4:].isdigit():
            return f"task_{int(part[4:]):03d}"
    return None


def attempts_from_default_results(path: Path) -> Iterable[Attempt]:
    data = read_json(path)
    source_label = path.parent.name
    run_contract = default_run_contract(data.get("info") or {}, source_label)
    simulations = data.get("simulations") or []
    for index, simulation in enumerate(simulations):
        if not isinstance(simulation, dict):
            continue
        task_id = simulation.get("task_id")
        if not task_id:
            continue
        reward = as_float(extract_reward_info(simulation).get("reward"))
        run_id = simulation.get("id") or f"{source_label}:{task_id}:{index}"
        yield Attempt(
            harness="default",
            task_id=str(task_id),
            run_id=str(run_id),
            reward=reward,
            passed=is_pass(reward),
            termination_reason=simulation.get("termination_reason"),
            source_label=source_label,
            observed_tool_calls=extract_tool_calls(simulation),
            observed_messages=extract_message_timeline(simulation),
            evidence_paths=[f"{rel(path)}#simulations[{index}]"],
            run_contract=run_contract,
        )


def add_attempt(attempts: dict[tuple[str, str], Attempt], attempt: Attempt) -> None:
    key = (attempt.harness, attempt.run_id)
    existing = attempts.get(key)
    if existing is None:
        attempts[key] = attempt
    else:
        existing.merge(attempt)


def group_by_task(attempts: Iterable[Attempt], tasks: dict[str, Json]) -> list[Json]:
    grouped: dict[str, dict[str, list[Attempt]]] = defaultdict(lambda: defaultdict(list))
    for attempt in attempts:
        grouped[attempt.task_id][attempt.harness].append(attempt)

    all_task_ids = sorted(set(tasks) | set(grouped))
    summaries = []
    for task_id in all_task_ids:
        task_info = tasks.get(task_id, {"task_id": task_id})
        custom = sorted(grouped[task_id].get("custom", []), key=lambda item: item.run_id)
        default = sorted(grouped[task_id].get("default", []), key=lambda item: item.run_id)
        summaries.append(
            {
                "task_id": task_id,
                "custom_pass_count": count_passes(custom),
                "custom_run_count": len(custom),
                "default_pass_count": count_passes(default),
                "default_run_count": len(default),
                "custom_failed_run_ids": failed_run_ids(custom),
                "default_failed_run_ids": failed_run_ids(default),
                "gold_expected_actions": task_info.get("gold_expected_actions") or [],
                "reward_basis": task_info.get("reward_basis") or [],
                "required_documents": task_info.get("required_documents") or [],
                "task_definition_path": task_info.get("definition_path"),
                "custom_runs": [attempt_to_json(attempt) for attempt in custom],
                "default_runs": [attempt_to_json(attempt) for attempt in default],
                "evidence_paths": sorted(
                    {
                        path
                        for attempt in custom + default
                        for path in attempt.evidence_paths
                    }
                ),
            }
        )
    return summaries


def count_passes(attempts: list[Attempt]) -> int:
    return sum(1 for attempt in attempts if attempt.passed)


def failed_run_ids(attempts: list[Attempt]) -> list[str]:
    return [attempt.run_id for attempt in attempts if not attempt.passed]


def attempt_to_json(attempt: Attempt) -> Json:
    return {
        "run_id": attempt.run_id,
        "source_label": attempt.source_label,
        "reward": attempt.reward,
        "passed": attempt.passed,
        "termination_reason": attempt.termination_reason,
        "returncode": attempt.returncode,
        "error_type": attempt.error_type,
        "observed_tool_calls": attempt.observed_tool_calls,
        "observed_messages": attempt.observed_messages,
        "event_counts": attempt.event_counts,
        "evidence_paths": attempt.evidence_paths,
        "run_contract": attempt.run_contract,
    }


def write_json(path: Path, summary: list[Json], inputs: Json) -> None:
    payload = {
        "schema_version": 1,
        "repo_root": str(REPO_ROOT),
        "inputs": inputs,
        "task_count": len(summary),
        "tasks": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, summary: list[Json]) -> None:
    fieldnames = [
        "task_id",
        "custom_pass_count",
        "custom_run_count",
        "default_pass_count",
        "default_run_count",
        "custom_failed_run_ids",
        "default_failed_run_ids",
        "gold_expected_actions",
        "reward_basis",
        "custom_rewards_by_run",
        "default_rewards_by_run",
        "custom_observed_tool_calls",
        "default_observed_tool_calls",
        "custom_observed_messages",
        "default_observed_messages",
        "evidence_paths",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "task_id": row["task_id"],
                    "custom_pass_count": row["custom_pass_count"],
                    "custom_run_count": row["custom_run_count"],
                    "default_pass_count": row["default_pass_count"],
                    "default_run_count": row["default_run_count"],
                    "custom_failed_run_ids": compact_json(row["custom_failed_run_ids"]),
                    "default_failed_run_ids": compact_json(row["default_failed_run_ids"]),
                    "gold_expected_actions": compact_json(row["gold_expected_actions"]),
                    "reward_basis": compact_json(row["reward_basis"]),
                    "custom_rewards_by_run": compact_json(rewards_by_run(row["custom_runs"])),
                    "default_rewards_by_run": compact_json(rewards_by_run(row["default_runs"])),
                    "custom_observed_tool_calls": compact_json(tool_calls_by_run(row["custom_runs"])),
                    "default_observed_tool_calls": compact_json(tool_calls_by_run(row["default_runs"])),
                    "custom_observed_messages": compact_json(messages_by_run(row["custom_runs"])),
                    "default_observed_messages": compact_json(messages_by_run(row["default_runs"])),
                    "evidence_paths": compact_json(row["evidence_paths"]),
                }
            )


def rewards_by_run(runs: list[Json]) -> dict[str, float | None]:
    return {run["run_id"]: run.get("reward") for run in runs}


def tool_calls_by_run(runs: list[Json]) -> dict[str, list[Json]]:
    return {run["run_id"]: run.get("observed_tool_calls") or [] for run in runs}


def messages_by_run(runs: list[Json]) -> dict[str, list[Json]]:
    return {run["run_id"]: run.get("observed_messages") or [] for run in runs}


def main() -> int:
    args = parse_args()
    tasks = load_task_definitions(args.tasks_dir)

    attempts: dict[tuple[str, str], Attempt] = {}
    artifact_paths = custom_artifact_paths(args)
    default_paths = default_result_paths(args)

    for path in artifact_paths:
        for attempt in attempts_from_custom_artifact(path):
            add_attempt(attempts, attempt)

    if not args.no_default_scans:
        for attempt in attempts_from_custom_runs(
            args.custom_runs_dir, args.custom_source_prefix
        ):
            add_attempt(attempts, attempt)

    for path in default_paths:
        for attempt in attempts_from_default_results(path):
            add_attempt(attempts, attempt)

    summary = group_by_task(attempts.values(), tasks)
    inputs = {
        "custom_artifacts": [rel(path) for path in artifact_paths],
        "custom_runs_dir": rel(args.custom_runs_dir),
        "custom_source_prefixes": args.custom_source_prefix,
        "default_results": [rel(path) for path in default_paths],
        "default_source_prefixes": args.default_source_prefix,
        "tasks_dir": rel(args.tasks_dir),
    }
    write_json(args.output_json, summary, inputs)
    write_csv(args.output_csv, summary)
    print(f"wrote {rel(args.output_json)}")
    print(f"wrote {rel(args.output_csv)}")
    print(f"tasks={len(summary)} attempts={len(attempts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
