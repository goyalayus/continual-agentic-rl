#!/usr/bin/env python3
"""Check whether the benchmark comparison has enough scored attempts."""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SUMMARY_PATH = EXPERIMENT_DIR / "comparison_summary.json"
TASKS_DIR = REPO_ROOT / "data/tau2/domains/banking_knowledge/tasks"
EXPECTED_TASKS = 97
EXPECTED_RUNS_PER_HARNESS = 4
DEFAULT_REQUIRED_CUSTOM_PREFIX = "baseline_custom_azure_gpt55low_"
DEFAULT_REQUIRED_DEFAULT_PREFIX = "baseline_default_tau_bm25_azure_gpt55low_"
CUSTOM_RUN_SEEDS = (
    (1, 849558),
    (2, 551167),
    (3, 811445),
    (4, 613921),
)
EXPECTED_DEFAULT_SOURCE_LABEL = (
    "baseline_default_tau_bm25_azure_gpt55low_4trials_seed4101"
)
EXPECTED_CUSTOM_CONTRACT = {
    "model": "azure/gpt-5.5",
    "max_steps": 100,
    "max_errors": 10,
    "max_tokens": 768,
    "temperature": 1.0,
    "reasoning_effort": "low",
    "subagent_delegation": "batch",
    "retrieval_mode": "hybrid",
    "auto_resume": True,
}
EXPECTED_DEFAULT_CONTRACT = {
    "save_to": EXPECTED_DEFAULT_SOURCE_LABEL,
    "domain_name": "banking_knowledge",
    "retrieval_config": "bm25",
    "retrieval_config_kwargs": None,
    "num_trials": 4,
    "max_steps": 100,
    "max_errors": 10,
    "seed": 4101,
    "agent_implementation": "llm_agent",
    "agent_llm": "azure/gpt-5.5",
    "agent_llm_args": {
        "temperature": 1.0,
        "max_tokens": 768,
        "reasoning_effort": "low",
    },
    "user_implementation": "user_simulator",
    "user_llm": "azure/gpt-5.5",
    "user_llm_args": {
        "temperature": 1.0,
        "max_tokens": 768,
        "reasoning_effort": "low",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument(
        "--custom-only",
        action="store_true",
        help="Only require the four scored custom runs per task.",
    )
    parser.add_argument(
        "--required-custom-prefix",
        default=DEFAULT_REQUIRED_CUSTOM_PREFIX,
        help="Require the summary's custom source prefixes to equal this value.",
    )
    parser.add_argument(
        "--required-default-prefix",
        default=DEFAULT_REQUIRED_DEFAULT_PREFIX,
        help="Require the summary's default source prefixes to equal this value.",
    )
    parser.add_argument(
        "--required-default-source-label",
        default=EXPECTED_DEFAULT_SOURCE_LABEL,
        help="Require default runs to come from this exact save_to/source label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print(f"missing summary: {args.summary}")
        return 2

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = payload.get("tasks") or []
    errors: list[str] = []
    inputs = payload.get("inputs") or {}

    check_source_prefixes(
        errors,
        label="custom",
        actual=inputs.get("custom_source_prefixes") or [],
        required=args.required_custom_prefix,
    )
    if not args.custom_only:
        check_source_prefixes(
            errors,
            label="default",
            actual=inputs.get("default_source_prefixes") or [],
            required=args.required_default_prefix,
        )

    if len(rows) != EXPECTED_TASKS:
        errors.append(f"expected {EXPECTED_TASKS} tasks, found {len(rows)}")
    check_task_coverage(errors, rows)

    for row in rows:
        task_id = row.get("task_id")
        custom_runs = row.get("custom_runs") or []
        default_runs = row.get("default_runs") or []
        check_run_count_and_ids(
            errors,
            task_id=task_id,
            harness="custom",
            runs=custom_runs,
            expected_count=EXPECTED_RUNS_PER_HARNESS,
        )
        custom_source_labels = {
            str(run.get("source_label") or "") for run in custom_runs
        }
        expected_custom_source_labels = custom_source_labels_for_prefix(
            args.required_custom_prefix
        )
        if custom_source_labels != expected_custom_source_labels:
            errors.append(
                f"{task_id}: invalid custom source labels: expected "
                f"{sorted(expected_custom_source_labels)!r}, "
                f"found {sorted(custom_source_labels)!r}"
            )
        check_custom_source_label_counts(
            errors,
            task_id=task_id,
            labels=[str(run.get("source_label") or "") for run in custom_runs],
            expected_labels=expected_custom_source_labels,
        )
        for run in custom_runs:
            reason = invalid_run_reason(run)
            if reason:
                errors.append(f"{task_id}: invalid custom run {run.get('run_id')}: {reason}")
            contract_reason = invalid_contract_reason(
                run.get("run_contract") or {}, EXPECTED_CUSTOM_CONTRACT
            )
            if contract_reason:
                errors.append(
                    f"{task_id}: invalid custom contract {run.get('run_id')}: "
                    f"{contract_reason}"
                )
        if not args.custom_only:
            check_run_count_and_ids(
                errors,
                task_id=task_id,
                harness="default",
                runs=default_runs,
                expected_count=EXPECTED_RUNS_PER_HARNESS,
            )
            default_source_labels = {
                str(run.get("source_label") or "") for run in default_runs
            }
            if default_source_labels != {args.required_default_source_label}:
                errors.append(
                    f"{task_id}: invalid default source labels: expected "
                    f"{[args.required_default_source_label]!r}, "
                    f"found {sorted(default_source_labels)!r}"
                )
            default_label_count = sum(
                1
                for run in default_runs
                if run.get("source_label") == args.required_default_source_label
            )
            if default_label_count != EXPECTED_RUNS_PER_HARNESS:
                errors.append(
                    f"{task_id}: expected {EXPECTED_RUNS_PER_HARNESS} default runs "
                    f"with source label {args.required_default_source_label!r}, "
                    f"found {default_label_count}"
                )
            for run in default_runs:
                reason = invalid_run_reason(run)
                if reason:
                    errors.append(
                        f"{task_id}: invalid default run {run.get('run_id')}: {reason}"
                    )
                contract_reason = invalid_contract_reason(
                    run.get("run_contract") or {},
                    expected_default_contract(args.required_default_source_label),
                )
                if contract_reason:
                    errors.append(
                        f"{task_id}: invalid default contract {run.get('run_id')}: "
                        f"{contract_reason}"
                    )
        custom_scored = [run for run in custom_runs if run.get("reward") is not None]
        default_scored = [run for run in default_runs if run.get("reward") is not None]
        if len(custom_scored) != EXPECTED_RUNS_PER_HARNESS:
            errors.append(
                f"{task_id}: expected {EXPECTED_RUNS_PER_HARNESS} scored custom runs, "
                f"found {len(custom_scored)}"
            )
        if not args.custom_only and len(default_scored) != EXPECTED_RUNS_PER_HARNESS:
            errors.append(
                f"{task_id}: expected {EXPECTED_RUNS_PER_HARNESS} scored default runs, "
                f"found {len(default_scored)}"
            )

    if errors:
        print("incomplete")
        for error in errors[:40]:
            print(f"- {error}")
        if len(errors) > 40:
            print(f"- ... {len(errors) - 40} more")
        return 1

    custom_passes = sum(row.get("custom_pass_count") or 0 for row in rows)
    default_passes = sum(row.get("default_pass_count") or 0 for row in rows)
    total = EXPECTED_TASKS * EXPECTED_RUNS_PER_HARNESS
    print("complete")
    print(f"custom: {custom_passes}/{total} = {custom_passes / total:.4f}")
    if not args.custom_only:
        print(f"default: {default_passes}/{total} = {default_passes / total:.4f}")
    return 0


def check_source_prefixes(
    errors: list[str], *, label: str, actual: list[str], required: str
) -> None:
    if actual != [required]:
        errors.append(
            f"invalid {label} source prefixes: expected {[required]!r}, found {actual!r}"
        )


def custom_source_labels_for_prefix(prefix: str) -> set[str]:
    return {f"{prefix}r{run_index}_s{seed}" for run_index, seed in CUSTOM_RUN_SEEDS}


def expected_default_contract(source_label: str) -> dict:
    expected = dict(EXPECTED_DEFAULT_CONTRACT)
    expected["save_to"] = source_label
    return expected


def expected_task_ids() -> list[str]:
    return sorted(path.stem for path in TASKS_DIR.glob("task_*.json"))


def check_task_coverage(errors: list[str], rows: list[dict]) -> None:
    expected_ids = set(expected_task_ids())
    if len(expected_ids) != EXPECTED_TASKS:
        errors.append(
            f"expected {EXPECTED_TASKS} task definition files, found {len(expected_ids)}"
        )
        return

    task_ids = [str(row.get("task_id") or "") for row in rows]
    counts = Counter(task_ids)
    duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
    actual_ids = set(task_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    if duplicates:
        errors.append("duplicate task rows: " + ", ".join(duplicates[:20]))
    if missing:
        errors.append("missing task rows: " + ", ".join(missing[:20]))
    if extra:
        errors.append("unexpected task rows: " + ", ".join(extra[:20]))


def check_run_count_and_ids(
    errors: list[str],
    *,
    task_id: str,
    harness: str,
    runs: list[dict],
    expected_count: int,
) -> None:
    if len(runs) != expected_count:
        errors.append(
            f"{task_id}: expected {expected_count} total {harness} runs, found {len(runs)}"
        )
    run_ids = [str(run.get("run_id") or "") for run in runs]
    missing_run_ids = [index for index, run_id in enumerate(run_ids, start=1) if not run_id]
    if missing_run_ids:
        errors.append(
            f"{task_id}: {harness} runs missing run_id at positions "
            + ", ".join(str(index) for index in missing_run_ids[:20])
        )
    counts = Counter(run_id for run_id in run_ids if run_id)
    duplicates = sorted(run_id for run_id, count in counts.items() if count > 1)
    if duplicates:
        errors.append(
            f"{task_id}: duplicate {harness} run ids: " + ", ".join(duplicates[:20])
        )


def check_custom_source_label_counts(
    errors: list[str],
    *,
    task_id: str,
    labels: list[str],
    expected_labels: set[str],
) -> None:
    counts = Counter(labels)
    for label in sorted(expected_labels):
        if counts[label] != 1:
            errors.append(
                f"{task_id}: expected one custom run with source label {label!r}, "
                f"found {counts[label]}"
            )


def invalid_run_reason(run: dict) -> str | None:
    error_type = str(run.get("error_type") or "").strip().lower()
    termination_reason = str(run.get("termination_reason") or "").strip().lower()
    if error_type:
        return f"error_type={error_type}"
    if "infrastructure_error" in termination_reason:
        return f"termination_reason={termination_reason}"
    if "content_filter" in termination_reason or "cyber_policy" in termination_reason:
        return f"termination_reason={termination_reason}"
    return None


def invalid_contract_reason(actual: dict, expected: dict) -> str | None:
    if not actual:
        return "missing run_contract"
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not values_equal(actual_value, expected_value):
            mismatches.append(
                f"{key} expected {expected_value!r}, found {actual_value!r}"
            )
    if mismatches:
        return "; ".join(mismatches[:6])
    return None


def values_equal(actual, expected) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


if __name__ == "__main__":
    raise SystemExit(main())
