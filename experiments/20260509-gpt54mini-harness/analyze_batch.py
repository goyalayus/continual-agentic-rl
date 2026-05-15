#!/usr/bin/env python3
"""Summarize Tau3 custom harness batch artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        metadata = next((row for row in records if row.get("type") == "bench_run"), {})
        rows = [row for row in records if row.get("type") == "task_trace"]
        return metadata, rows

    batch = load_json(path)
    metadata = {
        "bench_run_id": batch.get("batch_name") or batch.get("bench_run_id"),
        "task_count": len(batch.get("rows", [])),
    }
    return metadata, batch["rows"]


def event_counts(run_dir: Path) -> Counter:
    counts = Counter()
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return counts
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        counts[row.get("event_type", "unknown")] += 1
    return counts


def classify_row(row: dict[str, Any]) -> str:
    if row.get("skipped_reason"):
        return f"not_run_{row['skipped_reason']}"
    if row.get("returncode") != 0:
        tail = row.get("output_tail") or row.get("stdout_tail") or ""
        if "rate limit" in tail.lower() or "429" in tail:
            return "provider_rate_limit"
        if "temperature" in tail and "UnsupportedParamsError" in tail:
            return "provider_temperature"
        if "Prompt tokens limit exceeded" in tail:
            return "provider_prompt_limit"
        lower_tail = tail.lower()
        if any(
            pattern in lower_tail
            for pattern in (
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
        return "harness_or_provider_error"
    result = row.get("result") or {}
    if result.get("reward") == 1.0:
        return "success"
    if result.get("reward") == 0.0:
        return "reward_failure"
    return "completed_unknown_reward"


def embedded_event_counts(row: dict[str, Any]) -> Counter:
    counts = Counter()
    for event in ((row.get("artifacts") or {}).get("events") or []):
        if isinstance(event, dict):
            counts[event.get("event_type", "unknown")] += 1
    return counts


def analysis_path(path: Path) -> Path:
    name = path.name
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    else:
        name = path.stem
    return EXPERIMENT_DIR / f"{name}_analysis.md"


def main() -> int:
    args = parse_args()
    metadata, rows = load_artifact(args.artifact)
    groups = defaultdict(list)
    for row in rows:
        groups[classify_row(row)].append(row)

    run_name = metadata.get("bench_run_id") or args.artifact.stem
    lines = [f"# {run_name}", ""]
    lines.append("## Summary")
    lines.append(f"- tasks: {metadata.get('task_count', len(rows))}")
    if metadata.get("parallelism") is not None:
        lines.append(f"- parallelism: {metadata['parallelism']}")
    if metadata.get("model"):
        lines.append(f"- model: {metadata['model']}")
    for label, label_rows in sorted(groups.items()):
        lines.append(f"- {label}: {len(label_rows)}")

    lines.append("")
    lines.append("## Rows")
    for row in rows:
        result = row.get("result") or {}
        run_dir = Path(result["run_dir"]) if result.get("run_dir") else None
        events = embedded_event_counts(row) or (event_counts(run_dir) if run_dir else Counter())
        event_text = ", ".join(f"{k}={v}" for k, v in events.most_common(6))
        lines.append(
            "- "
            f"{row['task_id']}: {classify_row(row)}, "
            f"reward={result.get('reward')}, "
            f"termination={result.get('termination_reason')}, "
            f"seconds={row.get('elapsed_seconds')}, "
            f"events=[{event_text}]"
        )
        if row.get("returncode") != 0:
            tail = (row.get("output_tail") or row.get("stdout_tail") or "").strip().splitlines()
            if tail:
                lines.append(f"  tail: {tail[-1][:500]}")

    output_path = analysis_path(args.artifact)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
