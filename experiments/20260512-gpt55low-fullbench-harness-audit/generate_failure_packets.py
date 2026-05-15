#!/usr/bin/env python3
"""Generate offline per-task review packets from comparison_summary.json.

This is deliberately read-only with respect to benchmark artifacts. It consumes
the summary created by analyze_comparison.py and writes audit packets inside this
experiment directory only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EXPERIMENT_DIR / "comparison_summary.json"
CHECK_COMPLETENESS = EXPERIMENT_DIR / "check_completeness.py"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "failure_packets"
INCOMPLETE_OUTPUT_DIR = EXPERIMENT_DIR / "failure_packets_incomplete"
EXPECTED_RUNS_PER_HARNESS = 4

Json = dict[str, Any]


@dataclass(frozen=True)
class Completeness:
    complete: bool
    returncode: int
    output: str

    @property
    def status(self) -> str:
        return "complete" if self.complete else "incomplete"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-task human/Codex review packets from comparison_summary.json."
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="comparison_summary.json produced by analyze_comparison.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Packet output directory. Defaults to failure_packets, or "
        "failure_packets_incomplete with --allow-incomplete.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Packet format to write.",
    )
    parser.add_argument(
        "--include",
        choices=("custom-failures", "all"),
        default="all",
        help="Which tasks should get packets.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write clearly marked incomplete packets even if check_completeness.py fails.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing packet directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def run_completeness_check(summary_path: Path) -> Completeness:
    result = subprocess.run(
        ["python3", str(CHECK_COMPLETENESS), "--summary", str(summary_path)],
        cwd=EXPERIMENT_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return Completeness(
        complete=result.returncode == 0,
        returncode=result.returncode,
        output=output.strip(),
    )


def repo_relative(path: Path) -> str:
    repo_root = EXPERIMENT_DIR.parents[1]
    try:
        return str(path.resolve().relative_to(repo_root))
    except (OSError, ValueError):
        return str(path)


def selected_rows(rows: list[Json], include: str) -> list[Json]:
    if include == "all":
        return rows
    return [
        row
        for row in rows
        if scored_pass_count(row.get("custom_runs") or []) < EXPECTED_RUNS_PER_HARNESS
    ]


def scored_pass_count(runs: list[Json]) -> int:
    return sum(1 for run in runs if run.get("reward") == 1.0)


def scored_count(runs: list[Json]) -> int:
    return sum(1 for run in runs if run.get("reward") is not None)


def failed_runs(runs: list[Json]) -> list[Json]:
    return [run for run in runs if run.get("reward") != 1.0]


def tool_call_names(runs: list[Json]) -> list[str]:
    names: list[str] = []
    for run in runs:
        for call in run.get("observed_tool_calls") or []:
            name = call.get("name")
            if name:
                names.append(str(name))
    return sorted(set(names))


def packet_payload(row: Json, completeness: Completeness) -> Json:
    custom_runs = row.get("custom_runs") or []
    default_runs = row.get("default_runs") or []
    custom_failed = failed_runs(custom_runs)
    default_failed = failed_runs(default_runs)
    return {
        "schema_version": 1,
        "packet_status": completeness.status,
        "task_id": row.get("task_id"),
        "anti_cheating_rules": [
            "Use this packet only for one-by-one audit after the full benchmark is complete.",
            "Do not write task-specific prompt fixes.",
            "Do not leak task ids, expected actions, reward labels, or hidden answers into prompts.",
            "If a fix is needed, keep it general and limited to prompts/tool descriptions.",
        ],
        "counts": {
            "custom_pass_count": row.get("custom_pass_count", 0),
            "custom_run_count": row.get("custom_run_count", len(custom_runs)),
            "custom_scored_count": scored_count(custom_runs),
            "default_pass_count": row.get("default_pass_count", 0),
            "default_run_count": row.get("default_run_count", len(default_runs)),
            "default_scored_count": scored_count(default_runs),
        },
        "expected_actions": row.get("gold_expected_actions") or [],
        "reward_basis": row.get("reward_basis") or [],
        "required_documents": row.get("required_documents") or [],
        "custom_failed_runs": [run_summary(run) for run in custom_failed],
        "custom_failed_run_ids": [run.get("run_id") for run in custom_failed],
        "observed_custom_tool_calls": {
            str(run.get("run_id")): run.get("observed_tool_calls") or []
            for run in custom_failed
        },
        "observed_custom_messages": {
            str(run.get("run_id")): run.get("observed_messages") or []
            for run in custom_failed
        },
        "observed_custom_tool_names": tool_call_names(custom_failed),
        "default_runs": [run_summary(run) for run in default_runs],
        "default_failed_run_ids": [run.get("run_id") for run in default_failed],
        "observed_default_messages": {
            str(run.get("run_id")): run.get("observed_messages") or []
            for run in default_runs
        },
        "evidence_paths": row.get("evidence_paths") or [],
        "task_definition_path": row.get("task_definition_path"),
        "audit_notes": {
            "exact_custom_failure_mode": "",
            "suspected_harness_behavior": "",
            "general_prompt_or_tool_description_fix_idea": "",
            "anti_cheating_check": "",
            "codex_review_confirmation": "",
            "reviewer_verdict": "",
        },
    }


def run_summary(run: Json) -> Json:
    return {
        "run_id": run.get("run_id"),
        "source_label": run.get("source_label"),
        "reward": run.get("reward"),
        "passed": run.get("passed"),
        "termination_reason": run.get("termination_reason"),
        "returncode": run.get("returncode"),
        "error_type": run.get("error_type"),
        "event_counts": run.get("event_counts") or {},
        "evidence_paths": run.get("evidence_paths") or [],
        "observed_tool_calls": run.get("observed_tool_calls") or [],
        "observed_messages": run.get("observed_messages") or [],
    }


def markdown_packet(packet: Json, completeness: Completeness) -> str:
    status = str(packet["packet_status"]).upper()
    title = f"# {status} Failure Review Packet: {packet['task_id']}"
    lines = [
        title,
        "",
        f"Packet status: `{packet['packet_status']}`",
    ]
    if not completeness.complete:
        lines.extend(
            [
                "",
                "> INCOMPLETE: check_completeness.py did not pass. Treat this as a dry-run packet only.",
                "",
                "```text",
                completeness.output,
                "```",
            ]
        )

    counts = packet["counts"]
    lines.extend(
        [
            "",
            "## Audit Guardrails",
            "",
            "- Audit one task at a time.",
            "- No task-specific fixes.",
            "- No prompt leaks from expected actions, reward labels, task ids, or hidden answers.",
            "- Fix ideas must stay general and limited to prompts/tool descriptions.",
            "",
            "## Counts",
            "",
            "| custom passes | custom scored | custom runs | default passes | default scored | default runs |",
            "| --- | --- | --- | --- | --- | --- |",
            "| "
            + " | ".join(
                str(value)
                for value in (
                    counts["custom_pass_count"],
                    counts["custom_scored_count"],
                    counts["custom_run_count"],
                    counts["default_pass_count"],
                    counts["default_scored_count"],
                    counts["default_run_count"],
                )
            )
            + " |",
            "",
            "## Expected Actions",
            "",
            fenced_json(packet["expected_actions"]),
            "",
            "## Reward Basis",
            "",
            fenced_json(packet["reward_basis"]),
            "",
            "## Custom Failed Runs",
            "",
            failed_runs_table(packet["custom_failed_runs"]),
            "",
            "## Observed Custom Tool Calls In Failed Runs",
            "",
            fenced_json(packet["observed_custom_tool_calls"]),
            "",
            "## Observed Custom Messages In Failed Runs",
            "",
            fenced_json(packet["observed_custom_messages"]),
            "",
            "## Default Runs",
            "",
            default_runs_table(packet["default_runs"]),
            "",
            "## Observed Default Messages",
            "",
            fenced_json(packet["observed_default_messages"]),
            "",
            "## Evidence Paths",
            "",
            evidence_list(packet),
            "",
            "## Human/Codex Audit Notes",
            "",
            "- Exact custom failure mode:",
            "- Suspected harness behavior:",
            "- General prompt/tool-description fix idea:",
            "- Anti-cheating check:",
            "- Codex review confirmation:",
            "- Reviewer verdict:",
            "",
        ]
    )
    return "\n".join(lines)


def fenced_json(value: Any) -> str:
    return "```json\n" + dump_json(value) + "\n```"


def failed_runs_table(runs: list[Json]) -> str:
    if not runs:
        return "_No custom failed runs in this packet._"
    lines = [
        "| run id | reward | termination | error type | observed tool names | evidence count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        names = sorted(
            {
                str(call.get("name"))
                for call in run.get("observed_tool_calls") or []
                if call.get("name")
            }
        )
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    run.get("run_id"),
                    run.get("reward"),
                    run.get("termination_reason"),
                    run.get("error_type"),
                    ", ".join(names),
                    len(run.get("evidence_paths") or []),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def default_runs_table(runs: list[Json]) -> str:
    if not runs:
        return "_No default runs found._"
    lines = [
        "| run id | reward | passed | termination | observed tool names | evidence count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        names = sorted(
            {
                str(call.get("name"))
                for call in run.get("observed_tool_calls") or []
                if call.get("name")
            }
        )
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    run.get("run_id"),
                    run.get("reward"),
                    run.get("passed"),
                    run.get("termination_reason"),
                    ", ".join(names),
                    len(run.get("evidence_paths") or []),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def evidence_list(packet: Json) -> str:
    paths = list(packet.get("evidence_paths") or [])
    task_path = packet.get("task_definition_path")
    if task_path and task_path not in paths:
        paths.insert(0, task_path)
    if not paths:
        return "_No evidence paths found._"
    return "\n".join(f"- `{path}`" for path in paths)


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(
                f"output directory already exists: {path}\n"
                "Use --force to overwrite generated packets."
            )
        for child in sorted(path.iterdir()):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                raise SystemExit(f"refusing to remove nested directory: {child}")
    path.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(path: Path) -> Path:
    resolved = path if path.is_absolute() else EXPERIMENT_DIR / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(EXPERIMENT_DIR)
    except ValueError as exc:
        raise SystemExit(
            f"output directory must stay inside this experiment directory: {EXPERIMENT_DIR}"
        ) from exc
    return resolved


def write_packets(
    rows: list[Json],
    output_dir: Path,
    packet_format: str,
    completeness: Completeness,
) -> list[str]:
    written: list[str] = []
    suffix = ".md" if packet_format == "markdown" else ".json"
    for row in rows:
        packet = packet_payload(row, completeness)
        task_id = str(packet["task_id"])
        path = output_dir / f"{task_id}{suffix}"
        if packet_format == "markdown":
            body = markdown_packet(packet, completeness)
        else:
            body = dump_json(packet) + "\n"
        path.write_text(body, encoding="utf-8")
        written.append(repo_relative(path))
    return written


def write_manifest(
    path: Path,
    summary_path: Path,
    packet_paths: list[str],
    completeness: Completeness,
    include: str,
    packet_format: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packet_status": completeness.status,
        "summary_path": repo_relative(summary_path),
        "include": include,
        "format": packet_format,
        "packet_count": len(packet_paths),
        "packet_paths": packet_paths,
        "completeness_check": {
            "command": f"python3 {repo_relative(CHECK_COMPLETENESS)}",
            "returncode": completeness.returncode,
            "output": completeness.output,
        },
        "anti_cheating_rules": [
            "No task-specific fixes.",
            "No prompt leaks from expected actions, reward labels, task ids, or hidden answers.",
            "Use packets for audit only; do not modify harness behavior from this script.",
        ],
    }
    path.write_text(dump_json(manifest) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.summary = args.summary.resolve()
    if not args.summary.exists():
        raise SystemExit(f"missing summary: {args.summary}")

    completeness = run_completeness_check(args.summary)
    if not completeness.complete and not args.allow_incomplete:
        print("refusing to generate final failure packets: check_completeness.py failed")
        print(completeness.output)
        print("\nUse --allow-incomplete to write clearly marked dry-run packets.")
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = INCOMPLETE_OUTPUT_DIR if not completeness.complete else DEFAULT_OUTPUT_DIR
    output_dir = resolve_output_dir(output_dir)
    prepare_output_dir(output_dir, args.force)

    payload = read_json(args.summary)
    rows = selected_rows(payload.get("tasks") or [], args.include)
    packet_paths = write_packets(rows, output_dir, args.format, completeness)
    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        args.summary,
        packet_paths,
        completeness,
        args.include,
        args.format,
    )

    print(f"status={completeness.status}")
    print(f"wrote {len(packet_paths)} packets")
    print(f"manifest={repo_relative(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
