#!/usr/bin/env python3
"""Compile reviewed failure packets into the final offline audit artifacts.

This script is intentionally boring: it reads reviewer-written notes from the
generated Markdown packets and writes only failure_audit.md plus
failure_audit.json inside this experiment directory. It does not import or
modify any harness code.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKETS_DIR = EXPERIMENT_DIR / "failure_packets"
DEFAULT_MARKDOWN_OUTPUT = EXPERIMENT_DIR / "failure_audit.md"
DEFAULT_JSON_OUTPUT = EXPERIMENT_DIR / "failure_audit.json"
DEFAULT_FIX_INPUT_MARKDOWN = EXPERIMENT_DIR / "generic_fix_input.md"
DEFAULT_FIX_INPUT_JSON = EXPERIMENT_DIR / "generic_fix_input.json"
EXPECTED_TASKS = 97

Json = dict[str, Any]

NOTE_FIELDS = {
    "Exact custom failure mode": "exact_custom_failure_mode",
    "Suspected harness behavior": "suspected_harness_behavior",
    "General prompt/tool-description fix idea": (
        "general_prompt_or_tool_description_fix_idea"
    ),
    "Anti-cheating check": "anti_cheating_check",
    "Codex review confirmation": "codex_review_confirmation",
    "Reviewer verdict": "reviewer_verdict",
}

PLACEHOLDER_NOTES = {
    "",
    "todo",
    "tbd",
    "n/a",
    "na",
    "none",
    "blank",
    "fill me",
    "fill in",
}

FIX_NOTE_FIELDS = (
    "general_prompt_or_tool_description_fix_idea",
    "anti_cheating_check",
)
DIAGNOSIS_FIELDS = (
    "exact_custom_failure_mode",
    "suspected_harness_behavior",
)
CODEX_REVIEW_FIELD = "codex_review_confirmation"
VAGUE_DIAGNOSIS_NOTES = {
    "generic issue",
    "needs investigation",
    "same as above",
    "unclear",
    "unknown",
    "not sure",
    "todo",
    "tbd",
}
NO_FAILURE_PATTERNS = (
    "no custom failure",
    "all custom runs passed",
    "passed all custom runs",
    "custom passed",
)


@dataclass(frozen=True)
class Packet:
    path: Path
    task_id: str
    packet_status: str
    counts: Json
    expected_actions: Any
    reward_basis: Any
    custom_failed_runs: list[Json]
    default_runs: list[Json]
    evidence_paths: list[str]
    audit_notes: Json

    @property
    def relative_path(self) -> str:
        return repo_relative(self.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile reviewed failure packet Markdown files."
    )
    parser.add_argument(
        "--packets-dir",
        type=Path,
        default=DEFAULT_PACKETS_DIR,
        help="Directory created by generate_failure_packets.py.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Final Markdown audit path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Final machine-readable audit path.",
    )
    parser.add_argument(
        "--fix-input-markdown-output",
        type=Path,
        default=DEFAULT_FIX_INPUT_MARKDOWN,
        help="Sanitized Markdown handoff for prompt/tool-description fixes.",
    )
    parser.add_argument(
        "--fix-input-json-output",
        type=Path,
        default=DEFAULT_FIX_INPUT_JSON,
        help="Sanitized machine-readable handoff for prompt/tool-description fixes.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def repo_relative(path: Path) -> str:
    repo_root = EXPERIMENT_DIR.parents[1]
    try:
        return str(path.resolve().relative_to(repo_root))
    except (OSError, ValueError):
        return str(path)


def resolve_inside_experiment(path: Path, label: str) -> Path:
    resolved = path if path.is_absolute() else EXPERIMENT_DIR / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(EXPERIMENT_DIR)
    except ValueError as exc:
        raise SystemExit(
            f"{label} must stay inside this experiment directory: {EXPERIMENT_DIR}"
        ) from exc
    return resolved


def manifest_packet_paths(packets_dir: Path) -> list[Path]:
    manifest_path = packets_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing packet manifest: {manifest_path}\n"
            "Run generate_failure_packets.py after check_completeness.py passes."
        )

    manifest = read_json(manifest_path)
    if manifest.get("packet_status") != "complete":
        raise SystemExit(
            f"refusing incomplete packets: manifest packet_status="
            f"{manifest.get('packet_status')!r}"
        )
    if manifest.get("format") != "markdown":
        raise SystemExit(
            f"refusing non-Markdown packets: manifest format={manifest.get('format')!r}"
        )

    packet_paths: list[Path] = []
    for raw_path in manifest.get("packet_paths") or []:
        path = (EXPERIMENT_DIR.parents[1] / str(raw_path)).resolve()
        try:
            path.relative_to(packets_dir)
        except ValueError as exc:
            raise SystemExit(
                f"manifest packet path escapes packet dir: {raw_path}"
            ) from exc
        packet_paths.append(path)

    if not packet_paths:
        raise SystemExit("manifest has no packet paths")

    missing = [repo_relative(path) for path in packet_paths if not path.exists()]
    if missing:
        raise SystemExit(
            "missing packet files:\n" + "\n".join(f"- {p}" for p in missing)
        )

    actual = sorted(path.resolve() for path in packets_dir.glob("*.md"))
    expected = sorted(path.resolve() for path in packet_paths)
    extra = [repo_relative(path) for path in actual if path not in expected]
    if extra:
        raise SystemExit(
            "packet directory contains Markdown files not listed in manifest:\n"
            + "\n".join(f"- {path}" for path in extra)
        )

    return sorted(packet_paths, key=lambda path: path.name)


def parse_packet(path: Path) -> Packet:
    text = path.read_text(encoding="utf-8")
    task_id = parse_task_id(text, path)
    packet_status = parse_packet_status(text, path)
    if packet_status != "complete":
        raise SystemExit(
            f"refusing incomplete packet {repo_relative(path)}: "
            f"packet_status={packet_status!r}"
        )
    if "INCOMPLETE:" in text:
        raise SystemExit(
            f"refusing incomplete packet marker in {repo_relative(path)}"
        )

    expected_actions = parse_json_section(text, "Expected Actions", path)
    audit_notes = parse_audit_notes(text, path)
    validate_audit_notes(audit_notes, path)

    packet = Packet(
        path=path,
        task_id=task_id,
        packet_status=packet_status,
        counts=parse_counts(text, path),
        expected_actions=expected_actions,
        reward_basis=parse_json_section(text, "Reward Basis", path),
        custom_failed_runs=parse_table_section(text, "Custom Failed Runs"),
        default_runs=parse_table_section(text, "Default Runs"),
        evidence_paths=parse_evidence_paths(text),
        audit_notes=audit_notes,
    )
    validate_fix_notes_are_general(packet)
    validate_task_specific_diagnosis(packet)
    validate_codex_review_confirmation(packet)
    return packet


def parse_task_id(text: str, path: Path) -> str:
    match = re.search(
        r"^#\s+COMPLETE Failure Review Packet:\s+(\S+)\s*$",
        text,
        re.M,
    )
    if not match:
        raise SystemExit(f"cannot find task id in {repo_relative(path)}")
    return match.group(1)


def parse_packet_status(text: str, path: Path) -> str:
    match = re.search(r"^Packet status:\s+`([^`]+)`\s*$", text, re.M)
    if not match:
        raise SystemExit(f"cannot find packet status in {repo_relative(path)}")
    return match.group(1).strip()


def parse_json_section(text: str, heading: str, path: Path) -> Any:
    section = section_text(text, heading)
    match = re.search(r"```json\n(.*?)\n```", section, re.S)
    if not match:
        raise SystemExit(
            f"missing JSON block under {heading!r} in {repo_relative(path)}"
        )
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"invalid JSON block under {heading!r} in {repo_relative(path)}: {exc}"
        ) from exc


def parse_counts(text: str, path: Path) -> Json:
    section = section_text(text, "Counts")
    rows = [line for line in section.splitlines() if line.startswith("|")]
    if len(rows) < 3:
        raise SystemExit(f"missing counts table in {repo_relative(path)}")
    headers = split_table_row(rows[0])
    values = split_table_row(rows[2])
    if len(headers) != len(values):
        raise SystemExit(f"malformed counts table in {repo_relative(path)}")
    return {slug(header): parse_int(value) for header, value in zip(headers, values)}


def parse_table_section(text: str, heading: str) -> list[Json]:
    section = section_text(text, heading)
    rows = [line for line in section.splitlines() if line.startswith("|")]
    if len(rows) < 3:
        return []
    headers = [slug(header) for header in split_table_row(rows[0])]
    parsed: list[Json] = []
    for row in rows[2:]:
        values = split_table_row(row)
        parsed.append(dict(zip(headers, values, strict=False)))
    return parsed


def parse_evidence_paths(text: str) -> list[str]:
    section = section_text(text, "Evidence Paths")
    paths: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^-\s+`(.+)`\s*$", line)
        if match:
            paths.append(match.group(1))
    return paths


def parse_audit_notes(text: str, path: Path) -> Json:
    section = section_text(text, "Human/Codex Audit Notes")
    notes: dict[str, list[str]] = {key: [] for key in NOTE_FIELDS.values()}
    current_key: str | None = None

    labels = "|".join(re.escape(label) for label in NOTE_FIELDS)
    field_pattern = re.compile(rf"^-\s+({labels}):\s*(.*)$")

    for raw_line in section.splitlines():
        line = raw_line.rstrip()
        match = field_pattern.match(line)
        if match:
            current_key = NOTE_FIELDS[match.group(1)]
            if match.group(2).strip():
                notes[current_key].append(match.group(2).strip())
            continue
        if current_key and line.strip():
            notes[current_key].append(line.strip())

    missing_fields = [key for key, value in notes.items() if not value]
    if missing_fields:
        raise SystemExit(
            f"blank audit-note fields in {repo_relative(path)}: "
            + ", ".join(missing_fields)
        )
    return {key: "\n".join(value).strip() for key, value in notes.items()}


def validate_audit_notes(notes: Json, path: Path) -> None:
    bad_fields: list[str] = []
    for key, value in notes.items():
        normalized = str(value).strip().lower().strip(".")
        if normalized in PLACEHOLDER_NOTES:
            bad_fields.append(key)
    if bad_fields:
        raise SystemExit(
            f"placeholder audit-note fields in {repo_relative(path)}: "
            + ", ".join(bad_fields)
        )


def validate_fix_notes_are_general(packet: Packet) -> None:
    blocked_terms = packet_specific_blocked_terms(packet)
    bad_terms: dict[str, list[str]] = {}
    for field in FIX_NOTE_FIELDS:
        text = str(packet.audit_notes.get(field) or "")
        found = [
            term
            for term in blocked_terms
            if term and re.search(rf"\b{re.escape(term)}\b", text, flags=re.I)
        ]
        if found:
            bad_terms[field] = sorted(set(found), key=str.lower)
    if bad_terms:
        details = "; ".join(
            f"{field}: {', '.join(terms[:10])}" for field, terms in bad_terms.items()
        )
        raise SystemExit(
            f"task-specific terms in fix/anti-cheating notes for "
            f"{repo_relative(packet.path)}: {details}"
        )


def validate_task_specific_diagnosis(packet: Packet) -> None:
    for field in DIAGNOSIS_FIELDS:
        note = str(packet.audit_notes.get(field) or "").strip()
        normalized = note.lower().strip(".")
        if normalized in VAGUE_DIAGNOSIS_NOTES or len(note) < 40:
            raise SystemExit(
                f"vague {field} in {repo_relative(packet.path)}: {note!r}"
            )

    if packet.custom_failed_runs:
        exact = str(packet.audit_notes.get("exact_custom_failure_mode") or "")
        run_ids = [
            str(row.get("run_id") or "")
            for row in packet.custom_failed_runs
            if row.get("run_id")
        ]
        if run_ids and not any(run_id in exact for run_id in run_ids):
            raise SystemExit(
                f"exact_custom_failure_mode in {repo_relative(packet.path)} must "
                "name at least one failed custom run id"
            )
    else:
        combined = " ".join(
            str(packet.audit_notes.get(field) or "").lower()
            for field in DIAGNOSIS_FIELDS
        )
        if not any(pattern in combined for pattern in NO_FAILURE_PATTERNS):
            raise SystemExit(
                f"passing packet {repo_relative(packet.path)} must explicitly say "
                "there was no custom failure"
            )


def validate_codex_review_confirmation(packet: Packet) -> None:
    note = str(packet.audit_notes.get(CODEX_REVIEW_FIELD) or "").strip()
    if not re.search(r"\bcodex\b", note, flags=re.I):
        raise SystemExit(
            f"missing Codex review confirmation in {repo_relative(packet.path)}"
        )
    if not re.search(r"\b(personally reviewed|reviewed this task|reviewed)\b", note, flags=re.I):
        raise SystemExit(
            f"weak Codex review confirmation in {repo_relative(packet.path)}: {note!r}"
        )


def packet_specific_blocked_terms(packet: Packet) -> set[str]:
    terms = {packet.task_id}
    match = re.fullmatch(r"task_(\d{3})", packet.task_id)
    if match:
        terms.add(f"task {match.group(1)}")

    collect_argument_terms(packet.reward_basis, terms)
    for action in packet.expected_actions if isinstance(packet.expected_actions, list) else []:
        if not isinstance(action, dict):
            continue
        for key in ("action_id", "name"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                terms.add(value.strip())
        collect_argument_terms(action.get("arguments"), terms)
    for run in [*packet.custom_failed_runs, *packet.default_runs]:
        if not isinstance(run, dict):
            continue
        for key in ("run_id", "source_label"):
            value = run.get(key)
            if isinstance(value, str) and value.strip():
                terms.add(value.strip())
    return {term for term in terms if len(term) >= 4}


def collect_argument_terms(value: Any, terms: set[str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            collect_argument_terms(child, terms)
        return
    if isinstance(value, list):
        for child in value:
            collect_argument_terms(child, terms)
        return
    if isinstance(value, str):
        cleaned = value.strip()
        if len(cleaned) >= 6 and not cleaned.lower() in {"true", "false", "none"}:
            terms.add(cleaned)
        return
    if isinstance(value, (int, float)):
        text = str(value)
        if len(text) >= 6:
            terms.add(text)


def section_text(text: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.M)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s+", text[start:], re.M)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def split_table_row(row: str) -> list[str]:
    cells = row.strip().strip("|").split("|")
    return [cell.strip().replace("\\|", "|") for cell in cells]


def parse_int(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def audit_payload(packets: list[Packet], packets_dir: Path) -> Json:
    custom_passes = sum(
        int(packet.counts.get("custom_passes", 0)) for packet in packets
    )
    custom_runs = sum(int(packet.counts.get("custom_runs", 0)) for packet in packets)
    default_passes = sum(
        int(packet.counts.get("default_passes", 0)) for packet in packets
    )
    default_runs = sum(int(packet.counts.get("default_runs", 0)) for packet in packets)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packets_dir": repo_relative(packets_dir),
        "packet_count": len(packets),
        "guardrails": [
            "This is an offline compiler for reviewer-written notes only.",
            "It refuses incomplete packets and blank audit-note fields.",
            "It does not modify harness behavior.",
            "Fix ideas must be generic prompt/tool-description guidance.",
            "Do not leak task ids, expected actions, labels, or hidden answers.",
        ],
        "aggregate_counts": {
            "custom_passes": custom_passes,
            "custom_runs": custom_runs,
            "default_passes": default_passes,
            "default_runs": default_runs,
        },
        "tasks": [packet_to_json(packet) for packet in packets],
    }


def validate_packet_coverage(packets: list[Packet]) -> None:
    task_ids = [packet.task_id for packet in packets]
    unique_task_ids = sorted(set(task_ids))
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if len(packets) != EXPECTED_TASKS:
        raise SystemExit(f"expected {EXPECTED_TASKS} packets, found {len(packets)}")
    if len(unique_task_ids) != EXPECTED_TASKS:
        raise SystemExit(
            f"expected {EXPECTED_TASKS} unique task ids, found {len(unique_task_ids)}"
        )
    if duplicates:
        raise SystemExit("duplicate packet task ids: " + ", ".join(duplicates[:20]))


def packet_to_json(packet: Packet) -> Json:
    return {
        "task_id": packet.task_id,
        "packet_path": packet.relative_path,
        "packet_status": packet.packet_status,
        "counts": packet.counts,
        "expected_actions": packet.expected_actions,
        "reward_basis": packet.reward_basis,
        "custom_failed_runs": packet.custom_failed_runs,
        "default_runs": packet.default_runs,
        "evidence_paths": packet.evidence_paths,
        "audit_notes": packet.audit_notes,
    }


def markdown_audit(payload: Json) -> str:
    counts = payload["aggregate_counts"]
    lines = [
        "# Failure Audit",
        "",
        f"Generated at: `{payload['generated_at']}`",
        f"Packets: `{payload['packet_count']}` from `{payload['packets_dir']}`",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(f"- {rule}" for rule in payload["guardrails"])
    lines.extend(
        [
            "",
            "## Aggregate Counts",
            "",
            "| custom passes | custom runs | default passes | default runs |",
            "| --- | --- | --- | --- |",
            "| "
            + " | ".join(
                str(counts[key])
                for key in (
                    "custom_passes",
                    "custom_runs",
                    "default_passes",
                    "default_runs",
                )
            )
            + " |",
            "",
            "## Task Audits",
            "",
        ]
    )

    for task in payload["tasks"]:
        notes = task["audit_notes"]
        counts = task["counts"]
        lines.extend(
            [
                f"### {task['task_id']}",
                "",
                f"Packet: `{task['packet_path']}`",
                "",
                "| custom passes | custom runs | default passes | default runs |",
                "| --- | --- | --- | --- |",
                "| "
                + " | ".join(
                    str(counts.get(key, ""))
                    for key in (
                        "custom_passes",
                        "custom_runs",
                        "default_passes",
                        "default_runs",
                    )
                )
                + " |",
                "",
                f"- Exact custom failure mode: {notes['exact_custom_failure_mode']}",
                f"- Suspected harness behavior: {notes['suspected_harness_behavior']}",
                "- General prompt/tool-description fix idea: "
                + notes["general_prompt_or_tool_description_fix_idea"],
                f"- Anti-cheating check: {notes['anti_cheating_check']}",
                f"- Codex review confirmation: {notes['codex_review_confirmation']}",
                f"- Reviewer verdict: {notes['reviewer_verdict']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(payload: Json, markdown_output: Path, json_output: Path) -> None:
    markdown_output.write_text(markdown_audit(payload), encoding="utf-8")
    json_output.write_text(dump_json(payload) + "\n", encoding="utf-8")


def sanitized_fix_input_payload(payload: Json, source_audit: Path = DEFAULT_JSON_OUTPUT) -> Json:
    grouped: dict[tuple[str, str], int] = {}
    for task in payload.get("tasks") or []:
        notes = task.get("audit_notes") or {}
        fix = normalize_note(notes.get("general_prompt_or_tool_description_fix_idea"))
        anti_cheating = normalize_note(notes.get("anti_cheating_check"))
        if not fix:
            continue
        grouped[(fix, anti_cheating)] = grouped.get((fix, anti_cheating), 0) + 1

    items = [
        {
            "generic_fix_idea": fix,
            "anti_cheating_check": anti_cheating,
            "supporting_task_count": count,
        }
        for (fix, anti_cheating), count in sorted(
            grouped.items(), key=lambda item: (-item[1], item[0][0].lower())
        )
    ]
    return {
        "schema_version": 1,
        "generated_at": payload.get("generated_at"),
        "source_audit": repo_relative(source_audit),
        "purpose": (
            "Sanitized handoff for prompt/tool-description fixes. It intentionally "
            "omits task ids, expected actions, reward basis, evidence paths, "
            "message timelines, customer data, and hidden argument values."
        ),
        "guardrails": [
            "Use this file, not per-task packets, while editing prompts/tool descriptions.",
            "Do not add task-specific examples, task ids, expected actions, account ids, customer data, or hidden benchmark labels.",
            "Prompt/tool-description changes must describe general behavior only.",
        ],
        "item_count": len(items),
        "items": items,
    }


def normalize_note(value: Any) -> str:
    return " ".join(str(value or "").split())


def markdown_fix_input(payload: Json) -> str:
    lines = [
        "# Generic Fix Input",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "This is the only audit-derived artifact intended for prompt/tool-description editing.",
        "It omits task ids, expected actions, reward labels, evidence paths, message timelines, and customer data.",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(f"- {rule}" for rule in payload["guardrails"])
    lines.extend(["", "## Generic Fix Ideas", ""])
    for index, item in enumerate(payload.get("items") or [], start=1):
        lines.extend(
            [
                f"### Idea {index}",
                "",
                f"Supporting task count: `{item['supporting_task_count']}`",
                "",
                f"Fix idea: {item['generic_fix_idea']}",
                "",
                f"Anti-cheating check: {item['anti_cheating_check']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_fix_input_outputs(
    audit_payload_value: Json,
    source_audit: Path,
    markdown_output: Path,
    json_output: Path,
) -> None:
    payload = sanitized_fix_input_payload(audit_payload_value, source_audit)
    validate_sanitized_fix_input(payload)
    markdown_output.write_text(markdown_fix_input(payload), encoding="utf-8")
    json_output.write_text(dump_json(payload) + "\n", encoding="utf-8")


def validate_sanitized_fix_input(payload: Json) -> None:
    forbidden_keys = {
        "task_id",
        "expected_actions",
        "reward_basis",
        "custom_failed_runs",
        "default_runs",
        "evidence_paths",
        "observed_messages",
        "observed_tool_calls",
    }
    text = dump_json(payload)
    for key in forbidden_keys:
        if f'"{key}"' in text:
            raise SystemExit(f"sanitized fix input leaked forbidden key: {key}")


def main() -> int:
    args = parse_args()
    packets_dir = resolve_inside_experiment(args.packets_dir, "packets-dir")
    markdown_output = resolve_inside_experiment(args.markdown_output, "markdown-output")
    json_output = resolve_inside_experiment(args.json_output, "json-output")
    fix_input_markdown_output = resolve_inside_experiment(
        args.fix_input_markdown_output, "fix-input-markdown-output"
    )
    fix_input_json_output = resolve_inside_experiment(
        args.fix_input_json_output, "fix-input-json-output"
    )

    if not packets_dir.exists():
        raise SystemExit(f"missing packets directory: {packets_dir}")

    packet_paths = manifest_packet_paths(packets_dir)
    packets = [parse_packet(path) for path in packet_paths]
    validate_packet_coverage(packets)
    payload = audit_payload(packets, packets_dir)
    write_outputs(payload, markdown_output, json_output)
    write_fix_input_outputs(
        payload,
        json_output,
        fix_input_markdown_output,
        fix_input_json_output,
    )

    print(f"compiled {len(packets)} packets")
    print(f"markdown={repo_relative(markdown_output)}")
    print(f"json={repo_relative(json_output)}")
    print(f"fix_input_markdown={repo_relative(fix_input_markdown_output)}")
    print(f"fix_input_json={repo_relative(fix_input_json_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
