#!/usr/bin/env python3
"""Final objective audit for the full harness-improvement loop."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
TASKS_DIR = REPO_ROOT / "data/tau2/domains/banking_knowledge/tasks"

BASELINE_SUMMARY = EXPERIMENT_DIR / "comparison_summary.json"
FAILURE_AUDIT_MD = EXPERIMENT_DIR / "failure_audit.md"
FAILURE_AUDIT_JSON = EXPERIMENT_DIR / "failure_audit.json"
GENERIC_FIX_INPUT_MD = EXPERIMENT_DIR / "generic_fix_input.md"
GENERIC_FIX_INPUT_JSON = EXPERIMENT_DIR / "generic_fix_input.json"
POSTFIX_SUMMARY = EXPERIMENT_DIR / "postfix_comparison_summary.json"
BASELINE_LAUNCH_STATE = EXPERIMENT_DIR / "baseline_launch_state.json"
POSTFIX_LAUNCH_STATE = EXPERIMENT_DIR / "postfix_launch_state.json"

EXPECTED_TASKS = 97
EXPECTED_ATTEMPTS = 388
REQUIRED_AUDIT_NOTE_FIELDS = {
    "exact_custom_failure_mode",
    "suspected_harness_behavior",
    "general_prompt_or_tool_description_fix_idea",
    "anti_cheating_check",
    "codex_review_confirmation",
    "reviewer_verdict",
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


@dataclass
class Check:
    name: str
    passed: bool
    evidence: str


def run_command(args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return result.returncode, output.strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_counts(path: Path) -> tuple[int, int, int]:
    payload = load_json(path)
    rows = payload.get("tasks") or []
    custom_scored = sum(
        sum(1 for run in row.get("custom_runs") or [] if run.get("reward") is not None)
        for row in rows
    )
    custom_passes = sum(row.get("custom_pass_count") or 0 for row in rows)
    return len(rows), custom_scored, custom_passes


def check_baseline_complete() -> Check:
    if not BASELINE_SUMMARY.exists():
        return Check("baseline 4x custom/default run", False, "missing comparison_summary.json")
    code, output = run_command(
        [
            "python3",
            str(EXPERIMENT_DIR / "check_completeness.py"),
            "--summary",
            str(BASELINE_SUMMARY),
        ]
    )
    launch_errors = validate_launch_state(BASELINE_LAUNCH_STATE, "baseline")
    evidence = output
    if launch_errors:
        evidence += "\nlaunch_state_errors:\n" + "\n".join(
            f"- {error}" for error in launch_errors
        )
    return Check("baseline 4x custom/default run", code == 0 and not launch_errors, evidence)


def check_failure_audit() -> Check:
    if not FAILURE_AUDIT_JSON.exists() or not FAILURE_AUDIT_MD.exists():
        return Check(
            "per-task failure audit table",
            False,
            "missing failure_audit.md or failure_audit.json",
        )
    payload = load_json(FAILURE_AUDIT_JSON)
    packet_count = int(payload.get("packet_count") or 0)
    tasks = payload.get("tasks") or []
    errors = []
    if payload.get("schema_version") != 1:
        errors.append(f"schema_version={payload.get('schema_version')!r}")
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    task_counts = Counter(task_ids)
    duplicates = sorted(task_id for task_id, count in task_counts.items() if count > 1)
    expected_ids = set(expected_task_ids())
    actual_ids = set(task_ids)
    missing_task_ids = sorted(expected_ids - actual_ids)
    extra_task_ids = sorted(actual_ids - expected_ids)
    if duplicates:
        errors.append("duplicate task ids: " + ", ".join(duplicates[:20]))
    if missing_task_ids:
        errors.append("missing task ids: " + ", ".join(missing_task_ids[:20]))
    if extra_task_ids:
        errors.append("unexpected task ids: " + ", ".join(extra_task_ids[:20]))
    if len(expected_ids) != EXPECTED_TASKS:
        errors.append(f"expected_task_file_count={len(expected_ids)}")

    missing_notes = []
    placeholder_notes = []
    vague_notes = []
    for task in tasks:
        notes = task.get("audit_notes") or {}
        missing_fields = sorted(REQUIRED_AUDIT_NOTE_FIELDS - set(notes))
        blanks = [
            key
            for key in REQUIRED_AUDIT_NOTE_FIELDS
            if not str(notes.get(key) or "").strip()
        ]
        if missing_fields:
            missing_notes.append(f"{task.get('task_id')}: missing {', '.join(missing_fields)}")
        if blanks:
            missing_notes.append(f"{task.get('task_id')}: {', '.join(blanks)}")
        for key in REQUIRED_AUDIT_NOTE_FIELDS:
            normalized = str(notes.get(key) or "").strip().lower().strip(".")
            if normalized in PLACEHOLDER_NOTES:
                placeholder_notes.append(f"{task.get('task_id')}: {key}")
        for key in ("exact_custom_failure_mode", "suspected_harness_behavior"):
            note = str(notes.get(key) or "").strip()
            normalized = note.lower().strip(".")
            if normalized in VAGUE_DIAGNOSIS_NOTES or (note and len(note) < 40):
                vague_notes.append(f"{task.get('task_id')}: {key}={note!r}")
        confirmation = str(notes.get("codex_review_confirmation") or "").strip()
        if confirmation and (
            not re.search(r"\bcodex\b", confirmation, flags=re.I)
            or not re.search(
                r"\b(personally reviewed|reviewed this task|reviewed)\b",
                confirmation,
                flags=re.I,
            )
        ):
            vague_notes.append(
                f"{task.get('task_id')}: weak codex_review_confirmation={confirmation!r}"
            )
    passed = (
        packet_count == EXPECTED_TASKS
        and len(tasks) == EXPECTED_TASKS
        and not missing_notes
        and not placeholder_notes
        and not vague_notes
        and not errors
    )
    evidence = (
        f"packet_count={packet_count}, task_rows={len(tasks)}, "
        f"blank_note_rows={len(missing_notes)}, "
        f"placeholder_note_rows={len(placeholder_notes)}, "
        f"vague_note_rows={len(vague_notes)}, structural_errors={len(errors)}"
    )
    details = [*errors, *missing_notes[:20], *placeholder_notes[:20], *vague_notes[:20]]
    if details:
        evidence += "\n" + "\n".join(details[:60])
    return Check("per-task failure audit table", passed, evidence)


def check_generic_fix_input() -> Check:
    if not GENERIC_FIX_INPUT_JSON.exists() or not GENERIC_FIX_INPUT_MD.exists():
        return Check(
            "sanitized generic fix handoff",
            False,
            "missing generic_fix_input.md or generic_fix_input.json",
        )
    payload = load_json(GENERIC_FIX_INPUT_JSON)
    items = payload.get("items") or []
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    forbidden_markers = (
        '"task_id"',
        '"expected_actions"',
        '"reward_basis"',
        '"custom_failed_runs"',
        '"default_runs"',
        '"evidence_paths"',
        '"observed_messages"',
        '"observed_tool_calls"',
    )
    leaks = [marker for marker in forbidden_markers if marker in text]
    if re.search(r"\btask_\d{3}\b", text):
        leaks.append("task_id_pattern")
    if re.search(r"\btask\s+\d{3}\b", text, flags=re.I):
        leaks.append("task_number_pattern")
    if FAILURE_AUDIT_JSON.exists():
        task_data_leaks = task_data_terms_in_text(text, load_json(FAILURE_AUDIT_JSON))
        leaks.extend(f"task_data:{term}" for term in task_data_leaks[:10])
    passed = payload.get("schema_version") == 1 and bool(items) and not leaks
    evidence = (
        f"schema_version={payload.get('schema_version')!r}, "
        f"item_count={len(items)}, forbidden_marker_hits={leaks}"
    )
    return Check("sanitized generic fix handoff", passed, evidence)


def expected_task_ids() -> list[str]:
    return sorted(path.stem for path in TASKS_DIR.glob("task_*.json"))


def task_data_terms_in_text(text: str, audit_payload: dict) -> list[str]:
    normalized = text.lower()
    terms = blocked_terms_from_audit(audit_payload)
    return sorted({term for term in terms if term.lower() in normalized}, key=str.lower)


def blocked_terms_from_audit(payload: dict) -> set[str]:
    terms: set[str] = set()
    for task in payload.get("tasks") or []:
        task_id = task.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            terms.add(task_id.strip())
            match = re.fullmatch(r"task_(\d{3})", task_id.strip())
            if match:
                terms.add(f"task {match.group(1)}")
        collect_argument_terms(task.get("reward_basis"), terms)
        for action in task.get("expected_actions") or []:
            if not isinstance(action, dict):
                continue
            for key in ("action_id", "name"):
                value = action.get(key)
                if isinstance(value, str) and value.strip():
                    terms.add(value.strip())
            collect_argument_terms(action.get("arguments"), terms)
        for key in ("custom_failed_runs", "default_runs"):
            for run in task.get(key) or []:
                if not isinstance(run, dict):
                    continue
                for run_key in ("run_id", "source_label"):
                    value = run.get(run_key)
                    if isinstance(value, str) and value.strip():
                        terms.add(value.strip())
    return {term for term in terms if len(term) >= 6}


def collect_argument_terms(value, terms: set[str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            collect_argument_terms(child, terms)
        return
    if isinstance(value, list):
        for child in value:
            collect_argument_terms(child, terms)
        return
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 6:
            terms.add(text)
        return
    if isinstance(value, (int, float)):
        text = str(value)
        if len(text) >= 6:
            terms.add(text)


def check_fix_branch() -> Check:
    code, branch = run_command(["git", "branch", "--show-current"])
    if code != 0:
        return Check("new prompt/tool-description fix branch", False, branch)
    branch = branch.strip()
    if branch in {"", "main", "master"}:
        return Check(
            "new prompt/tool-description fix branch",
            False,
            f"current branch is {branch!r}; expected a non-main fix branch",
        )
    code, output = run_command(
        [
            "python3",
            str(EXPERIMENT_DIR / "check_prompt_only_diff.py"),
            "--base",
            "main",
        ]
    )
    return Check("new prompt/tool-description fix branch", code == 0, output)


def check_postfix_complete() -> Check:
    if not POSTFIX_SUMMARY.exists():
        return Check("post-fix 4x custom rerun", False, "missing postfix_comparison_summary.json")
    code, output = run_command(
        [
            "python3",
            str(EXPERIMENT_DIR / "check_completeness.py"),
            "--summary",
            str(POSTFIX_SUMMARY),
            "--custom-only",
            "--required-custom-prefix",
            "postfix_custom_azure_gpt55low_",
        ]
    )
    launch_errors = validate_launch_state(POSTFIX_LAUNCH_STATE, "postfix")
    evidence = output
    if launch_errors:
        evidence += "\nlaunch_state_errors:\n" + "\n".join(
            f"- {error}" for error in launch_errors
        )
    return Check("post-fix 4x custom rerun", code == 0 and not launch_errors, evidence)


def validate_launch_state(path: Path, expected_label: str) -> list[str]:
    if not path.exists():
        return [f"missing launch state: {path}"]
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"invalid launch state JSON: {exc}"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"launch state schema_version={payload.get('schema_version')!r}")
    if payload.get("label") != expected_label:
        errors.append(
            f"launch state label expected {expected_label!r}, found {payload.get('label')!r}"
        )
    if not payload.get("head_commit"):
        errors.append("launch state missing head_commit")
    if "status_short" not in payload or not isinstance(payload.get("status_short"), list):
        errors.append("launch state missing status_short list")
    return errors


def check_accuracy_measurement() -> Check:
    if not POSTFIX_SUMMARY.exists():
        return Check("post-fix accuracy measurement", False, "missing postfix summary")
    task_count, custom_scored, custom_passes = summary_counts(POSTFIX_SUMMARY)
    passed = task_count == EXPECTED_TASKS and custom_scored == EXPECTED_ATTEMPTS
    evidence = (
        f"tasks={task_count}, custom_scored={custom_scored}, "
        f"custom_accuracy={custom_passes}/{custom_scored if custom_scored else EXPECTED_ATTEMPTS}"
    )
    return Check("post-fix accuracy measurement", passed, evidence)


def main() -> int:
    checks = [
        check_baseline_complete(),
        check_failure_audit(),
        check_generic_fix_input(),
        check_fix_branch(),
        check_postfix_complete(),
        check_accuracy_measurement(),
    ]
    for check in checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"[{mark}] {check.name}")
        if check.evidence:
            print(check.evidence)
        print()
    if all(check.passed for check in checks):
        print("completion_audit: complete")
        return 0
    print("completion_audit: incomplete")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
