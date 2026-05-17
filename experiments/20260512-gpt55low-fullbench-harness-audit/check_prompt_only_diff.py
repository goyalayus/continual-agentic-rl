#!/usr/bin/env python3
"""Guard the later fix branch against accidental harness-code edits."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_JSON = EXPERIMENT_DIR / "failure_audit.json"
DEFAULT_FIX_INPUT_JSON = EXPERIMENT_DIR / "generic_fix_input.json"
ALLOWED_PATHS = {
    "custom_harness/tau3_custom_harness/prompts.py",
    "custom_harness/tau3_custom_harness/agent.py",
    "default_harness/src/tau2/domains/banking_knowledge/tools.py",
}

CODEISH_LINE = re.compile(
    r"^[+-]\s*("
    r"def\s+|class\s+|import\s+|from\s+|return\b|if\s+|elif\s+|else:|"
    r"for\s+|while\s+|with\s+|try:|except\s+|raise\s+|self\.|"
    r"[A-Za-z_][A-Za-z0-9_]*\s*=|[A-Za-z_][A-Za-z0-9_]*\("
    r")"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a fix branch only changed prompt/tool-description files. "
            "This is a guardrail, not a substitute for manual anti-cheating review."
        )
    )
    parser.add_argument(
        "--base",
        default="main",
        help="Base ref for git diff. Use the pre-fix branch/base commit.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_JSON,
        help=(
            "Optional failure_audit.json used to block exact task ids and "
            "expected-action argument values from added prompt text."
        ),
    )
    parser.add_argument(
        "--fix-input-json",
        type=Path,
        default=DEFAULT_FIX_INPUT_JSON,
        help=(
            "Sanitized generic_fix_input.json that should be used as the "
            "audit-derived input for prompt/tool-description edits."
        ),
    )
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    )


def changed_paths(base: str) -> list[str]:
    outputs = [
        git("diff", "--name-only", f"{base}..."),
        git("diff", "--cached", "--name-only"),
        git("diff", "--name-only"),
        untracked_guard_paths(),
    ]
    paths: set[str] = set()
    for output in outputs:
        paths.update(line.strip() for line in output.splitlines() if line.strip())
    return sorted(paths)


def untracked_guard_paths() -> str:
    output = git("ls-files", "--others", "--exclude-standard")
    guarded_prefixes = (
        "custom_harness/tau3_custom_harness/",
        "default_harness/src/tau2/domains/banking_knowledge/",
    )
    paths = [
        line
        for line in output.splitlines()
        if line.startswith(guarded_prefixes)
    ]
    return "\n".join(paths)


def diff_text(base: str) -> str:
    diffs = [
        git("diff", "--no-ext-diff", f"{base}...", "--", *sorted(ALLOWED_PATHS)),
        git("diff", "--cached", "--no-ext-diff", "--", *sorted(ALLOWED_PATHS)),
        git("diff", "--no-ext-diff", "--", *sorted(ALLOWED_PATHS)),
    ]
    return "\n".join(diff for diff in diffs if diff)


def suspicious_changed_lines(diff: str) -> list[str]:
    suspicious: list[str] = []
    for raw_line in diff.splitlines():
        if raw_line.startswith(("+++", "---", "@@", "diff --git", "index ")):
            continue
        if CODEISH_LINE.match(raw_line):
            suspicious.append(raw_line)
    return suspicious


def added_lines(diff: str) -> list[str]:
    lines: list[str] = []
    for raw_line in diff.splitlines():
        if raw_line.startswith(("+++", "---", "@@", "diff --git", "index ")):
            continue
        if raw_line.startswith("+"):
            lines.append(raw_line[1:])
    return lines


def blocked_terms_from_audit(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
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
            action_id = action.get("action_id")
            if isinstance(action_id, str) and action_id.strip():
                terms.add(action_id.strip())
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


def prompt_leak_lines(diff: str, audit_json: Path) -> list[str]:
    terms = blocked_terms_from_audit(audit_json)
    if not terms:
        return []
    leaked: list[str] = []
    for line in added_lines(diff):
        normalized = line.lower()
        matches = [term for term in terms if term.lower() in normalized]
        if matches:
            leaked.append(f"+{line}  <-- contains {matches[0]!r}")
    return leaked


def validate_fix_input(path: Path, audit_json: Path | None = None) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing sanitized fix input: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        errors.append(f"invalid fix input schema_version: {payload.get('schema_version')!r}")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        errors.append("sanitized fix input has no items")
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
    for marker in forbidden_markers:
        if marker in text:
            errors.append(f"sanitized fix input contains forbidden marker: {marker}")
    if re.search(r"\btask_\d{3}\b", text):
        errors.append("sanitized fix input contains a task id")
    if re.search(r"\btask\s+\d{3}\b", text, flags=re.I):
        errors.append("sanitized fix input contains a task-number reference")
    if audit_json is not None and audit_json.exists():
        leaked_terms = task_data_terms_in_text(text, audit_json)
        if leaked_terms:
            errors.append(
                "sanitized fix input contains task-specific data: "
                + ", ".join(repr(term) for term in leaked_terms[:10])
            )
    return errors


def task_data_terms_in_text(text: str, audit_json: Path) -> list[str]:
    normalized = text.lower()
    terms = blocked_terms_from_audit(audit_json)
    return sorted(
        {term for term in terms if term.lower() in normalized},
        key=str.lower,
    )


def main() -> int:
    args = parse_args()
    paths = changed_paths(args.base)
    outside = [path for path in paths if path not in ALLOWED_PATHS]
    if outside:
        print("failed: changed files outside prompt/tool-description allowlist")
        for path in outside:
            print(f"- {path}")
        print("\nallowed paths:")
        for path in sorted(ALLOWED_PATHS):
            print(f"- {path}")
        return 1

    diff = diff_text(args.base)
    suspicious = suspicious_changed_lines(diff)
    if suspicious:
        print("failed: diff contains code-looking changed lines")
        print("This may be a false positive, but it needs manual review.")
        for line in suspicious[:80]:
            print(line)
        if len(suspicious) > 80:
            print(f"... {len(suspicious) - 80} more")
        return 1

    leaks = prompt_leak_lines(diff, args.audit_json)
    if leaks:
        print("failed: diff appears to contain task-specific benchmark data")
        print(
            "Do not paste task ids, exact expected-action ids, user ids, emails, "
            "addresses, dates, or hidden arguments into prompts/tool descriptions."
        )
        for line in leaks[:80]:
            print(line)
        if len(leaks) > 80:
            print(f"... {len(leaks) - 80} more")
        return 1

    fix_input_errors = validate_fix_input(args.fix_input_json, args.audit_json)
    if fix_input_errors:
        print("failed: sanitized generic fix input is missing or unsafe")
        for error in fix_input_errors:
            print(f"- {error}")
        return 1

    print("passed: changed paths are limited to prompt/tool-description surfaces")
    print(f"sanitized fix input: {args.fix_input_json}")
    print("manual anti-cheating review is still required before rerun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
