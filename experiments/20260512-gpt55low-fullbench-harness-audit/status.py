#!/usr/bin/env python3
"""Print the current state of the harness-audit experiment."""

from __future__ import annotations

import json
import argparse
import os
import re
import subprocess
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EXPERIMENT_DIR / "comparison_summary.json"
WATCHER_LOG_PATH = EXPERIMENT_DIR / "key_provider_watcher.stdout.log"
EXPECTED_TASKS = 97
EXPECTED_PER_HARNESS = 4
WATCHER_UNIT = "tau2-openrouter-key-watch.service"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def openrouter_key_source() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "environment"
    if (EXPERIMENT_DIR / ".env.local").is_file():
        return ".env.local"
    return "missing"


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def key_provider_watcher_process_running() -> bool:
    marker = "experiments/20260512-gpt55low-fullbench-harness-audit/wait_for_key_and_provider_and_run.sh"
    output = run_text(["pgrep", "-af", marker])
    lines = [
        line
        for line in output.splitlines()
        if "pgrep" not in line and "status.py" not in line
    ]
    return bool(lines)


def user_linger_enabled() -> bool:
    user = os.environ.get("USER")
    if not user:
        return False
    output = run_text(["loginctl", "show-user", user, "-p", "Linger"])
    return output == "Linger=yes"


def key_provider_watcher_status() -> tuple[bool, str]:
    show_output = run_text(
        [
            "systemctl",
            "--user",
            "show",
            WATCHER_UNIT,
            "--property=ActiveState",
            "--property=FragmentPath",
            "--no-pager",
        ]
    )
    props = dict(
        line.split("=", 1)
        for line in show_output.splitlines()
        if "=" in line
    )
    systemd_active = props.get("ActiveState") == "active"
    process_running = key_provider_watcher_process_running()
    if not (systemd_active or process_running):
        return False, "not running"

    enabled_state = run_text(["systemctl", "--user", "is-enabled", WATCHER_UNIT])
    if enabled_state == "enabled":
        if user_linger_enabled():
            return True, "running (enabled user service; linger enabled)"
        return True, "running (enabled user service)"

    fragment_path = props.get("FragmentPath", "")
    if "transient" in fragment_path or fragment_path.startswith("/run/"):
        return True, "running (transient service)"
    return True, "running"


def sanitize_status_text(text: str) -> str:
    text = re.sub(r'("user_id"\s*:\s*")[^"]+(")', r"\1[REDACTED]\2", text)
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_OPENROUTER_KEY]", text)
    text = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[REDACTED_OPENAI_KEY]", text)
    return text


def last_provider_preflight_failure(log_path: Path = WATCHER_LOG_PATH) -> str:
    if not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""

    for line in reversed(lines):
        if line.startswith("preflight_failed "):
            return sanitize_status_text(line)
    return ""


def print_next_baseline_step() -> None:
    key_source = openrouter_key_source()
    watcher_running, watcher_label = key_provider_watcher_status()
    last_failure = last_provider_preflight_failure()
    print(f"openrouter key: {key_source}")
    print(f"key/provider watcher: {watcher_label}")
    if last_failure:
        label = "last provider preflight failure"
        if key_source == "missing":
            label += " (previous key attempt)"
        print(f"{label}: {last_failure}")
    if key_source == "missing":
        if watcher_running:
            print("next: run setup_openrouter_env.sh; the watcher will launch the baseline after preflight")
            print("direct: setup_and_run_full_baselines_openrouter.sh will stop the watcher after key entry")
        else:
            print("next: run setup_and_run_full_baselines_openrouter.sh")
            print("manual: run setup_openrouter_env.sh, then run run_full_baselines_openrouter.sh")
            print("watcher: start_key_provider_watcher.sh can wait for key/provider readiness")
    else:
        if watcher_running:
            print("next: watcher will launch run_full_baselines_openrouter.sh after provider preflight")
        else:
            print("next: run run_full_baselines_openrouter.sh")


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print("status: no comparison summary yet")
        print_next_baseline_step()
        return 2

    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = payload.get("tasks") or []
    expected_total = EXPECTED_TASKS * EXPECTED_PER_HARNESS

    custom_scored = 0
    default_scored = 0
    custom_unscored = 0
    default_unscored = 0
    custom_passes = 0
    default_passes = 0

    incomplete_tasks: list[str] = []
    for row in rows:
        custom_runs = row.get("custom_runs") or []
        default_runs = row.get("default_runs") or []
        task_custom_scored = [run for run in custom_runs if run.get("reward") is not None]
        task_default_scored = [run for run in default_runs if run.get("reward") is not None]
        custom_scored += len(task_custom_scored)
        default_scored += len(task_default_scored)
        custom_unscored += len(custom_runs) - len(task_custom_scored)
        default_unscored += len(default_runs) - len(task_default_scored)
        custom_passes += sum(1 for run in task_custom_scored if run.get("passed"))
        default_passes += sum(1 for run in task_default_scored if run.get("passed"))
        if (
            len(task_custom_scored) != EXPECTED_PER_HARNESS
            or len(task_default_scored) != EXPECTED_PER_HARNESS
        ):
            incomplete_tasks.append(row.get("task_id") or "unknown")

    print(f"tasks: {len(rows)}/{EXPECTED_TASKS}")
    print(f"custom scored: {custom_scored}/{expected_total}")
    print(f"default scored: {default_scored}/{expected_total}")
    print(f"custom unscored/infra: {custom_unscored}")
    print(f"default unscored/infra: {default_unscored}")
    if custom_scored:
        print(f"custom partial accuracy: {custom_passes}/{custom_scored} = {custom_passes / custom_scored:.4f}")
    if default_scored:
        print(f"default partial accuracy: {default_passes}/{default_scored} = {default_passes / default_scored:.4f}")

    if len(rows) != EXPECTED_TASKS or incomplete_tasks:
        print("status: incomplete")
        print(f"incomplete task count: {len(incomplete_tasks)}")
        print_next_baseline_step()
        return 1

    print("status: complete")
    print("next: begin one-by-one failure audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
