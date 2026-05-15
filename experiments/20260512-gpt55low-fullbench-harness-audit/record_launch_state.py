#!/usr/bin/env python3
"""Record the non-secret repo state used for a benchmark launch."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Launch label, such as baseline or postfix.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(path)


def resolve_output(path: Path) -> Path:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved = resolved.resolve()
    try:
        resolved.relative_to(EXPERIMENT_DIR)
    except ValueError as exc:
        raise SystemExit(
            f"output must stay inside experiment directory: {EXPERIMENT_DIR}"
        ) from exc
    return resolved


def launch_state(label: str, output: Path) -> dict:
    status_short = lines(git("status", "--short"))
    staged_files = lines(git("diff", "--cached", "--name-only"))
    unstaged_files = lines(git("diff", "--name-only"))
    untracked_files = lines(git("ls-files", "--others", "--exclude-standard"))
    return {
        "schema_version": 1,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_path": repo_relative(output),
        "branch": git("branch", "--show-current"),
        "head_commit": git("rev-parse", "HEAD"),
        "dirty": bool(status_short),
        "status_short": status_short,
        "staged_files": staged_files,
        "unstaged_files": unstaged_files,
        "untracked_files": untracked_files,
    }


def main() -> int:
    args = parse_args()
    output = resolve_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = launch_state(args.label, output)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(f"launch_state={repo_relative(output)}")
    print(f"branch={payload['branch']}")
    print(f"head_commit={payload['head_commit']}")
    print(f"dirty={payload['dirty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
