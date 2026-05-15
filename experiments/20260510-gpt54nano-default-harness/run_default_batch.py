#!/usr/bin/env python3
"""Run Tau's default banking harness for GPT-5.4 nano."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "azure/gpt-5.4-nano"
DEFAULT_SAVE_TO = "gpt54nano_default_banking_bm25_1trial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task ids for smoke/debug runs. Omit to run all banking tasks.",
    )
    parser.add_argument("--save-to", default=DEFAULT_SAVE_TO)
    parser.add_argument("--retrieval-config", default="bm25")
    parser.add_argument("--max-concurrency", type=int, default=97)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--reasoning-enabled", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument("--llm-log-mode", choices=["all", "latest"], default="all")
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.replace("export ", "").strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def patched_env(env_file: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(env_file))
    if env.get("AZURE_OPENAI_API_KEY") and not env.get("AZURE_API_KEY"):
        env["AZURE_API_KEY"] = env["AZURE_OPENAI_API_KEY"]
    if env.get("AZURE_OPENAI_ENDPOINT") and not env.get("AZURE_API_BASE"):
        env["AZURE_API_BASE"] = env["AZURE_OPENAI_ENDPOINT"]
    if env.get("AZURE_OPENAI_API_VERSION") and not env.get("AZURE_API_VERSION"):
        env["AZURE_API_VERSION"] = env["AZURE_OPENAI_API_VERSION"]
    return env


def require_env(env: dict[str, str]) -> None:
    missing = [
        name
        for name in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION")
        if not env.get(name)
    ]
    if missing:
        raise SystemExit("Missing Azure env vars: " + ", ".join(missing))


def require_model_env(model: str, env: dict[str, str]) -> None:
    if model.startswith("azure/"):
        require_env(env)
        return
    if model.startswith("openrouter/") and not env.get("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY for OpenRouter model run")


def llm_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    elif args.reasoning_enabled:
        payload["reasoning"] = {"enabled": True}
    return payload


def results_path(save_to: str) -> Path:
    return REPO_ROOT / "data" / "simulations" / save_to / "results.json"


def score_results(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    simulations = data.get("simulations", [])
    rewards = [(sim.get("reward_info") or {}).get("reward") for sim in simulations]
    finished_rewards = [reward for reward in rewards if isinstance(reward, int | float)]
    successes = [
        reward for reward in finished_rewards if abs(float(reward) - 1.0) < 1e-6
    ]
    infra_errors = [
        sim
        for sim in simulations
        if sim.get("termination_reason") == "infrastructure_error"
    ]
    return {
        "results_path": str(path),
        "task_count": len(data.get("tasks", [])),
        "simulation_count": len(simulations),
        "scored_simulation_count": len(finished_rewards),
        "success_count": len(successes),
        "avg_reward": (
            sum(float(reward) for reward in finished_rewards) / len(finished_rewards)
        )
        if finished_rewards
        else 0.0,
        "score_percent": (100.0 * len(successes) / len(finished_rewards))
        if finished_rewards
        else 0.0,
        "infra_error_count": len(infra_errors),
    }


def main() -> int:
    args = parse_args()
    env = patched_env(args.env_file)
    require_model_env(args.model, env)
    model_args = llm_args(args)
    command = [
        "uv",
        "run",
        "--extra",
        "knowledge",
        "tau2",
        "run",
        "--domain",
        "banking_knowledge",
        "--retrieval-config",
        args.retrieval_config,
        "--agent-llm",
        args.model,
        "--user-llm",
        args.model,
        "--agent-llm-args",
        json.dumps(model_args),
        "--user-llm-args",
        json.dumps(model_args),
        "--num-trials",
        "1",
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-steps",
        str(args.max_steps),
        "--max-errors",
        str(args.max_errors),
        "--timeout",
        str(args.timeout),
        "--seed",
        str(args.seed),
        "--save-to",
        args.save_to,
        "--log-level",
        "ERROR",
    ]
    if args.verbose_logs:
        command.extend(["--verbose-logs", "--llm-log-mode", args.llm_log_mode])
    if args.tasks:
        command.extend(["--task-ids", *args.tasks])
    if args.auto_resume:
        command.append("--auto-resume")

    print(" ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_path = EXPERIMENT_DIR / f"{args.save_to}.log"
    output_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout[-5000:], flush=True)
    if completed.returncode != 0:
        print(f"log={output_path}", flush=True)
        return completed.returncode

    score = score_results(results_path(args.save_to))
    score_path = EXPERIMENT_DIR / f"{args.save_to}_score.json"
    score_path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(score, indent=2), flush=True)
    print(f"log={output_path}", flush=True)
    print(f"score={score_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
