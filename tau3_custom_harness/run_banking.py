#!/usr/bin/env python3
"""Run banking tasks with the custom planner/subagent harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.utils.llm_utils import set_llm_log_dir, set_llm_log_mode

from tau3_custom_harness.agent import PlannerSubagentAgent
from tau3_custom_harness.logger import HarnessLogger
from tau3_custom_harness.retrieval import BankingHybridRetriever
from tau3_custom_harness.user import SafeUserSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="task_001")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-model", default=os.environ.get("TAU3_AGENT_MODEL", "gpt-4.1"))
    parser.add_argument("--user-model", default=os.environ.get("TAU3_USER_MODEL", "gpt-4.1"))
    parser.add_argument(
        "--subagent-model",
        default=os.environ.get("TAU3_SUBAGENT_MODEL"),
        help="Defaults to --agent-model.",
    )
    parser.add_argument(
        "--subagent-delegation",
        choices=["single", "batch"],
        default=os.environ.get("TAU3_SUBAGENT_DELEGATION", "batch"),
        help="Knowledge delegation surface exposed to the planner.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("TAU3_TEMPERATURE", "0.2")),
    )
    parser.add_argument(
        "--agent-llm-args-json",
        default=os.environ.get("TAU3_AGENT_LLM_ARGS_JSON"),
        help="Optional JSON object merged into planner LLM args.",
    )
    parser.add_argument(
        "--user-llm-args-json",
        default=os.environ.get("TAU3_USER_LLM_ARGS_JSON"),
        help="Optional JSON object merged into user-simulator LLM args.",
    )
    parser.add_argument(
        "--subagent-llm-args-json",
        default=os.environ.get("TAU3_SUBAGENT_LLM_ARGS_JSON"),
        help="Optional JSON object merged into KB subagent LLM args.",
    )
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "tau3_custom_harness_runs")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id supplied by a parent batch runner.",
    )
    parser.add_argument(
        "--s3-uri",
        default=os.environ.get("TAU3_TRACE_S3_URI"),
        help="Optional s3://bucket/prefix destination for the finished run folder.",
    )
    parser.add_argument(
        "--s3-strict",
        action="store_true",
        help="Fail the run if optional S3 upload fails.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional per-simulation wallclock timeout in seconds.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def llm_args_from_json(raw_json: str | None, *, temperature: float) -> dict:
    args = {"temperature": temperature}
    if not raw_json:
        return args
    try:
        loaded = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid LLM args JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit("LLM args JSON must be an object")
    args.update(loaded)
    return args


def safe_log_dict(value: dict) -> dict:
    redacted = {}
    for key, item in value.items():
        key_lower = key.lower()
        if any(secret_word in key_lower for secret_word in ("key", "token", "secret", "password")):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = item
    return redacted


def run_banking_task(
    *,
    task_id: str,
    max_steps: int,
    max_errors: int,
    seed: int,
    agent_model: str,
    user_model: str,
    subagent_model: str | None,
    subagent_delegation: str,
    temperature: float,
    agent_llm_args_json: str | None,
    user_llm_args_json: str | None,
    subagent_llm_args_json: str | None,
    log_dir: Path,
    run_id: str | None = None,
    s3_uri: str | None = None,
    s3_strict: bool = False,
    skip_eval: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one banking task in the current process."""
    task = next((item for item in get_tasks() if item.id == task_id), None)
    if task is None:
        available = ", ".join(t.id for t in get_tasks()[:10])
        raise ValueError(f"Unknown task id {task_id}. First available: {available}")

    logger = HarnessLogger(log_dir=log_dir, run_id=run_id)
    agent = None
    set_llm_log_dir(logger.run_dir / "llm_calls")
    set_llm_log_mode("all")
    try:
        agent_llm_args = llm_args_from_json(
            agent_llm_args_json, temperature=temperature
        )
        user_llm_args = llm_args_from_json(
            user_llm_args_json, temperature=temperature
        )
        subagent_llm_args = llm_args_from_json(
            subagent_llm_args_json, temperature=temperature
        )

        logger.log(
            "run_start",
            task_id=task.id,
            agent_model=agent_model,
            user_model=user_model,
            subagent_model=subagent_model or agent_model,
            subagent_delegation=subagent_delegation,
            agent_llm_args=safe_log_dict(agent_llm_args),
            user_llm_args=safe_log_dict(user_llm_args),
            subagent_llm_args=safe_log_dict(subagent_llm_args),
        )

        environment = get_environment(retrieval_variant="no_knowledge", task=task)
        retriever = BankingHybridRetriever(event_logger=logger.log)
        user_tools = environment.get_user_tools(include=task.user_tools)
        agent = PlannerSubagentAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=agent_model,
            llm_args=agent_llm_args,
            subagent_llm=subagent_model or agent_model,
            subagent_llm_args=subagent_llm_args,
            retriever=retriever,
            kb_document_count=len(retriever.docs),
            default_user_tools=user_tools,
            subagent_delegation=subagent_delegation,
            logger_=logger,
        )
        user = SafeUserSimulator(
            tools=user_tools or None,
            instructions=str(task.user_scenario),
            llm=user_model,
            llm_args=user_llm_args,
            logger_=logger,
        )
        orchestrator = Orchestrator(
            domain="banking_knowledge",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            simulation_id=logger.run_id,
            timeout=timeout,
            validate_communication=True,
        )

        env_kwargs = {"retrieval_variant": "no_knowledge", "task": task}
        if skip_eval:
            simulation = orchestrator.run()
            simulation.policy = orchestrator.environment.get_policy()
            simulation.reward_info = None
        else:
            simulation = run_simulation(
                orchestrator,
                evaluation_type=EvaluationType.ALL,
                env_kwargs=env_kwargs,
            )
        output_path = logger.write_json(
            "simulation.json", simulation.model_dump(mode="json")
        )
        evidence_path = logger.write_json(
            "kb_evidence.json", agent.knowledge_evidence_report()
        )
        result = {
            "run_id": logger.run_id,
            "task_id": task.id,
            "termination_reason": simulation.termination_reason,
            "reward": simulation.reward_info.reward if simulation.reward_info else None,
            "run_dir": str(logger.run_dir),
        }
        logger.write_json("result.json", result)
        logger.log(
            "run_done",
            task_id=task.id,
            termination_reason=simulation.termination_reason,
            reward=simulation.reward_info.reward if simulation.reward_info else None,
            output_path=str(output_path),
            evidence_path=str(evidence_path),
        )

        if s3_uri:
            logger.sync_to_s3(s3_uri, strict=s3_strict)
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        logger.log("run_error", task_id=task.id, error=str(exc), traceback=tb)
        error_payload = {
            "run_id": logger.run_id,
            "task_id": task.id,
            "error": str(exc),
            "traceback": tb,
        }
        if agent is not None:
            error_payload["kb_evidence"] = agent.knowledge_evidence_report()
        logger.write_json("run_error.json", error_payload)
        if s3_uri:
            logger.sync_to_s3(s3_uri, strict=False)
        raise
    finally:
        set_llm_log_dir(None)
        set_llm_log_mode("latest")


def main() -> int:
    args = parse_args()
    try:
        result = run_banking_task(
            task_id=args.task_id,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=args.seed,
            agent_model=args.agent_model,
            user_model=args.user_model,
            subagent_model=args.subagent_model,
            subagent_delegation=args.subagent_delegation,
            temperature=args.temperature,
            agent_llm_args_json=args.agent_llm_args_json,
            user_llm_args_json=args.user_llm_args_json,
            subagent_llm_args_json=args.subagent_llm_args_json,
            log_dir=args.log_dir,
            run_id=args.run_id,
            s3_uri=args.s3_uri,
            s3_strict=args.s3_strict,
            skip_eval=args.skip_eval,
            timeout=args.timeout,
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
