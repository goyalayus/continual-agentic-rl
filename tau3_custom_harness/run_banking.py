#!/usr/bin/env python3
"""Run banking tasks with the custom planner/subagent harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

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
        "--s3-uri",
        default=os.environ.get("TAU3_TRACE_S3_URI"),
        help="Optional s3://bucket/prefix destination for the finished run folder.",
    )
    parser.add_argument(
        "--s3-strict",
        action="store_true",
        help="Fail the run if optional S3 upload fails.",
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


def main() -> int:
    args = parse_args()
    task = next((item for item in get_tasks() if item.id == args.task_id), None)
    if task is None:
        available = ", ".join(t.id for t in get_tasks()[:10])
        raise SystemExit(f"Unknown task id {args.task_id}. First available: {available}")

    logger = HarnessLogger(log_dir=args.log_dir)
    agent = None
    set_llm_log_dir(logger.run_dir / "llm_calls")
    set_llm_log_mode("all")
    try:
        agent_llm_args = llm_args_from_json(
            args.agent_llm_args_json, temperature=args.temperature
        )
        user_llm_args = llm_args_from_json(
            args.user_llm_args_json, temperature=args.temperature
        )
        subagent_llm_args = llm_args_from_json(
            args.subagent_llm_args_json, temperature=args.temperature
        )

        logger.log(
            "run_start",
            task_id=task.id,
            agent_model=args.agent_model,
            user_model=args.user_model,
            subagent_model=args.subagent_model or args.agent_model,
            agent_llm_args=safe_log_dict(agent_llm_args),
            user_llm_args=safe_log_dict(user_llm_args),
            subagent_llm_args=safe_log_dict(subagent_llm_args),
        )

        environment = get_environment(retrieval_variant="no_knowledge", task=task)
        retriever = BankingHybridRetriever(event_logger=logger.log)
        agent = PlannerSubagentAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=args.agent_model,
            llm_args=agent_llm_args,
            subagent_llm=args.subagent_model or args.agent_model,
            subagent_llm_args=subagent_llm_args,
            retriever=retriever,
            logger_=logger,
        )
        user = SafeUserSimulator(
            tools=environment.get_user_tools(include=task.user_tools) or None,
            instructions=str(task.user_scenario),
            llm=args.user_model,
            llm_args=user_llm_args,
            logger_=logger,
        )
        orchestrator = Orchestrator(
            domain="banking_knowledge",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=args.seed,
            simulation_id=logger.run_id,
            validate_communication=True,
        )

        env_kwargs = {"retrieval_variant": "no_knowledge", "task": task}
        if args.skip_eval:
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
        print(json.dumps(result, indent=2), flush=True)

        if args.s3_uri:
            logger.sync_to_s3(args.s3_uri, strict=args.s3_strict)
        return 0
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
        if args.s3_uri:
            logger.sync_to_s3(args.s3_uri, strict=False)
        raise
    finally:
        set_llm_log_dir(None)
        set_llm_log_mode("latest")


if __name__ == "__main__":
    raise SystemExit(main())
