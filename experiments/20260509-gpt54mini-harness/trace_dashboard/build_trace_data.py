#!/usr/bin/env python3
"""Build compact trace data for the static agentic trace dashboard."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "experiments"
    / "20260509-gpt54mini-harness"
    / "runs"
    / "gpt54nano_custom_hybrid_unlimited_task004_task006_20260511_task_006"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "task_006_trace.json"
TASKS_JSON = (
    REPO_ROOT
    / "data"
    / "tau2"
    / "domains"
    / "banking_knowledge"
    / "tasks.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_llm_calls(llm_dir: Path) -> list[dict[str, Any]]:
    calls = []
    for path in sorted(llm_dir.glob("*.json")):
        row = load_json(path)
        row["file"] = path.name
        calls.append(row)
    return calls


def usage_of(call: dict[str, Any]) -> dict[str, Any]:
    response = call.get("response") or {}
    usage = response.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost": float(response.get("cost") or 0.0),
        "reasoning_tokens": int(response.get("reasoning_tokens") or 0),
    }


def add_usage(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0)
        + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
        "cost": a.get("cost", 0.0) + b.get("cost", 0.0),
        "reasoning_tokens": a.get("reasoning_tokens", 0)
        + b.get("reasoning_tokens", 0),
    }


def compact_text(value: Any, limit: int = 900) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    text = " ".join(text.split()) if len(text) <= limit else text
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def message_brief(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": message.get("role"),
        "turn_idx": message.get("turn_idx"),
        "timestamp": message.get("timestamp"),
        "content": message.get("content"),
        "content_preview": compact_text(message.get("content"), 360),
        "tool_calls": message.get("tool_calls"),
        "requestor": message.get("requestor"),
        "error": message.get("error"),
        "usage": message.get("usage"),
        "cost": message.get("cost"),
    }


def request_messages(call: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    messages = (call.get("request") or {}).get("messages") or []
    visible = [m for m in messages if m.get("role") != "system"]
    return [
        {
            "role": m.get("role"),
            "content": m.get("content"),
            "content_preview": compact_text(m.get("content"), 420),
            "tool_calls": m.get("tool_calls"),
            "tool_call_id": m.get("tool_call_id"),
        }
        for m in visible[-limit:]
    ]


def response_brief(call: dict[str, Any]) -> dict[str, Any]:
    response = call.get("response") or {}
    return {
        "content": response.get("content"),
        "content_preview": compact_text(response.get("content"), 640),
        "tool_calls": response.get("tool_calls"),
        "finish_reason": (
            ((response.get("raw_data") or {}).get("choices") or [{}])[0].get(
                "finish_reason"
            )
        ),
    }


def event_key(event: dict[str, Any]) -> float:
    return float(event.get("monotonic_seconds") or 0.0)


def build_subagent_runs(
    events: list[dict[str, Any]], subagent_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    runs = []
    call_index = 0
    start_indexes = [
        index for index, event in enumerate(events) if event.get("event_type") == "subagent_start"
    ]
    for run_number, start_index in enumerate(start_indexes, start=1):
        start = events[start_index]
        done_index = None
        for index in range(start_index + 1, len(events)):
            if events[index].get("event_type") == "subagent_done":
                done_index = index
                break
            if events[index].get("event_type") == "subagent_start":
                break
        run_events = events[start_index : (done_index + 1 if done_index else start_index + 1)]
        done = events[done_index] if done_index is not None else {}
        turns = int(done.get("turns") or 0)
        run_calls = subagent_calls[call_index : call_index + turns] if turns else []
        call_index += len(run_calls)

        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "reasoning_tokens": 0,
        }
        for call in run_calls:
            usage = add_usage(usage, usage_of(call))

        child_events = []
        for event in run_events:
            event_type = event.get("event_type")
            if event_type == "kb_search":
                child_events.append(
                    {
                        "type": "search",
                        "query": event.get("query"),
                        "top_k": event.get("top_k"),
                        "doc_ids": event.get("doc_ids") or [],
                    }
                )
            elif event_type == "kb_read":
                child_events.append(
                    {
                        "type": "read_doc",
                        "doc_id": event.get("doc_id"),
                        "char_count": event.get("char_count"),
                    }
                )

        runs.append(
            {
                "id": f"subagent-{run_number}",
                "number": run_number,
                "depth": start.get("depth"),
                "timestamp": start.get("timestamp"),
                "question": start.get("question"),
                "context": start.get("context"),
                "answer": done.get("answer", ""),
                "answer_preview": compact_text(done.get("answer", ""), 900),
                "turns": turns,
                "usage": usage,
                "events": child_events,
                "llm_calls": [
                    {
                        "file": call.get("file"),
                        "timestamp": call.get("timestamp"),
                        "usage": usage_of(call),
                        "input_messages": request_messages(call, limit=6),
                        "output": response_brief(call),
                    }
                    for call in run_calls
                ],
            }
        )
    return runs


def call_kind(call: dict[str, Any]) -> str:
    name = call.get("call_name", "")
    if "user" in name:
        return "user"
    if "subagent" in name or "kb" in name:
        return "subagent"
    return "planner"


def build_agentic_nodes(
    llm_calls: list[dict[str, Any]], subagent_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    nodes = []
    subagent_index = 0
    planner_number = 0
    user_number = 0
    for call in llm_calls:
        kind = call_kind(call)
        if kind == "subagent":
            continue
        if kind == "user":
            user_number += 1
            nodes.append(
                {
                    "id": f"user-model-{user_number}",
                    "type": "user_model",
                    "title": f"User simulator call {user_number}",
                    "timestamp": call.get("timestamp"),
                    "usage": usage_of(call),
                    "input_messages": request_messages(call, limit=6),
                    "output": response_brief(call),
                    "children": [],
                }
            )
            continue

        planner_number += 1
        output = response_brief(call)
        tool_calls = output.get("tool_calls") or []
        children = []
        for tool_call in tool_calls:
            if tool_call.get("name") == "ask_knowledge_subagent":
                if subagent_index < len(subagent_runs):
                    children.append(subagent_runs[subagent_index])
                    subagent_index += 1
        nodes.append(
            {
                "id": f"planner-{planner_number}",
                "type": "planner",
                "title": f"Main agent planner call {planner_number}",
                "timestamp": call.get("timestamp"),
                "usage": usage_of(call),
                "input_messages": request_messages(call, limit=8),
                "output": output,
                "children": children,
            }
        )
    return nodes


def public_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index, message in enumerate(messages):
        for tool_call in message.get("tool_calls") or []:
            calls.append(
                {
                    "message_index": index,
                    "turn_idx": message.get("turn_idx"),
                    "timestamp": message.get("timestamp"),
                    "role": message.get("role"),
                    "name": tool_call.get("name"),
                    "arguments": tool_call.get("arguments"),
                    "requestor": tool_call.get("requestor"),
                }
            )
    return calls


def build_observations(
    task: dict[str, Any], simulation: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> list[dict[str, str]]:
    observations = []
    reward_info = simulation.get("reward_info") or {}
    actions = ((task.get("evaluation_criteria") or {}).get("actions")) or []
    if actions:
        expected = actions[0]
        expected_args = expected.get("arguments") or {}
        observations.append(
            {
                "level": "info",
                "title": "Expected action",
                "text": f"{expected.get('requestor')} should call {expected.get('name')} with {expected_args}.",
            }
        )

    card_calls = [
        call
        for call in tool_calls
        if call.get("name") == "apply_for_credit_card"
        and isinstance(call.get("arguments"), dict)
    ]
    wrong_cards = [
        call
        for call in card_calls
        if call["arguments"].get("card_type") != "EcoCard"
    ]
    eco_calls = [
        call for call in card_calls if call["arguments"].get("card_type") == "EcoCard"
    ]
    if wrong_cards:
        observations.append(
            {
                "level": "bad",
                "title": "DB was poisoned before the correct action",
                "text": "The user first applied for "
                f"{wrong_cards[0]['arguments'].get('card_type')}, then later applied for EcoCard. "
                "The evaluator action check matched EcoCard, but DB match failed because the state had an extra/wrong application.",
            }
        )
    if eco_calls:
        observations.append(
            {
                "level": "good",
                "title": "Expected EcoCard action appeared",
                "text": f"EcoCard was applied at public turn {eco_calls[-1].get('turn_idx')}.",
            }
        )
    stop_turns = [
        message.get("turn_idx")
        for message in simulation.get("messages", [])
        if "###STOP###" in (message.get("content") or "")
    ]
    if stop_turns:
        observations.append(
            {
                "level": "warn",
                "title": "Late stop",
                "text": f"The final stop token appeared at public turn {stop_turns[-1]}, long after the EcoCard application.",
            }
        )
    if reward_info:
        observations.append(
            {
                "level": "bad" if reward_info.get("reward") == 0 else "good",
                "title": "Reward result",
                "text": f"Reward {reward_info.get('reward')}; DB match {((reward_info.get('db_check') or {}).get('db_match'))}.",
            }
        )
    return observations


def aggregate_usage(llm_calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals = {
        "all": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "reasoning_tokens": 0,
            "calls": 0,
        }
    }
    for name in ("planner", "subagent", "user"):
        totals[name] = dict(totals["all"])
    for call in llm_calls:
        kind = call_kind(call)
        usage = usage_of(call)
        for bucket in ("all", kind):
            current = totals[bucket]
            merged = add_usage(current, usage)
            merged["calls"] = current.get("calls", 0) + 1
            totals[bucket] = merged
    return totals


def build_trace(run_dir: Path) -> dict[str, Any]:
    result = load_json(run_dir / "result.json")
    simulation = load_json(run_dir / "simulation.json")
    events = load_events(run_dir / "events.jsonl")
    llm_calls = load_llm_calls(run_dir / "llm_calls")
    task_id = result["task_id"]
    tasks = {task["id"]: task for task in load_json(TASKS_JSON)}
    task = tasks[task_id]

    subagent_calls = [call for call in llm_calls if call_kind(call) == "subagent"]
    subagent_runs = build_subagent_runs(events, subagent_calls)
    agentic_nodes = build_agentic_nodes(llm_calls, subagent_runs)
    public_messages = [message_brief(message) for message in simulation.get("messages", [])]
    tool_calls = public_tool_calls(simulation.get("messages", []))
    event_counts = Counter(event.get("event_type") for event in events)

    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "result": result,
        "simulation_summary": {
            "task_id": task_id,
            "duration": simulation.get("duration"),
            "termination_reason": simulation.get("termination_reason"),
            "reward": (simulation.get("reward_info") or {}).get("reward"),
            "agent_cost": simulation.get("agent_cost"),
            "user_cost": simulation.get("user_cost"),
            "message_count": len(public_messages),
            "event_counts": dict(event_counts),
        },
        "task": {
            "id": task_id,
            "scenario": task.get("user_scenario"),
            "evaluation_criteria": task.get("evaluation_criteria"),
            "required_documents": task.get("required_documents"),
            "user_tools": task.get("user_tools"),
        },
        "reward_info": simulation.get("reward_info"),
        "usage_totals": aggregate_usage(llm_calls),
        "observations": build_observations(task, simulation, tool_calls),
        "public_timeline": public_messages,
        "public_tool_calls": tool_calls,
        "agentic_tree": agentic_nodes,
        "subagent_runs": subagent_runs,
        "kb_evidence": load_json(run_dir / "kb_evidence.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data = build_trace(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
