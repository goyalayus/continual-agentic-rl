#!/usr/bin/env python3
"""Local smoke checks for the custom harness.

This does not call an LLM and does not write a simulation run.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.environment.tool import as_tool

import tau3_custom_harness.agent as agent_module
import tau3_custom_harness.user as user_module
from tau3_custom_harness.agent import PlannerSubagentAgent
from tau3_custom_harness.logger import HarnessLogger
from tau3_custom_harness.retrieval import BankingHybridRetriever
from tau3_custom_harness.user import SafeUserSimulator


def public_noop(value: str) -> str:
    """Public test tool.

    Args:
        value: Any value.

    Returns:
        Echoed value.
    """
    return value


def unlock_discoverable_agent_tool(agent_tool_name: str) -> str:
    """Public test shim with the same name as the banking wrapper.

    Args:
        agent_tool_name: Hidden tool name.

    Returns:
        Echoed tool name.
    """
    return agent_tool_name


def assert_mixed_internal_public_calls_retry() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="internal",
                        name="ask_knowledge_subagent",
                        arguments={"question": "policy?", "context": ""},
                    ),
                    ToolCall(
                        id="public",
                        name="public_noop",
                        arguments={"value": "bad mix"},
                    ),
                ],
                cost=0.01,
            )
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="public2",
                    name="public_noop",
                    arguments={"value": "visible"},
                )
            ],
            cost=0.02,
        )

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(public_noop)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.tool_calls is not None
        assert [call.name for call in message.tool_calls] == ["public_noop"]
        hidden_tool_messages = [item for item in state.messages if getattr(item, "role", None) == "tool"]
        assert not hidden_tool_messages, "mixed public calls must not become hidden fake tool results"
    finally:
        agent_module.generate = original_generate


def assert_internal_kb_call_stays_internal() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="internal",
                        name="ask_knowledge_subagent",
                        arguments={"question": "policy?", "context": ""},
                    )
                ],
                cost=0.01,
            )
        return AssistantMessage.text("I checked the policy.", cost=0.02)

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(public_noop)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        agent.ask_knowledge_subagent = lambda question, context="": "policy note"
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.content == "I checked the policy."
        hidden_tool_messages = [item for item in state.messages if getattr(item, "role", None) == "tool"]
        assert len(hidden_tool_messages) == 1
        assert hidden_tool_messages[0].content == "policy note"
        assert message.cost == 0.03
    finally:
        agent_module.generate = original_generate


def assert_failed_kb_lookup_blocks_state_changing_tools() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="internal",
                        name="ask_knowledge_subagent",
                        arguments={"question": "policy?", "context": ""},
                    )
                ],
                cost=0.01,
            )
        if len(calls) == 2:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="public",
                        name="public_noop",
                        arguments={"value": "unsafe after kb failure"},
                    )
                ],
                cost=0.02,
            )
        return AssistantMessage.text("I need to retry the policy lookup.", cost=0.03)

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(public_noop)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        agent.ask_knowledge_subagent = lambda question, context="": (_ for _ in ()).throw(
            RuntimeError("provider failed")
        )
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.content == "I need to retry the policy lookup."
        assert any(
            "last knowledge-base lookup failed" in (getattr(item, "content", "") or "")
            for item in state.messages
        )
    finally:
        agent_module.generate = original_generate


def assert_discoverable_tools_need_read_evidence() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="discoverable",
                        name="unlock_discoverable_agent_tool",
                        arguments={"agent_tool_name": "open_bank_account_4821"},
                    )
                ],
                cost=0.01,
            )
        return AssistantMessage.text("I need to read the source first.", cost=0.02)

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(unlock_discoverable_agent_tool)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.content == "I need to read the source first."
        assert any(
            "Planner evidence correction" in getattr(item, "content", "")
            for item in state.messages
        )
    finally:
        agent_module.generate = original_generate


def assert_discoverable_tools_allowed_after_read_evidence() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="discoverable",
                    name="unlock_discoverable_agent_tool",
                    arguments={"agent_tool_name": "open_bank_account_4821"},
                )
            ],
            cost=0.01,
        )

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(unlock_discoverable_agent_tool)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        agent._read_docs["doc_test"] = "Use open_bank_account_4821 to open the account."
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.tool_calls is not None
        assert [call.name for call in message.tool_calls] == [
            "unlock_discoverable_agent_tool"
        ]
    finally:
        agent_module.generate = original_generate


def assert_invalid_text_plus_tool_call_retries() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(
                role="assistant",
                content="I will do that.",
                tool_calls=[
                    ToolCall(
                        id="public",
                        name="public_noop",
                        arguments={"value": "mixed text"},
                    )
                ],
                cost=0.01,
            )
        return AssistantMessage.text("Clean response.", cost=0.02)

    original_generate = agent_module.generate
    agent_module.generate = fake_generate
    try:
        agent = PlannerSubagentAgent(
            [as_tool(public_noop)],
            "test policy",
            "fake-model",
            retriever=BankingHybridRetriever(),
        )
        state = agent.get_init_state()
        message = agent._generate_next_message(UserMessage.text("hello"), state)
        assert message.content == "Clean response."
        assert message.tool_calls is None
        assert message.cost == 0.03
    finally:
        agent_module.generate = original_generate


def assert_s3_sync_is_nonfatal_without_aws() -> None:
    old_path = os.environ.get("PATH", "")
    with tempfile.TemporaryDirectory() as tmp:
        logger = HarnessLogger(log_dir=Path(tmp))
        logger.log("test_event")
        try:
            os.environ["PATH"] = ""
            assert logger.sync_to_s3("s3://example-bucket/prefix") is False
        finally:
            os.environ["PATH"] = old_path


def assert_user_empty_response_retries_to_stop() -> None:
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return AssistantMessage(role="assistant", content=None, cost=0.01)
        return AssistantMessage.text("###STOP###", cost=0.02)

    original_generate = user_module.generate
    user_module.generate = fake_generate
    try:
        user = SafeUserSimulator(llm="fake-model", instructions="test")
        state = user.get_init_state()
        message = user._generate_next_message(
            AssistantMessage.text("Done."), state
        )
        assert message.content == "###STOP###"
        assert message.cost == 0.03
    finally:
        user_module.generate = original_generate


def main() -> int:
    old_key = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        retriever = BankingHybridRetriever()
        assert len(retriever.docs) == 698
        assert len(retriever.chunks) == 716
        assert retriever.embeddings.shape[0] == len(retriever.chunk_ids)

        hits = retriever.search(
            "cash back dispute wrong rewards transaction",
            top_k=3,
        )
        assert hits, "expected at least one cash back dispute hit"
        assert any("cash" in hit.summary.lower() for hit in hits)
        assert "Applying Resolved Cash Back Dispute Corrections" in hits[0].title

        offer_hits = retriever.search(
            "mailed credit card offer expired unavailable",
            top_k=8,
        )
        assert offer_hits, "expected mailed offer retrieval hits"

        doc_text = retriever.read_doc(hits[0].doc_id)
        assert doc_text.startswith("# ")
        assert len(doc_text) > 200

        task = get_tasks()[0]
        env = get_environment(retrieval_variant="no_knowledge", task=task)
        agent = PlannerSubagentAgent(
            env.get_tools(),
            env.get_policy(),
            "gpt-4.1",
            retriever=retriever,
        )
        planner_tool_names = [tool.name for tool in agent.tools]
        assert "ask_knowledge_subagent" in planner_tool_names
        assert not {"KB_search", "grep", "shell"} & set(planner_tool_names)
        assert [tool.name for tool in agent._subagent_tools(1)] == [
            "search",
            "read_doc",
            "ask_knowledge_subagent",
        ]
        assert [tool.name for tool in agent._subagent_tools(2)] == [
            "search",
            "read_doc",
        ]
        give_result = env.tools.give_discoverable_user_tool(
            "get_referral_link",
            {"user_id": "123", "card_name": "Gold Rewards Card"},
        )
        assert "Invalid JSON" not in give_result
        user_result = env.user_tools.call_discoverable_user_tool(
            "get_referral_link",
            {"user_id": "123", "card_name": "Gold Rewards Card"},
        )
        assert "Invalid JSON" not in user_result
        assert_mixed_internal_public_calls_retry()
        assert_internal_kb_call_stays_internal()
        assert_failed_kb_lookup_blocks_state_changing_tools()
        assert_discoverable_tools_need_read_evidence()
        assert_discoverable_tools_allowed_after_read_evidence()
        assert_invalid_text_plus_tool_call_retries()
        assert_s3_sync_is_nonfatal_without_aws()
        assert_user_empty_response_retries_to_stop()
    finally:
        if old_key is not None:
            os.environ["OPENROUTER_API_KEY"] = old_key

    print("custom harness smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
