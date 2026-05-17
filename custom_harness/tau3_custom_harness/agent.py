"""Planner agent with internal KB subagents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
import re
from threading import Lock
from typing import Iterable, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate, set_llm_log_dir, set_llm_log_mode

from tau3_custom_harness.logger import HarnessLogger
from tau3_custom_harness.prompts import planner_system_prompt, subagent_system_prompt
from tau3_custom_harness.retrieval import BankingHybridRetriever


class PlannerState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


@dataclass
class InternalToolResult:
    content: str
    error: bool = False


class PlannerSubagentAgent(LLMConfigMixin, HalfDuplexAgent[PlannerState]):
    """Tau half-duplex agent with planner -> KB subagent retrieval."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        *,
        retriever: BankingHybridRetriever | None = None,
        llm_args: Optional[dict] = None,
        subagent_llm: str | None = None,
        subagent_llm_args: Optional[dict] = None,
        max_planner_internal_turns: int | None = None,
        max_subagent_turns: int | None = None,
        kb_document_count: int | None = None,
        default_user_tools: list[Tool] | None = None,
        subagent_delegation: str = "batch",
        logger_: HarnessLogger | None = None,
    ):
        if subagent_delegation not in {"single", "batch"}:
            raise ValueError("subagent_delegation must be 'single' or 'batch'")
        self.retriever = retriever or BankingHybridRetriever()
        self.subagent_llm = subagent_llm or llm
        self.subagent_llm_args = dict(subagent_llm_args or llm_args or {})
        self.max_planner_internal_turns = max_planner_internal_turns
        self.max_subagent_turns = max_subagent_turns
        self.kb_document_count = kb_document_count
        self.default_user_tools = list(default_user_tools or [])
        self.subagent_delegation = subagent_delegation
        self.logger = logger_ or HarnessLogger()
        self._auxiliary_cost_this_turn = 0.0
        self._read_docs: dict[str, str] = {}
        self._policy_lookup_failed = False
        self._state_lock = Lock()

        if self.subagent_delegation == "single":
            self._internal_tool_names = {"ask_knowledge_subagent"}
            knowledge_tools = [as_tool(self.ask_knowledge_subagent)]
        else:
            self._internal_tool_names = {"ask_knowledge_subagents"}
            knowledge_tools = [as_tool(self.ask_knowledge_subagents)]
        self._discoverable_tool_argument_names = {
            "unlock_discoverable_agent_tool": "agent_tool_name",
            "call_discoverable_agent_tool": "agent_tool_name",
            "give_discoverable_user_tool": "discoverable_tool_name",
        }
        public_tools = tools + knowledge_tools
        super().__init__(
            tools=public_tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    def set_seed(self, seed: int):
        super().set_seed(seed)
        self.subagent_llm_args["seed"] = seed

    @property
    def system_prompt(self) -> str:
        return planner_system_prompt(
            self.domain_policy,
            kb_document_count=self.kb_document_count,
            default_user_tools=self._format_default_user_tools(),
            subagent_delegation=self.subagent_delegation,
        )

    def _format_default_user_tools(self) -> str:
        if not self.default_user_tools:
            return "None listed for this task."

        blocks = []
        for tool in self.default_user_tools:
            signature = getattr(tool, "__signature__", "")
            description = ""
            try:
                description = tool.openai_schema["function"].get("description", "")
            except Exception:
                description = tool.short_desc or tool.long_desc
            description = " ".join(str(description).split())
            block = f"- {tool.name}{signature}"
            if description:
                block += f": {description}"
            blocks.append(block)
        return "\n".join(blocks)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> PlannerState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return PlannerState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: PlannerState
    ) -> tuple[AssistantMessage, PlannerState]:
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: PlannerState
    ) -> AssistantMessage:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        internal_cost = 0.0
        self._auxiliary_cost_this_turn = 0.0
        for turn in self._turns(self.max_planner_internal_turns):
            assistant_message = generate(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="custom_planner_response",
                **self.llm_args,
            )
            internal_cost += assistant_message.cost or 0.0

            if self._has_invalid_public_message_shape(assistant_message):
                self.logger.log(
                    "planner_invalid_shape_retry",
                    turn=turn,
                    has_text=assistant_message.has_text_content(),
                    has_tool_calls=assistant_message.is_tool_call(),
                )
                state.messages.append(
                    UserMessage.text(
                        "Planner format correction: each assistant turn must either "
                        "send a customer message or make tool calls, never both and "
                        "never neither. Retry with exactly one valid turn shape."
                    )
                )
                continue

            if self._has_mixed_internal_and_public_tool_calls(assistant_message):
                self.logger.log(
                    "planner_mixed_tool_retry",
                    turn=turn,
                    tool_names=self._tool_call_names(assistant_message),
                )
                state.messages.append(
                    UserMessage.text(
                        f"Planner format correction: {self._knowledge_tool_name()} cannot "
                        "be mixed with banking DB/action tools. Retry with only "
                        f"{self._knowledge_tool_name()}, or make only public banking tool "
                        "calls, or send a customer message."
                    )
                )
                continue

            undiscovered_tools = self._undiscovered_discoverable_tool_calls(
                assistant_message
            )
            if undiscovered_tools:
                self.logger.log(
                    "planner_discoverable_tool_evidence_retry",
                    turn=turn,
                    missing_evidence=undiscovered_tools,
                )
                state.messages.append(
                    UserMessage.text(
                        "Planner evidence correction: before using a discoverable "
                        "agent or user tool, ask the knowledge subagents to search "
                        "and read the KB document containing the exact tool name. "
                        "Then retry the tool call after the source text has been read."
                    )
                )
                continue

            blocked_tools = self._blocked_after_policy_lookup_failure(assistant_message)
            if blocked_tools:
                self.logger.log(
                    "planner_policy_lookup_failure_block",
                    turn=turn,
                    tool_names=blocked_tools,
                )
                state.messages.append(
                    UserMessage.text(
                        "Planner policy correction: the last knowledge-base lookup "
                        "failed, so you cannot use state-changing banking tools yet. "
                        "Retry the KB lookup with a narrower question, ask the "
                        "customer for non-sensitive clarification, or transfer if "
                        "that is the only safe path."
                    )
                )
                continue

            if not self._has_internal_tool_call(assistant_message):
                assistant_message.cost = (
                    internal_cost + self._auxiliary_cost_this_turn
                )
                return assistant_message

            state.messages.append(assistant_message)
            tool_messages = self._execute_internal_tool_calls(assistant_message)
            state.messages.extend(tool_messages)
            self.logger.log(
                "planner_internal_tool_turn",
                turn=turn,
                tool_count=len(tool_messages),
            )

        logger.warning("Planner hit internal tool turn limit")
        return AssistantMessage.text(
            "I need a moment to check the policy before I can continue.",
            cost=internal_cost + self._auxiliary_cost_this_turn,
        )

    def ask_knowledge_subagent(self, question: str, context: str = "") -> str:
        """Ask one knowledge-base subagent to research banking policy.

        Use this for all knowledge-base research in single-subagent mode. Ask
        one narrow question at a time. If the task has another independent
        policy/tool question, call this tool again in a later internal turn.

        Args:
            question: The exact policy question the planner needs answered.
            context: Customer/task context that helps the subagent search.

        Returns:
            A compact research note for the planner.
        """
        return self._run_subagent(question=question, context=context, depth=1)

    def ask_knowledge_subagents(self, requests: list[dict] | str) -> str:
        """Ask one or more knowledge-base subagents to research banking policy.

        Use this for all knowledge-base research. The requests list may contain
        1 to 4 items. Use one item for one narrow question. Use 2 to 4 items
        when the task has independent policy/tool questions that can be checked
        separately, such as verification rules, product policy, hidden tool
        names, default-vs-discoverable tool choice, or escalation rules.

        Args:
            requests: JSON array/list of request objects. Each object should
                include label, question, and optional context. Maximum 4
                requests per call; extra requests are ignored.

        Returns:
            Labeled research notes for the planner.
        """
        content, _error = self._run_subagent_batch(requests)
        return content

    def _run_subagent_batch(self, requests: list[dict] | str) -> tuple[str, bool]:
        normalized, note = self._normalize_subagent_requests(requests)
        if not normalized:
            return "Error: ask_knowledge_subagents requires at least one request.", True

        results_by_index: dict[int, dict[str, str]] = {}
        with ThreadPoolExecutor(max_workers=len(normalized)) as executor:
            futures = {
                executor.submit(
                    self._run_subagent,
                    question=request["question"],
                    context=request["context"],
                    depth=1,
                ): (index, request)
                for index, request in enumerate(normalized)
            }
            for future in as_completed(futures):
                index, request = futures[future]
                try:
                    answer = future.result()
                    status = "ok"
                except Exception as exc:
                    answer = f"Error: {exc}"
                    status = "error"
                    self.logger.log(
                        "subagent_batch_error",
                        label=request["label"],
                        error=str(exc),
                    )
                results_by_index[index] = {
                    "label": request["label"],
                    "status": status,
                    "answer": answer,
                }

        ordered_results = [
            results_by_index[index] for index in range(len(normalized))
        ]
        response = self._format_subagent_batch_results(ordered_results)
        if note:
            response = note + "\n\n" + response
        all_failed = all(result["status"] == "error" for result in ordered_results)
        return response, all_failed

    def _normalize_subagent_requests(
        self, requests: list[dict] | str
    ) -> tuple[list[dict[str, str]], str]:
        note = ""
        raw_requests: object = requests
        if isinstance(requests, str):
            try:
                raw_requests = json.loads(requests)
            except json.JSONDecodeError:
                return [], "Error: requests must be a JSON array/list."

        if isinstance(raw_requests, dict):
            raw_requests = [raw_requests]
        if not isinstance(raw_requests, list):
            return [], "Error: requests must be a JSON array/list."

        if len(raw_requests) > 4:
            note = (
                "Note: ask_knowledge_subagents accepts at most 4 requests per "
                "call; extra requests were ignored."
            )

        normalized = []
        for index, raw_request in enumerate(raw_requests[:4], start=1):
            if not isinstance(raw_request, dict):
                continue
            question = str(raw_request.get("question") or "").strip()
            if not question:
                continue
            label = str(raw_request.get("label") or f"request_{index}").strip()
            context = str(raw_request.get("context") or "").strip()
            normalized.append(
                {
                    "label": label or f"request_{index}",
                    "question": question,
                    "context": context,
                }
            )
        return normalized, note

    def _format_subagent_batch_results(self, results: list[dict[str, str]]) -> str:
        blocks = []
        for result in results:
            blocks.append(
                "\n".join(
                    [
                        f"## {result['label']}",
                        f"status: {result['status']}",
                        result["answer"].strip() or "(no answer)",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _run_subagent(self, question: str, context: str, depth: int) -> str:
        self._ensure_thread_llm_logging()
        prompt = "\n".join(
            [
                "Research this banking KB question for the planner.",
                "",
                f"Question: {question}",
                "",
                f"Context: {context or '(none provided)'}",
            ]
        )
        tools = self._subagent_tools(depth)
        messages: list[APICompatibleMessage] = [
            SystemMessage(
                role="system",
                content=subagent_system_prompt(
                    depth=depth,
                    kb_document_count=self.kb_document_count,
                ),
            ),
            # UserMessage is API-compatible, but importing it here only to appease
            # type checkers would add noise. The Tau generator accepts this shape.
        ]

        messages.append(UserMessage.text(prompt))
        self.logger.log(
            "subagent_start",
            depth=depth,
            question=question,
            context=context,
        )

        total_cost = 0.0
        for turn in self._turns(self.max_subagent_turns):
            assistant_message = generate(
                model=self.subagent_llm,
                tools=tools,
                messages=messages,
                call_name=f"custom_kb_subagent_depth_{depth}",
                **self.subagent_llm_args,
            )
            total_cost += assistant_message.cost or 0.0
            messages.append(assistant_message)

            if not assistant_message.is_tool_call():
                answer = assistant_message.content or ""
                self.logger.log(
                    "subagent_done",
                    depth=depth,
                    turns=turn + 1,
                    cost=total_cost,
                    answer=answer,
                )
                with self._state_lock:
                    self._auxiliary_cost_this_turn += total_cost
                return answer

            for tool_call in assistant_message.tool_calls or []:
                result = self._execute_subagent_tool(
                    tool_call.name, tool_call.arguments, depth
                )
                messages.append(
                    ToolMessage(
                        id=tool_call.id,
                        role="tool",
                        content=result.content,
                        requestor="assistant",
                        error=result.error,
                    )
                )

        self.logger.log("subagent_limit", depth=depth, cost=total_cost)
        with self._state_lock:
            self._auxiliary_cost_this_turn += total_cost
        return "The KB subagent hit its tool-turn limit before producing a final note."

    def _ensure_thread_llm_logging(self) -> None:
        """ContextVars do not cross ThreadPoolExecutor workers automatically."""
        set_llm_log_dir(self.logger.run_dir / "llm_calls")
        set_llm_log_mode("all")

    def _turns(self, max_turns: int | None) -> Iterable[int]:
        if max_turns is None:
            return count()
        return range(max_turns)

    def _subagent_tools(self, depth: int) -> list[Tool]:
        return [as_tool(self.search), as_tool(self.read_doc)]

    def search(self, query: str | list[str], top_k: int = 10) -> str:
        """Search banking knowledge documents.

        Returns only document ids, titles, and summaries. Use read_doc with a
        doc_id to inspect the full policy text.

        Args:
            query: Natural language search query, or up to 3 related queries.
            top_k: Maximum number of documents to return.

        Returns:
            Matching document summaries.
        """
        queries, was_truncated = self._normalize_search_queries(query)
        if not queries:
            return "No search query provided."

        hits = self.retriever.search(query=queries, top_k=top_k)
        self.logger.log(
            "kb_search",
            query=queries[0] if len(queries) == 1 else queries,
            query_count=len(queries),
            truncated=was_truncated,
            top_k=top_k,
            doc_ids=[hit.doc_id for hit in hits],
        )
        result = self.retriever.format_search_results(hits)
        if was_truncated:
            result = (
                "Note: search accepts at most 3 queries per call; extra queries "
                "were ignored.\n\n"
                + result
            )
        return result

    def _normalize_search_queries(self, query: str | list[str]) -> tuple[list[str], bool]:
        if isinstance(query, str):
            raw_queries = [query]
        else:
            raw_queries = query

        queries = [
            str(one_query).strip()
            for one_query in raw_queries
            if str(one_query).strip()
        ]
        was_truncated = len(queries) > 3
        queries = queries[:3]
        return queries, was_truncated

    def read_doc(self, doc_id: str) -> str:
        """Read one full banking knowledge document by document id.

        Args:
            doc_id: The exact document id returned by search.

        Returns:
            Full source text of the document.
        """
        try:
            content = self.retriever.read_doc(doc_id)
        except Exception as exc:
            self.logger.log("kb_read_error", doc_id=doc_id, error=str(exc))
            return f"Error: {exc}"

        self.logger.log("kb_read", doc_id=doc_id, char_count=len(content))
        with self._state_lock:
            self._read_docs[doc_id] = content
        return content

    def knowledge_evidence_report(self) -> dict:
        """Return the KB evidence gathered by hidden subagents in this run."""
        tool_evidence = {}
        for doc_id, content in self._read_docs.items():
            for tool_name in sorted(self._extract_tool_like_names(content)):
                tool_evidence.setdefault(tool_name, []).append(doc_id)
        return {
            "read_doc_ids": sorted(self._read_docs),
            "tool_evidence": tool_evidence,
        }

    def _execute_internal_tool_calls(
        self, assistant_message: AssistantMessage
    ) -> list[ToolMessage]:
        tool_messages = []
        for tool_call in assistant_message.tool_calls or []:
            if tool_call.name not in self._internal_tool_names:
                content = (
                    f"Error: {self._knowledge_tool_name()} was mixed with banking tools. "
                    "Call the knowledge subagents first, then call banking tools in a later turn."
                )
                error = True
            else:
                result = self._execute_internal_tool(tool_call.name, tool_call.arguments)
                content = result.content
                error = result.error
                if error:
                    self._policy_lookup_failed = True
            tool_messages.append(
                ToolMessage(
                    id=tool_call.id,
                    role="tool",
                    content=content,
                    requestor="assistant",
                    error=error,
                )
            )
        if tool_messages and not any(message.error for message in tool_messages):
            self._policy_lookup_failed = False
        return tool_messages

    def _execute_internal_tool(
        self, name: str, arguments: dict
    ) -> InternalToolResult:
        try:
            if name == "ask_knowledge_subagent":
                return InternalToolResult(
                    self.ask_knowledge_subagent(
                        question=arguments.get("question", ""),
                        context=arguments.get("context", ""),
                    )
                )
            if name == "ask_knowledge_subagents":
                content, error = self._run_subagent_batch(
                    arguments.get("requests", [])
                )
                return InternalToolResult(
                    content,
                    error,
                )
            return InternalToolResult(f"Error: unknown internal tool {name}", True)
        except Exception as exc:
            self.logger.log("internal_tool_error", tool=name, error=str(exc))
            return InternalToolResult(f"Error: {exc}", True)

    def _execute_subagent_tool(
        self, name: str, arguments: dict, depth: int
    ) -> InternalToolResult:
        try:
            if name == "search":
                return InternalToolResult(
                    self.search(
                        query=arguments.get("query", ""),
                        top_k=int(arguments.get("top_k", 10)),
                    )
                )
            if name == "read_doc":
                return InternalToolResult(
                    self.read_doc(doc_id=arguments.get("doc_id", ""))
                )
            return InternalToolResult(f"Error: tool {name} is not available here.", True)
        except Exception as exc:
            self.logger.log(
                "subagent_tool_error", tool=name, depth=depth, error=str(exc)
            )
            return InternalToolResult(f"Error: {exc}", True)

    def _has_internal_tool_call(self, message: AssistantMessage) -> bool:
        if not message.is_tool_call():
            return False
        return any(
            tool_call.name in self._internal_tool_names
            for tool_call in message.tool_calls or []
        )

    def _has_mixed_internal_and_public_tool_calls(
        self, message: AssistantMessage
    ) -> bool:
        if not message.is_tool_call():
            return False
        names = self._tool_call_names(message)
        has_internal = any(name in self._internal_tool_names for name in names)
        has_public = any(name not in self._internal_tool_names for name in names)
        return has_internal and has_public

    def _tool_call_names(self, message: AssistantMessage) -> list[str]:
        return [tool_call.name for tool_call in message.tool_calls or []]

    def _has_invalid_public_message_shape(self, message: AssistantMessage) -> bool:
        has_text = message.has_text_content()
        has_tools = message.is_tool_call()
        return has_text == has_tools

    def _blocked_after_policy_lookup_failure(
        self, message: AssistantMessage
    ) -> list[str]:
        if not self._policy_lookup_failed or not message.is_tool_call():
            return []
        blocked = []
        for tool_call in message.tool_calls or []:
            if self._is_safe_after_policy_lookup_failure(tool_call.name):
                continue
            blocked.append(tool_call.name)
        return blocked

    def _is_safe_after_policy_lookup_failure(self, tool_name: str) -> bool:
        return tool_name in {
            "ask_knowledge_subagent",
            "ask_knowledge_subagents",
            "transfer_to_human_agents",
            "get_current_time",
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
            "get_referrals_by_user",
            "get_credit_card_transactions_by_user",
            "get_credit_card_accounts_by_user",
            "list_discoverable_agent_tools",
        }

    def _undiscovered_discoverable_tool_calls(
        self, message: AssistantMessage
    ) -> list[dict[str, str]]:
        if not message.is_tool_call():
            return []

        missing = []
        for tool_call in message.tool_calls or []:
            argument_name = self._discoverable_tool_argument_names.get(tool_call.name)
            if argument_name is None:
                continue

            hidden_tool_name = self._tool_call_argument(tool_call.arguments, argument_name)
            if not hidden_tool_name:
                continue
            if self._has_read_evidence_for_tool(hidden_tool_name):
                continue
            missing.append(
                {
                    "wrapper_tool": tool_call.name,
                    "hidden_tool_name": hidden_tool_name,
                }
            )
        return missing

    def _tool_call_argument(self, arguments: object, name: str) -> str:
        if isinstance(arguments, dict):
            value = arguments.get(name)
            return str(value) if value is not None else ""
        return ""

    def _has_read_evidence_for_tool(self, tool_name: str) -> bool:
        pattern = re.compile(rf"\b{re.escape(tool_name)}\b")
        return any(pattern.search(content) for content in self._read_docs.values())

    def _extract_tool_like_names(self, text: str) -> set[str]:
        return set(re.findall(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){2,}\b", text))

    def _knowledge_tool_name(self) -> str:
        if self.subagent_delegation == "single":
            return "ask_knowledge_subagent"
        return "ask_knowledge_subagents"
