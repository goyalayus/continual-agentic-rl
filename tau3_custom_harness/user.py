"""User simulator wrapper with stricter message-shape recovery."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import UserState, ValidUserInputMessage
from tau2.utils.llm_utils import generate

from tau3_custom_harness.logger import HarnessLogger


class SafeUserSimulator(UserSimulator):
    """Retries empty user-simulator outputs instead of crashing the run."""

    def __init__(
        self,
        *args,
        logger_: HarnessLogger | None = None,
        max_invalid_retries: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.harness_logger = logger_
        self.max_invalid_retries = max_invalid_retries

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> UserMessage:
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )

        logger.debug(f"User responds to message: {message}")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            state.messages.append(message)
        elif message.has_content() or message.is_tool_call():
            state.messages.append(message)

        total_cost = 0.0
        for attempt in range(self.max_invalid_retries + 1):
            assistant_message = generate(
                model=self.llm,
                messages=state.system_messages + state.flip_roles(),
                tools=self.tools,
                call_name="user_simulator_response",
                **self.llm_args,
            )
            total_cost += assistant_message.cost or 0.0
            user_message = self._assistant_to_user_message(assistant_message)
            logger.debug(f"Response: {user_message.content}")

            if user_message.has_content() or user_message.is_tool_call():
                user_message.cost = total_cost or assistant_message.cost
                return user_message

            self._log_invalid_user_response(attempt, assistant_message)
            state.messages.append(
                AssistantMessage.text(
                    "User simulator format correction: respond with either a "
                    "normal customer message, a valid tool call, or ###STOP### "
                    "if your scenario goal is complete. Empty responses are invalid."
                )
            )

        return UserMessage.text("###STOP###", cost=total_cost or None)

    def _assistant_to_user_message(self, assistant_message: AssistantMessage) -> UserMessage:
        user_message = UserMessage(
            role="user",
            content=assistant_message.content,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )
        return user_message

    def _log_invalid_user_response(
        self, attempt: int, assistant_message: AssistantMessage
    ) -> None:
        if self.harness_logger is None:
            return
        finish_reason: Optional[str] = None
        try:
            finish_reason = assistant_message.raw_data["choices"][0]["finish_reason"]
        except (KeyError, IndexError, TypeError):
            pass
        self.harness_logger.log(
            "user_invalid_shape_retry",
            attempt=attempt,
            finish_reason=finish_reason,
        )
