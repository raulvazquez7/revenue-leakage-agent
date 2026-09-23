"""Scripted chat models for behaviour tests.

``GenericFakeChatModel.bind_tools`` raises ``NotImplementedError`` in
langchain-core 1.6.x, but the graph's agent and router nodes both call
``bind_tools`` / ``with_structured_output`` (which itself calls
``bind_tools``) before invoking. ``ScriptedChatModel`` overrides
``bind_tools`` to return itself so the scripted queue still drives the
conversation; ``with_structured_output(..., method="json_schema")`` then
parses a scripted tool call whose ``name`` matches the target schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(GenericFakeChatModel):
    """Fake chat model that returns scripted AIMessages and supports bind_tools."""

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ScriptedChatModel:
        return self


def scripted(*messages: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))


def route(route: str, intent: str, question: str | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteDecision",
                "id": f"rd-{uuid4().hex[:6]}",
                "args": {
                    "route": route,
                    "intent": intent,
                    "reason": "scripted",
                    "resolved_question": question,
                },
            }
        ],
    )


def call(name: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"tc-{uuid4().hex[:6]}"}],
    )
