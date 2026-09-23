"""Scripted chat models for behaviour tests.

``GenericFakeChatModel.bind_tools`` raises ``NotImplementedError`` in
langchain-core 1.6.x, but the graph's agent and router nodes both call
``bind_tools`` / ``with_structured_output`` (which itself calls
``bind_tools``) before invoking. ``ScriptedChatModel`` overrides
``bind_tools`` to return itself so the scripted queue still drives the
conversation; ``with_structured_output(..., method="json_schema")`` then
parses a scripted tool call whose ``name`` matches the target schema.

Under LangGraph's ``messages`` stream mode the model is *streamed*; the
upstream fake cannot stream ``tool_calls`` (it raises "No generations found"
for empty content), so ``ScriptedChatModel._stream`` emits tool calls as one
chunk and splits text on whitespace like the upstream fake.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGenerationChunk


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

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        assert isinstance(message, AIMessage)
        if message.tool_calls:
            chunks = [
                AIMessageChunk(
                    content=message.content,
                    id=message.id,
                    tool_call_chunks=[
                        tool_call_chunk(
                            name=call["name"],
                            args=json.dumps(call["args"]),
                            id=call["id"],
                            index=index,
                        )
                        for index, call in enumerate(message.tool_calls)
                    ],
                )
            ]
        else:
            tokens = [t for t in re.split(r"(\s)", message.text) if t]
            chunks = [AIMessageChunk(content=t, id=message.id) for t in tokens]
        for chunk in chunks:
            generation = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=generation)
            yield generation


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
