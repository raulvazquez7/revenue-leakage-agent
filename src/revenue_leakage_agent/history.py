"""Bounded conversation history for model calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

# trim_messages' optional text_splitter parameter is partially untyped.
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
    trim_messages,  # pyright: ignore[reportUnknownVariableType]
)

from revenue_leakage_agent.messages import extract_ai_text


def recent_history(
    messages: Sequence[AnyMessage], max_messages: int
) -> list[AnyMessage]:
    """Last messages within budget, starting on a HumanMessage.

    Unpaired tool traffic is dropped first (see :func:`drop_unpaired_tool_calls`),
    then starting on a HumanMessage guarantees no ToolMessage is orphaned from
    the AIMessage whose ``tool_calls`` produced it.

    If the current turn alone exceeds ``max_messages``, trimming would return
    nothing; in that case the whole current turn (from the last HumanMessage)
    is returned so the model always sees the latest request. That result can
    exceed ``max_messages``; its length is bounded by the graph's
    ``recursion_limit``, which caps agent/tool steps per turn.
    """

    messages = drop_unpaired_tool_calls(messages)
    # trim_messages is typed as returning list[BaseMessage]; it only drops
    # messages, so every element is still one of the AnyMessage inputs.
    trimmed = cast(
        list[AnyMessage],
        trim_messages(
            list(messages),
            max_tokens=max_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
            allow_partial=False,
        ),
    )
    if trimmed:
        return trimmed
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index:])
    return []


def drop_unpaired_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Drop AIMessages with unanswered ``tool_calls`` and orphan ToolMessages.

    Providers reject a request whose assistant tool calls lack a matching tool
    result. That happens when a turn is abandoned between the agent and tools
    nodes (a UI rerun mid-stream, a new message sent instead of resuming an
    approval interrupt, a crashed step), so the whole unanswered call and any
    partial results are removed from the model's view; the checkpointed state
    itself is left untouched.
    """

    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    kept: list[AnyMessage] = []
    kept_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            call_ids = {str(call["id"]) for call in message.tool_calls}
            if not call_ids <= answered:
                continue
            kept_call_ids |= call_ids
        elif (
            isinstance(message, ToolMessage)
            and message.tool_call_id not in kept_call_ids
        ):
            continue
        kept.append(message)
    return kept


def dialogue_history(
    messages: Sequence[AnyMessage], max_messages: int
) -> list[AnyMessage]:
    """Recent human/assistant dialogue within budget, without tool traffic.

    Keeps HumanMessages and AIMessages that carry text and no ``tool_calls``,
    then applies :func:`recent_history`. Used by the router and conversational
    nodes, which need the conversation, not the tool round-trips that would
    otherwise crowd prior turns out of the budget.
    """

    dialogue: list[AnyMessage] = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
        or (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and bool(extract_ai_text(message))
        )
    ]
    return recent_history(dialogue, max_messages)
