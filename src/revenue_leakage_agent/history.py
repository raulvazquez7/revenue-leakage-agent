"""Bounded conversation history for model calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

# trim_messages' optional text_splitter parameter is partially untyped.
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    trim_messages,  # pyright: ignore[reportUnknownVariableType]
)


def recent_history(
    messages: Sequence[AnyMessage], max_messages: int
) -> list[AnyMessage]:
    """Last messages within budget, starting on a HumanMessage.

    Starting on a HumanMessage guarantees no ToolMessage is orphaned from the
    AIMessage whose ``tool_calls`` produced it.

    If the current turn alone exceeds ``max_messages``, trimming would return
    nothing; in that case the whole current turn (from the last HumanMessage)
    is returned so the model always sees the latest request.
    """

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
