from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import BaseMessage


def extract_ai_text(message: BaseMessage) -> str:
    """Extract user-facing text from plain content or typed content blocks."""

    content_blocks = getattr(message, "content_blocks", None)
    if content_blocks is not None:
        block_parts: list[str] = []
        for block in cast(list[Any], content_blocks):
            if not isinstance(block, dict):
                continue
            block_map = cast(dict[str, Any], block)
            text = block_map.get("text")
            if block_map.get("type") == "text" and isinstance(text, str):
                block_parts.append(text)
        if block_parts:
            return "\n".join(block_parts)

    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_map = cast(dict[str, Any], block)
        if block_map.get("type") == "text" and isinstance(block_map.get("text"), str):
            parts.append(cast(str, block_map["text"]))
    if parts:
        return "\n".join(parts)
    return str(content)
