from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    active_scope: dict[str, Any] | None
    findings: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    applied_actions: list[dict[str, Any]]
    last_error: dict[str, Any] | None
    route_decision: dict[str, Any] | None
