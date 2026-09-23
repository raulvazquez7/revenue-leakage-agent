from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.nodes import (
    agent_node,
    conversational_node,
    route_from_decision,
    router_node,
)
from agents.state import AgentState
from agents.tools import get_tools


def build_graph() -> Any:
    """Build the V1 revenue leakage LangGraph."""

    builder: Any = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("conversation", conversational_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(get_tools(), handle_tool_errors=True))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_from_decision,
        {
            "conversation": "conversation",
            "investigation": "agent",
        },
    )
    builder.add_edge("conversation", END)
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=InMemorySaver())
