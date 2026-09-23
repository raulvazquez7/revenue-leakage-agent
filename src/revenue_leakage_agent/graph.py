from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.llm import AgentModels, build_default_models
from revenue_leakage_agent.nodes import (
    make_agent_node,
    make_conversational_node,
    make_router_node,
    route_from_decision,
)
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tools import get_tools

AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def build_graph(
    models: AgentModels | None = None,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> AgentGraph:
    """Build the revenue leakage graph.

    ``models`` default to :func:`build_default_models`. No checkpointer is attached
    unless one is passed; interactive interfaces pass ``InMemorySaver()`` so the
    approval ``interrupt()`` can resume.
    """

    models = models or build_default_models()
    tools = get_tools()

    # StateGraph's add_node/compile overloads carry partially-unknown library
    # generics under pyright strict; keep the builder untyped and cast the result.
    builder: Any = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("router", make_router_node(models.router))
    builder.add_node("conversation", make_conversational_node(models.conversational))
    builder.add_node("agent", make_agent_node(models.agent, tools))
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

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

    return cast(AgentGraph, builder.compile(checkpointer=checkpointer))


def make_graph() -> AgentGraph:
    """Graph factory for LangGraph Studio / ``langgraph dev`` (platform checkpoints)."""

    return build_graph(checkpointer=None)
