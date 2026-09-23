from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.llm import AgentModels, build_default_models
from revenue_leakage_agent.nodes import route_from_decision
from revenue_leakage_agent.state import AgentState


def _models() -> AgentModels:
    return build_default_models(AppSettings(openai_api_key="sk-test"))


def test_build_default_models_requires_api_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        build_default_models(AppSettings(openai_api_key=""))


def test_build_default_models_uses_configured_model_names() -> None:
    settings = AppSettings(
        openai_api_key="sk-test",
        router_model="router-model",
        agent_model="agent-model",
        conversational_model="chat-model",
    )

    models = build_default_models(settings)

    assert isinstance(models.router, ChatOpenAI)
    assert isinstance(models.agent, ChatOpenAI)
    assert isinstance(models.conversational, ChatOpenAI)
    assert models.router.model_name == "router-model"
    assert models.agent.model_name == "agent-model"
    assert models.conversational.model_name == "chat-model"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"route_decision": {"route": "investigation"}}, "investigation"),
        ({"route_decision": {"route": "conversation"}}, "conversation"),
        ({}, "conversation"),
    ],
)
def test_route_from_decision(state: AgentState, expected: str) -> None:
    assert route_from_decision(state) == expected


def test_build_graph_wires_nodes_and_context_schema() -> None:
    graph = build_graph(_models())

    assert {"router", "conversation", "agent", "tools"} <= set(graph.nodes)
    assert graph.context_schema is AgentContext
    assert graph.checkpointer is None  # pyright: ignore[reportUnknownMemberType]


def test_build_graph_accepts_checkpointer() -> None:
    saver = InMemorySaver()

    graph = build_graph(_models(), checkpointer=saver)

    assert graph.checkpointer is saver  # pyright: ignore[reportUnknownMemberType]
