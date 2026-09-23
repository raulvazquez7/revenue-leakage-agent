"""LangGraph Studio / ``langgraph dev`` compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from fakes import route, scripted
from revenue_leakage_agent import graph as graph_module
from revenue_leakage_agent.context import AgentContext, resolve_store
from revenue_leakage_agent.graph import make_graph
from revenue_leakage_agent.llm import AgentModels

REPO_ROOT = Path(__file__).resolve().parents[2]


def _models() -> AgentModels:
    return AgentModels(
        router=scripted(route("conversation", "chit_chat")),
        agent=scripted(AIMessage(content="agent must not run")),
        conversational=scripted(AIMessage(content="hello")),
    )


def test_make_graph_has_no_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dev server rejects graphs that bring their own checkpointer."""

    monkeypatch.setattr(graph_module, "build_default_models", _models)

    graph = make_graph()

    assert graph.checkpointer is None  # pyright: ignore[reportUnknownMemberType]


def test_langgraph_json_points_at_make_graph() -> None:
    config = json.loads((REPO_ROOT / "langgraph.json").read_text(encoding="utf-8"))
    module_path, _, attr = config["graphs"]["revenue_leakage_agent"].partition(":")

    assert (REPO_ROOT / module_path).is_file()
    assert attr == make_graph.__name__


def test_context_is_optional_for_platform_callers() -> None:
    """The server coerces a JSON context dict with ``AgentContext(**context)``."""

    context = AgentContext()

    assert context.store is None
    assert resolve_store(context).data_dir == resolve_store(None).data_dir


def test_graph_accepts_an_empty_context_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "build_default_models", _models)

    # The server passes the raw JSON dict; LangGraph coerces it at runtime.
    result = make_graph().invoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="hi")]},
        context={},  # pyright: ignore[reportArgumentType]
    )

    assert result["messages"][-1].content == "hello"


def test_context_json_schema_hides_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Studio renders the context form from this schema; the store is not JSON."""

    monkeypatch.setattr(graph_module, "build_default_models", _models)

    schema = make_graph().get_context_jsonschema()

    assert schema is not None
    assert "store" not in schema.get("properties", {})
