from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from conftest import run
from fakes import route, scripted
from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.llm import AgentModels
from revenue_leakage_agent.persistence import build_checkpointer
from revenue_leakage_agent.store import JsonStore


def _conversation_models(reply: str) -> AgentModels:
    return AgentModels(
        router=scripted(route("conversation", "chit_chat")),
        agent=scripted(AIMessage(content="agent must not run")),
        conversational=scripted(AIMessage(content=reply)),
    )


def test_build_checkpointer_defaults_to_memory() -> None:
    assert isinstance(
        build_checkpointer(AppSettings(checkpoint_db=None)), InMemorySaver
    )


def test_sqlite_checkpoints_survive_a_new_graph(
    tmp_path: Path, seeded_store: JsonStore
) -> None:
    db_path = tmp_path / "nested" / "checkpoints.sqlite"
    settings = AppSettings(checkpoint_db=db_path)

    first = build_checkpointer(settings)
    assert isinstance(first, SqliteSaver)
    graph = build_graph(_conversation_models("hello there"), checkpointer=first)
    run(graph, "t1", seeded_store, {"messages": [HumanMessage(content="hi")]})
    first.conn.close()

    second = build_checkpointer(settings)
    assert isinstance(second, SqliteSaver)
    restarted = build_graph(_conversation_models("unused"), checkpointer=second)
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}
    messages = restarted.get_state(config).values["messages"]
    second.conn.close()

    assert db_path.exists()
    assert [m.content for m in messages] == ["hi", "hello there"]


def test_empty_checkpoint_db_env_means_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHECKPOINT_DB", "")

    settings = AppSettings(_env_file=None)  # pyright: ignore[reportCallIssue]

    assert settings.checkpoint_db is None
