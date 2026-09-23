from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    """A JsonStore rooted in an isolated temporary data/sandbox tree (empty data)."""

    settings = AppSettings(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    return JsonStore(settings)


@pytest.fixture
def seeded_store(tmp_path: Path) -> JsonStore:
    """A JsonStore whose data/ is a copy of the repo's sample dataset."""

    data_dir = tmp_path / "data"
    shutil.copytree(REPO_DATA_DIR, data_dir)
    settings = AppSettings(data_dir=data_dir, sandbox_dir=tmp_path / "sandbox")
    return JsonStore(settings)


def run(
    graph: AgentGraph,
    thread: str,
    store: JsonStore,
    input_: AgentState | Command[Any] | None,
) -> dict[str, Any]:
    """Invoke ``graph`` for ``thread`` and return state values plus interrupts.

    ``input_`` is either the graph input (first turn) or a ``Command`` (resume
    after an ``interrupt()``). The returned mapping includes an ``interrupts``
    key holding ``graph.get_state(config).interrupts`` after the invoke.
    """

    config: RunnableConfig = {"configurable": {"thread_id": thread}}
    context = AgentContext(store=store)
    result = graph.invoke(  # pyright: ignore[reportUnknownMemberType]
        input_, config, context=context
    )
    interrupts = graph.get_state(config).interrupts
    return {**result, "interrupts": interrupts}
