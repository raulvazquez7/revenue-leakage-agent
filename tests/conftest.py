from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from revenue_leakage_agent.config import AppSettings, get_settings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def _no_langfuse_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force tracing off for every test under ``tests/``, regardless of ``.env``.

    Graph nodes call the process-wide, ``lru_cache``-d ``get_settings()``
    directly (not an injectable settings object), so a real developer
    ``.env`` with live LANGFUSE_* keys would otherwise make every
    ``ScriptedChatModel`` graph test export a real trace to that Langfuse
    project the moment tracing actually works. Env vars take precedence over
    ``.env`` values in pydantic-settings, so setting them to "" here is
    enough to make ``AppSettings().langfuse_*`` fall back to "unset" without
    needing to touch the ``.env`` file itself. The cache is cleared before
    (to drop any settings cached from import time or an earlier test) and
    after (so nothing here leaks into a later, unrelated ``get_settings()``
    call in the same process).

    ``tests/unit/test_tracing.py`` doesn't rely on ``get_settings()``/``.env`` at
    all — it builds ``AppSettings`` explicitly with dummy keys — so it is
    unaffected by this fixture forcing the ambient env empty.

    Not applied to ``evals/``: those are a separate, rootless pytest
    collection (run via ``pytest evals -m eval``, not under this
    ``tests/`` conftest's tree) that imports this module's ``run`` helper
    directly via ``sys.path``/``import conftest`` rather than through
    pytest's conftest discovery, so this autouse fixture is never collected
    or applied there; evals intentionally use the real ``.env`` settings.
    """

    for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
