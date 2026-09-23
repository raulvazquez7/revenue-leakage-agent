from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import AliasChoices

from revenue_leakage_agent.config import AppSettings, get_settings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore

REPO_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test under ``tests/`` on code-default settings.

    Graph nodes call the process-wide, ``lru_cache``-d ``get_settings()``
    directly (not an injectable settings object), so a developer's ``.env`` or
    exported variables would otherwise leak into tests: real LANGFUSE_* keys
    would export a trace from every ``ScriptedChatModel`` graph test, and
    values such as ``ROUTER_REASONING_EFFORT`` would change which OpenAI API
    the default models call. So this fixture stops ``AppSettings`` from
    reading any ``.env`` file and unsets every environment variable that maps
    to a setting (``data/`` access is unaffected: tests use absolute paths or
    the default relative ``data`` under the repo root). The settings cache is
    cleared before (to drop anything cached at import time or by an earlier
    test) and after (so nothing leaks into a later ``get_settings()`` call).

    Tests that need specific settings build ``AppSettings(...)`` explicitly or
    set environment variables themselves.

    Not applied to ``evals/``: those are a separate, rootless pytest
    collection (run via ``pytest evals -m eval``, not under this
    ``tests/`` conftest's tree) that imports this module's ``run`` helper
    directly via ``sys.path``/``import conftest`` rather than through
    pytest's conftest discovery, so this autouse fixture is never collected
    or applied there; evals intentionally use the real ``.env`` settings.
    """

    monkeypatch.setitem(AppSettings.model_config, "env_file", None)
    for name in _settings_env_names():
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings_env_names() -> set[str]:
    names: set[str] = set()
    for field_name, field in AppSettings.model_fields.items():
        names.add(field_name.upper())
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            names.update(str(choice).upper() for choice in alias.choices)
    return names


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
