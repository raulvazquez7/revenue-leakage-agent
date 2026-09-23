"""Live-model trajectory evals.

Unlike the scripted-model tests under ``tests/``, these drive the real
compiled graph (``build_graph()`` with real ``ChatOpenAI`` models from
``.env``) end to end and assert on tool-call order, forbidden tools, regexes
over the model's own final reply, and sandbox ledger side effects. They are
therefore non-deterministic and cost a small amount of OpenAI usage per run,
so they are excluded from ``uv run task test`` and from CI (see
``pyproject.toml``'s ``-m "not eval"`` addopt) and only run via
``uv run task eval``. See ``evals/README.md`` for the scenario table, how to
run them, and the last recorded results.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# ``tests/`` has no ``__init__.py`` (pytest adds its own directory to
# ``sys.path`` when collecting it directly); ``evals`` is a separate pytest
# rootless package that never collects ``tests/``, so we add it to the path
# ourselves and reuse its tiny graph-invoke helper instead of duplicating it.
_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from conftest import REPO_DATA_DIR, run  # noqa: E402
from evals.scenarios import SCENARIOS, Scenario  # noqa: E402
from revenue_leakage_agent.config import get_settings  # noqa: E402
from revenue_leakage_agent.graph import build_graph  # noqa: E402
from revenue_leakage_agent.messages import extract_ai_text  # noqa: E402
from revenue_leakage_agent.store import JsonStore  # noqa: E402

pytestmark = pytest.mark.eval


def _eval_store(tmp_path: Path) -> JsonStore:
    """A JsonStore whose data/ is a fresh copy of the repo's sample dataset.

    Mirrors ``tests/conftest.py``'s ``seeded_store`` fixture; re-implemented
    here (rather than imported) because pytest fixtures from a foreign
    rootless package are not reusable across ``tests/`` and ``evals/``.
    """

    from revenue_leakage_agent.config import AppSettings

    data_dir = tmp_path / "data"
    shutil.copytree(REPO_DATA_DIR, data_dir)
    settings = AppSettings(data_dir=data_dir, sandbox_dir=tmp_path / "sandbox")
    return JsonStore(settings)


def _tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name is not None:
            names.append(message.name)
    return names


def _assert_relative_order(actual: list[str], expected: tuple[str, ...]) -> None:
    """Every tool in ``expected`` must appear in ``actual``, in that relative order."""

    cursor = 0
    for tool_name in expected:
        try:
            cursor = actual.index(tool_name, cursor) + 1
        except ValueError:
            pytest.fail(
                f"Expected tool {tool_name!r} to appear (in order) in {actual!r}"
            )


def _ledger_count(store: JsonStore, filename: str) -> int:
    """Number of records in a sandbox ledger file; a missing file counts as 0."""

    path = store.sandbox_dir / filename
    if not path.exists():
        return 0
    payload: list[Any] = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return len(payload)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario(scenario: Scenario, tmp_path: Path) -> None:
    if not get_settings().openai_api_key:
        pytest.skip("OPENAI_API_KEY is not configured; skipping live-model evals.")

    store = _eval_store(tmp_path)
    graph = build_graph(checkpointer=InMemorySaver())
    thread = f"eval-{scenario.id}"

    result: dict[str, Any] = {}
    for turn in scenario.turns:
        result = run(
            graph, thread, store, {"messages": [HumanMessage(content=turn.user)]}
        )
        if result["interrupts"]:
            assert turn.approve is not None, (
                f"Scenario {scenario.id!r} interrupted but the turn has no "
                "approve decision."
            )
            decision = "approve" if turn.approve else "reject"
            result = run(graph, thread, store, Command(resume={"decision": decision}))

    tool_names = _tool_names(result["messages"])
    _assert_relative_order(tool_names, scenario.expected_tools)
    for forbidden in scenario.forbidden_tools:
        assert forbidden not in tool_names, (
            f"Scenario {scenario.id!r} called forbidden tool {forbidden!r} "
            f"(trajectory: {tool_names!r})"
        )

    final_reply = extract_ai_text(result["messages"][-1])
    for pattern in scenario.answer_patterns:
        assert re.search(pattern, final_reply, re.IGNORECASE), (
            f"Scenario {scenario.id!r} reply did not match {pattern!r}: {final_reply!r}"
        )

    for filename, expected_count in scenario.ledger_counts.items():
        assert _ledger_count(store, filename) == expected_count, (
            f"Scenario {scenario.id!r} expected {expected_count} record(s) in "
            f"{filename}"
        )
