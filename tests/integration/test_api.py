"""HTTP API behaviour with scripted chat models (no API key needed)."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from fakes import ScriptedChatModel, call, route, scripted
from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.interfaces.api import create_app
from revenue_leakage_agent.llm import AgentModels
from revenue_leakage_agent.persistence import build_checkpointer
from revenue_leakage_agent.store import JsonStore


def _client(
    store: JsonStore,
    *,
    router: ScriptedChatModel,
    agent: ScriptedChatModel | None = None,
    conversational: ScriptedChatModel | None = None,
) -> Iterator[TestClient]:
    models = AgentModels(
        router=router,
        agent=agent or scripted(AIMessage(content="agent must not run")),
        conversational=conversational or scripted(AIMessage(content="unused")),
    )
    graph = build_graph(models, checkpointer=InMemorySaver())
    with TestClient(create_app(graph=graph, store=store)) as client:
        yield client


@pytest.fixture
def chat_client(seeded_store: JsonStore) -> Iterator[TestClient]:
    yield from _client(
        seeded_store,
        router=scripted(route("conversation", "chit_chat")),
        conversational=scripted(AIMessage(content="hello")),
    )


def _new_thread(client: TestClient) -> str:
    response = client.post("/threads")
    assert response.status_code == 200
    return str(response.json()["thread_id"])


def test_health(chat_client: TestClient) -> None:
    response = chat_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_message_returns_only_this_turns_replies(chat_client: TestClient) -> None:
    thread_id = _new_thread(chat_client)

    response = chat_client.post(
        f"/threads/{thread_id}/messages", json={"content": "hi"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "thread_id": thread_id,
        "replies": ["hello"],
        "interrupt": None,
    }


def test_unknown_thread_is_404(chat_client: TestClient) -> None:
    thread_id = _new_thread(chat_client)

    assert chat_client.get(f"/threads/{thread_id}").status_code == 404
    response = chat_client.post(
        f"/threads/{thread_id}/resume", json={"decision": "approve"}
    )
    assert response.status_code == 404


def test_resume_without_pending_interrupt_is_409(chat_client: TestClient) -> None:
    thread_id = _new_thread(chat_client)
    chat_client.post(f"/threads/{thread_id}/messages", json={"content": "hi"})

    response = chat_client.post(
        f"/threads/{thread_id}/resume", json={"decision": "approve"}
    )

    assert response.status_code == 409


def test_invalid_decision_is_rejected(chat_client: TestClient) -> None:
    thread_id = _new_thread(chat_client)

    response = chat_client.post(
        f"/threads/{thread_id}/resume", json={"decision": "maybe"}
    )

    assert response.status_code == 422


def test_approval_flow_over_http_writes_ledger(seeded_store: JsonStore) -> None:
    router = scripted(
        route("investigation", "investigation"),
        route("investigation", "investigation"),
    )
    agent = scripted(
        call(
            "propose_make_good_invoice",
            plan_id="SUB-2001",
            amount="10000",
            reason="Missing June invoice",
        ),
        AIMessage(content="drafted, please approve"),
        call("apply"),
        AIMessage(content="applied"),
    )
    ledger_path = seeded_store.sandbox_dir / "make_good_invoices.json"

    for client in _client(seeded_store, router=router, agent=agent):
        thread_id = _new_thread(client)
        drafted = client.post(
            f"/threads/{thread_id}/messages",
            json={"content": "draft a make-good invoice for SUB-2001"},
        ).json()
        assert drafted["replies"] == ["drafted, please approve"]
        assert drafted["interrupt"] is None

        pending = client.get(f"/threads/{thread_id}").json()
        assert pending["pending_action"]["action_type"] == "make_good_invoice"

        applying = client.post(
            f"/threads/{thread_id}/messages", json={"content": "apply it"}
        ).json()
        assert applying["interrupt"]["type"] == "approval_required"
        assert applying["replies"] == []
        assert not ledger_path.exists()
        assert client.get(f"/threads/{thread_id}").json()["interrupt"] is not None

        resumed = client.post(
            f"/threads/{thread_id}/resume", json={"decision": "approve"}
        ).json()
        assert resumed["interrupt"] is None
        assert resumed["replies"] == ["applied"]

        final = client.get(f"/threads/{thread_id}").json()
        assert final["pending_action"] is None
        assert final["interrupt"] is None
        assert [a["status"] for a in final["applied_actions"]] == ["applied"]

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert [row["invoice_id"] for row in ledger] == ["INV-MG-0001"]


def test_default_app_imports_without_api_key(tmp_path: Path) -> None:
    """Building the module-level app must not construct models.

    Runs in a fresh interpreter from an empty directory so neither an already
    imported module nor a local ``.env`` can mask a missing key.
    """

    env = {**os.environ, "OPENAI_API_KEY": ""}
    result = subprocess.run(
        [sys.executable, "-c", "import revenue_leakage_agent.interfaces.api"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_injected_graph_without_checkpointer_is_rejected(
    seeded_store: JsonStore,
) -> None:
    models = AgentModels(
        router=scripted(route("conversation", "chit_chat")),
        agent=scripted(AIMessage(content="unused")),
        conversational=scripted(AIMessage(content="unused")),
    )

    with pytest.raises(ValueError, match="checkpointer"):
        create_app(graph=build_graph(models), store=seeded_store)


def test_message_while_approval_pending_is_409(seeded_store: JsonStore) -> None:
    router = scripted(
        route("investigation", "investigation"),
        route("investigation", "investigation"),
    )
    agent = scripted(
        call(
            "propose_make_good_invoice",
            plan_id="SUB-2001",
            amount="10000",
            reason="Missing June invoice",
        ),
        AIMessage(content="drafted, please approve"),
        call("apply"),
        AIMessage(content="applied"),
    )

    for client in _client(seeded_store, router=router, agent=agent):
        thread_id = _new_thread(client)
        client.post(
            f"/threads/{thread_id}/messages",
            json={"content": "draft a make-good invoice for SUB-2001"},
        )
        applying = client.post(
            f"/threads/{thread_id}/messages", json={"content": "apply it"}
        ).json()
        assert applying["interrupt"] is not None

        response = client.post(
            f"/threads/{thread_id}/messages", json={"content": "something else"}
        )

        assert response.status_code == 409
        assert client.get(f"/threads/{thread_id}").json()["interrupt"] is not None


def test_default_lifespan_uses_sqlite_and_closes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seeded_store: JsonStore
) -> None:
    from revenue_leakage_agent import graph as graph_module
    from revenue_leakage_agent.interfaces import api

    settings = AppSettings(checkpoint_db=tmp_path / "checkpoints.sqlite")
    savers: list[BaseCheckpointSaver[Any]] = []

    def recording_checkpointer(s: AppSettings) -> BaseCheckpointSaver[Any]:
        savers.append(build_checkpointer(s))
        return savers[-1]

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "build_checkpointer", recording_checkpointer)
    monkeypatch.setattr(
        graph_module,
        "build_default_models",
        lambda: AgentModels(
            router=scripted(route("conversation", "chit_chat")),
            agent=scripted(AIMessage(content="agent must not run")),
            conversational=scripted(AIMessage(content="hello")),
        ),
    )

    with TestClient(api.create_app(store=seeded_store)) as client:
        thread_id = _new_thread(client)
        reply = client.post(f"/threads/{thread_id}/messages", json={"content": "hi"})
        assert reply.json()["replies"] == ["hello"]

    saver = savers[0]
    assert isinstance(saver, SqliteSaver)
    with pytest.raises(sqlite3.ProgrammingError):
        saver.conn.execute("select 1")
