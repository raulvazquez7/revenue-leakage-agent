"""End-to-end graph behaviour tests driven by scripted chat models.

Each test builds the real compiled graph (``build_graph``) with an
``InMemorySaver`` checkpointer and one independently scripted
``ScriptedChatModel`` per node (router / agent / conversational). Assertions
target graph state and sandbox/ledger files on disk, never model prose,
per the task brief's "no API key needed" and "assert on state" constraints.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import run
from fakes import ScriptedChatModel, call, route, scripted
from revenue_leakage_agent.graph import AgentGraph, build_graph
from revenue_leakage_agent.llm import AgentModels
from revenue_leakage_agent.store import JsonStore

THREAD = "flow-test"


def _graph(router: ScriptedChatModel, agent: ScriptedChatModel) -> AgentGraph:
    """A graph whose conversational model must never be consumed by default."""

    conversational = scripted(AIMessage(content="unused"))
    models = AgentModels(router=router, agent=agent, conversational=conversational)
    return build_graph(models, checkpointer=InMemorySaver())


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_conversation_route_never_consumes_agent_model(seeded_store: JsonStore) -> None:
    router = scripted(route("conversation", "capability_question"))
    conversational_reply = "I investigate revenue leakage between plans and invoices."
    conversational = scripted(AIMessage(content=conversational_reply))
    agent = scripted(AIMessage(content="agent must not run"))
    models = AgentModels(router=router, agent=agent, conversational=conversational)
    graph = build_graph(models, checkpointer=InMemorySaver())

    result = run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="what can you do?")]},
    )

    assert result["messages"][-1].content == conversational_reply
    # The agent model's scripted queue is untouched: its single message is
    # still there, proving the agent node was never invoked for this turn.
    assert next(agent.messages) == AIMessage(content="agent must not run")


def test_investigation_loop_produces_missing_invoice_finding(
    seeded_store: JsonStore,
) -> None:
    router = scripted(route("investigation", "investigation"))
    agent = scripted(
        call("load_plan", plan_id="SUB-2001"),
        call("query_invoices", plan_id="SUB-2001"),
        AIMessage(content="done"),
    )
    graph = _graph(router, agent)

    result = run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="investigate SUB-2001")]},
    )

    findings = result["findings"]
    assert len(findings) == 1
    finding = findings[0]
    assert finding["type"] == "missing_invoice"
    assert finding["amount"] == "10000.00"
    assert finding["currency"] == "USD"
    assert finding["period_start"] == "2025-06-01"
    assert result["active_scope"]["plan_id"] == "SUB-2001"


def test_approve_flow_writes_ledger_and_audit_log(
    seeded_store: JsonStore,
) -> None:
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
    graph = _graph(router, agent)
    ledger_path = seeded_store.sandbox_dir / "make_good_invoices.json"
    audit_path = seeded_store.sandbox_dir / "audit_log.json"

    turn1 = run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="draft a make-good invoice for SUB-2001")]},
    )
    assert turn1["pending_action"]["action_type"] == "make_good_invoice"
    assert not ledger_path.exists()

    turn2 = run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="apply it")]},
    )
    assert len(turn2["interrupts"]) == 1
    assert turn2["interrupts"][0].value["type"] == "approval_required"

    turn3 = run(graph, THREAD, seeded_store, Command(resume={"decision": "approve"}))

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger) == 1
    assert ledger[0]["invoice_id"] == "INV-MG-0001"

    audit_log = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_log[-1]["event_type"] == "action_applied"
    assert turn3["pending_action"] is None


def test_reject_flow_leaves_no_ledger_file(seeded_store: JsonStore) -> None:
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
        AIMessage(content="ok, cancelled"),
    )
    graph = _graph(router, agent)
    ledger_path = seeded_store.sandbox_dir / "make_good_invoices.json"

    run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="draft a make-good invoice for SUB-2001")]},
    )
    run(graph, THREAD, seeded_store, {"messages": [HumanMessage(content="apply it")]})
    final = run(graph, THREAD, seeded_store, Command(resume={"decision": "reject"}))

    assert not ledger_path.exists()
    assert final["applied_actions"][-1]["status"] == "rejected"


def test_rollback_removes_ledger_record_and_marks_action_rolled_back(
    seeded_store: JsonStore,
) -> None:
    router = scripted(*(route("investigation", "investigation") for _ in range(3)))
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
        call("rollback"),
        AIMessage(content="rolled back"),
    )
    graph = _graph(router, agent)
    ledger_path = seeded_store.sandbox_dir / "make_good_invoices.json"
    audit_path = seeded_store.sandbox_dir / "audit_log.json"

    run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="draft a make-good invoice for SUB-2001")]},
    )
    run(graph, THREAD, seeded_store, {"messages": [HumanMessage(content="apply it")]})
    run(graph, THREAD, seeded_store, Command(resume={"decision": "approve"}))
    run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="roll it back")]},
    )
    final = run(graph, THREAD, seeded_store, Command(resume={"decision": "approve"}))

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger == []

    audit_log = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_log[-1]["event_type"] == "action_rolled_back"

    assert [a["status"] for a in final["applied_actions"]] == ["rolled_back"]


def test_double_apply_is_rejected_without_a_second_interrupt(
    seeded_store: JsonStore,
) -> None:
    """Review Focus 3: an already-applied action must not mutate the ledger again."""

    router = scripted(*(route("investigation", "investigation") for _ in range(3)))
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
        call("apply"),
        AIMessage(content="already applied"),
    )
    graph = _graph(router, agent)
    ledger_path = seeded_store.sandbox_dir / "make_good_invoices.json"

    run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="draft a make-good invoice for SUB-2001")]},
    )
    run(graph, THREAD, seeded_store, {"messages": [HumanMessage(content="apply it")]})
    run(graph, THREAD, seeded_store, Command(resume={"decision": "approve"}))

    second_apply = run(
        graph,
        THREAD,
        seeded_store,
        {"messages": [HumanMessage(content="apply it again")]},
    )

    tool_messages = _tool_messages(second_apply)
    payload = json.loads(str(tool_messages[-1].content))
    assert payload["ok"] is False
    assert payload["error"]["error_code"] == "ACTION_NOT_FOUND"
    assert not second_apply["interrupts"]

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger) == 1
