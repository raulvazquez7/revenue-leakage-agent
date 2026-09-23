from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from revenue_leakage_agent.context import AgentContext, resolve_store
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore
from revenue_leakage_agent.tools import (
    _is_approved,  # pyright: ignore[reportPrivateUsage]
    get_tools,
)

TEST_PLAN = {
    "plan_id": "SUB-TEST",
    "customer_name": "Context Store Customer",
    "total_value": 12000,
    "currency": "USD",
    "cadence": "Monthly",
    "start_date": "2025-01-01",
    "entitlements": ["Analytics"],
}


def _seed_plan(store: JsonStore) -> None:
    store.data_dir.mkdir(parents=True, exist_ok=True)
    (store.data_dir / "billing_plans.json").write_text(
        json.dumps([TEST_PLAN]), encoding="utf-8"
    )


def _tool_graph(checkpointer: InMemorySaver | None = None) -> Any:
    builder: Any = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("tools", ToolNode(get_tools(), handle_tool_errors=True))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=checkpointer)


def _run_tool(
    store: JsonStore,
    name: str,
    args: dict[str, Any],
    **state: Any,
) -> dict[str, Any]:
    """Execute one tool call through a real ToolNode with runtime context."""

    graph = _tool_graph()
    call = {"name": name, "args": args, "id": "call-1", "type": "tool_call"}
    return graph.invoke(
        {"messages": [AIMessage(content="", tool_calls=[call])], **state},
        context=AgentContext(store=store),
    )


def _tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call-1"
    return json.loads(str(message.content))


def test_load_plan_reads_from_context_store(store: JsonStore) -> None:
    _seed_plan(store)

    result = _run_tool(store, "load_plan", {"plan_id": "sub-test"})

    assert _tool_payload(result)["plan"]["customer_name"] == "Context Store Customer"
    assert result["active_scope"]["plan_id"] == "SUB-TEST"


def test_load_plan_reports_missing_plan(store: JsonStore) -> None:
    _seed_plan(store)

    result = _run_tool(store, "load_plan", {"plan_id": "SUB-2001"})

    assert _tool_payload(result)["error"]["error_code"] == "PLAN_NOT_FOUND"
    assert result["last_error"]["error_code"] == "PLAN_NOT_FOUND"


def test_propose_make_good_invoice_uses_state_findings(store: JsonStore) -> None:
    _seed_plan(store)
    findings = [
        {"plan_id": "SUB-TEST", "amount": "1000.00", "evidence": ["June missing."]}
    ]

    result = _run_tool(
        store,
        "propose_make_good_invoice",
        {"plan_id": "SUB-TEST", "amount": "1000.00", "reason": "Missed June."},
        findings=findings,
    )

    assert result["pending_action"]["action_type"] == "make_good_invoice"
    assert result["pending_action"]["evidence"] == ["June missing."]


def test_resolve_store_prefers_context_and_falls_back_to_default(
    store: JsonStore,
) -> None:
    assert resolve_store(AgentContext(store=store)) is store
    assert isinstance(resolve_store(None), JsonStore)


def test_apply_interrupts_then_writes_to_context_store(store: JsonStore) -> None:
    draft = {
        "action_id": "DRAFT-MG-TEST",
        "action_type": "make_good_invoice",
        "plan_id": "SUB-TEST",
        "amount": "1000.00",
        "currency": "USD",
        "reason": "Missed June.",
    }
    graph = _tool_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "apply-test"}}
    context = AgentContext(store=store)
    call: dict[str, Any] = {"name": "apply", "args": {}, "id": "call-1"}

    paused = graph.invoke(
        {
            "messages": [AIMessage(content="", tool_calls=[call])],
            "pending_action": draft,
        },
        config,
        context=context,
    )
    assert paused["__interrupt__"][0].value["type"] == "approval_required"
    assert not store.get_action_already_applied("DRAFT-MG-TEST")

    resumed = graph.invoke(
        Command(resume={"decision": "approve"}), config, context=context
    )

    assert _tool_payload(resumed)["status"] == "applied"
    assert resumed["pending_action"] is None
    assert store.get_action_already_applied("DRAFT-MG-TEST")


def test_load_plan_normalizes_whitespace_and_case(store: JsonStore) -> None:
    """Review Focus 4: plan IDs are trimmed and upper-cased before lookup."""

    _seed_plan(store)

    result = _run_tool(store, "load_plan", {"plan_id": " sub-test "})

    assert _tool_payload(result)["plan"]["plan_id"] == "SUB-TEST"
    assert result["active_scope"]["plan_id"] == "SUB-TEST"


def test_fx_convert_missing_rate_reports_error_and_completes_turn(
    seeded_store: JsonStore,
) -> None:
    """Review Focus 2: ToolNode(handle_tool_errors=True) turns the raised
    ValueError into an error ToolMessage instead of failing the graph run."""

    result = _run_tool(
        seeded_store,
        "fx_convert",
        {
            "amount": "100",
            "from_ccy": "EUR",
            "to_ccy": "USD",
            "on_date": "2099-01-01",
        },
    )

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "No FX rate for EUR->USD on 2099-01-01" in str(message.content)


@pytest.mark.parametrize(
    "decision",
    ["Sí", "yes", "APPROVE", {"decision": "approved"}],
)
def test_is_approved_true_cases(decision: Any) -> None:
    assert _is_approved(decision) is True


@pytest.mark.parametrize(
    "decision",
    ["no", "", {"decision": "maybe"}],
)
def test_is_approved_false_cases(decision: Any) -> None:
    assert _is_approved(decision) is False
