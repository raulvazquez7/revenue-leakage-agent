"""Pure helpers behind the Streamlit chat: stream text, interrupts, approval card."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Interrupt

from revenue_leakage_agent.interfaces.streaming import (
    TurnStream,
    approval_card,
    interrupt_payload,
    stream_text,
)


def _msg(message: Any, node: str) -> tuple[str, Any]:
    return ("messages", (message, {"langgraph_node": node}))


def test_stream_text_keeps_chunks_from_answering_nodes() -> None:
    assert stream_text((AIMessageChunk(content="Hi"), {"langgraph_node": "agent"})) == (
        "Hi"
    )
    assert (
        stream_text((AIMessageChunk(content="yo"), {"langgraph_node": "conversation"}))
        == "yo"
    )


def test_stream_text_drops_router_tools_and_whole_messages() -> None:
    assert (
        stream_text((AIMessageChunk(content="{"), {"langgraph_node": "router"})) == ""
    )
    tool_message = ToolMessage(content="{}", tool_call_id="t1")
    assert stream_text((tool_message, {"langgraph_node": "tools"})) == ""
    assert stream_text((AIMessage(content="dup"), {"langgraph_node": "agent"})) == ""


def test_stream_text_reads_typed_content_blocks_and_skips_tool_call_chunks() -> None:
    blocks = AIMessageChunk(content=[{"type": "text", "text": "Found", "index": 0}])
    assert stream_text((blocks, {"langgraph_node": "agent"})) == "Found"
    tool_call_chunk = AIMessageChunk(
        content=[],
        tool_call_chunks=[{"name": "load_plan", "args": "{", "id": "c1", "index": 0}],
    )
    assert stream_text((tool_call_chunk, {"langgraph_node": "agent"})) == ""


def test_interrupt_payload_from_updates_chunk() -> None:
    value = {"type": "approval_required", "action": {"action_id": "A1"}}
    chunk = {"__interrupt__": (Interrupt(value=value, id="i1"),)}

    assert interrupt_payload(chunk) == value
    assert interrupt_payload({"agent": {"messages": []}}) is None
    wrapped = interrupt_payload({"__interrupt__": (Interrupt(value="ok?", id="i2"),)})
    assert wrapped == {"value": "ok?"}


def test_turn_stream_yields_text_and_captures_interrupt() -> None:
    value: dict[str, Any] = {"type": "approval_required", "action": {}}
    events: list[tuple[str, Any]] = [
        ("updates", {"router": {"route_decision": {}}}),
        _msg(AIMessageChunk(content="Checking", id="m1"), "agent"),
        _msg(AIMessageChunk(content=" now.", id="m1"), "agent"),
        _msg(AIMessageChunk(content="Drafted.", id="m2"), "agent"),
        ("updates", {"__interrupt__": (Interrupt(value=value, id="i1"),)}),
    ]
    turn = TurnStream()

    text = "".join(turn.text(events))

    # A new model message starts a new paragraph instead of gluing sentences.
    assert text == "Checking now.\n\nDrafted."
    assert turn.interrupt == value


def test_approval_card_for_make_good_invoice() -> None:
    payload = {
        "type": "approval_required",
        "question": "Approve this sandbox mutation?",
        "action": {
            "action_id": "DRAFT-MG-1",
            "action_type": "make_good_invoice",
            "plan_id": "SUB-2001",
            "amount": "10000.00",
            "currency": "USD",
            "reason": "Missing June invoice.",
            "evidence": ["Expected 10000 USD", "No invoice found"],
        },
    }

    card = approval_card(payload)

    assert card.title == "Make-good invoice"
    assert card.question == "Approve this sandbox mutation?"
    assert card.fields == [
        ("Action ID", "DRAFT-MG-1"),
        ("Plan", "SUB-2001"),
        ("Amount", "10000.00 USD"),
    ]
    assert card.reason == "Missing June invoice."
    assert card.evidence == ["Expected 10000 USD", "No invoice found"]
    assert card.raw == payload["action"]


def test_approval_card_for_credit_memo_and_rollback() -> None:
    memo = approval_card(
        {
            "action": {
                "action_type": "credit_memo",
                "invoice_id": "INV-5022",
                "plan_id": "",
                "amount": "1200",
                "currency": "USD",
            }
        }
    )
    assert memo.title == "Credit memo"
    assert memo.question == "Approve this sandbox mutation?"
    assert ("Invoice", "INV-5022") in memo.fields
    assert all(label != "Plan" for label, _ in memo.fields)

    rollback = approval_card(
        {
            "question": "Approve rolling back this applied action?",
            "action": {
                "action_id": "DRAFT-CM-1",
                "action_type": "credit_memo",
                "status": "rollback_requested",
                "sandbox_record_id": "CM-0001",
            },
        }
    )
    assert rollback.title == "Roll back: Credit memo"
    assert ("Sandbox record", "CM-0001") in rollback.fields
    assert rollback.evidence == []


def test_approval_card_formats_plan_amendment_change_set() -> None:
    card = approval_card(
        {
            "action": {
                "action_type": "plan_amendment",
                "plan_id": "SUB-2014",
                "change_set": {"cadence": "Quarterly", "total_value": 96000},
            }
        }
    )

    assert card.title == "Plan amendment"
    assert ("Change set", "cadence: Quarterly; total_value: 96000") in card.fields


def test_approval_card_tolerates_unexpected_payload() -> None:
    card = approval_card({"value": "approve?"})

    assert card.title == "Sandbox action"
    assert card.fields == []
    assert card.raw == {}
