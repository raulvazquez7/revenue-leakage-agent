from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from typing import Any, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.messages import extract_ai_text

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)


def main() -> None:
    graph = build_graph()
    thread_id = f"cli-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}

    print("Revenue Leakage Agent CLI. Type 'exit' to quit.")
    print(f"Thread ID: {thread_id}")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        interrupt_payload = _run_stream(
            graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode="updates",
            )
        )
        while interrupt_payload is not None:
            action_raw = interrupt_payload.get("action", {})
            action = (
                cast(dict[str, Any], action_raw) if isinstance(action_raw, dict) else {}
            )
            print("\nApproval required:")
            print(f"- Action: {action.get('action_type')}")
            if action.get("plan_id"):
                print(f"- Plan: {action.get('plan_id')}")
            if action.get("invoice_id"):
                print(f"- Invoice: {action.get('invoice_id')}")
            if action.get("amount") is not None:
                print(f"- Amount: {action.get('amount')} {action.get('currency')}")
            if action.get("change_set") is not None:
                print(f"- Change set: {_to_json(action.get('change_set'))}")
            if action.get("sandbox_record_id"):
                print(f"- Sandbox record: {action.get('sandbox_record_id')}")
            if action.get("reason"):
                print(f"- Reason: {action.get('reason')}")
            decision = input("Approve? [approve/reject]: ").strip().lower()
            interrupt_payload = _run_stream(
                graph.stream(
                    Command(resume={"decision": decision}),
                    config,
                    stream_mode="updates",
                )
            )


def _run_stream(stream: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for chunk in stream:
        print(f"\n[trace] chunk={_to_json(chunk)}")
        if "__interrupt__" in chunk:
            interrupts = chunk["__interrupt__"]
            first_interrupt = interrupts[0]
            value = first_interrupt.value
            print(f"[interrupt] {_to_json(value)}")
            return (
                cast(dict[str, Any], value)
                if isinstance(value, dict)
                else {"value": value}
            )

        for node_name, update in chunk.items():
            print(f"[node] {node_name}")
            if not isinstance(update, dict):
                continue
            update_map = cast(dict[str, Any], update)
            _print_state_fields(update_map)
            messages_raw = update_map.get("messages", [])
            if not isinstance(messages_raw, list):
                continue
            messages = cast(list[Any], messages_raw)
            for message in messages:
                if isinstance(message, AIMessage):
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        print(f"[ai_tool_calls] {_to_json(tool_calls)}")
                    text = extract_ai_text(message)
                    if text:
                        print(f"\nAgent: {text}")
                elif isinstance(message, ToolMessage):
                    print(f"[tool:{message.name or 'unknown'}] {message.content}")
    return None


def _print_state_fields(update: dict[str, Any]) -> None:
    for key in (
        "route_decision",
        "active_scope",
        "findings",
        "pending_action",
        "applied_actions",
        "last_error",
    ):
        if key in update:
            print(f"[state:{key}] {_to_json(update[key])}")


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
