"""UI-agnostic helpers for streaming graph turns and rendering approvals.

Kept free of Streamlit so the logic is unit-testable; the Streamlit app only
feeds :meth:`TurnStream.text` to ``st.write_stream`` and draws
:func:`approval_card`'s fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph
from revenue_leakage_agent.state import AgentState

# Nodes whose model output is user-facing. The router's structured-output
# tokens are internal and must never reach the chat.
ANSWER_NODES = frozenset({"agent", "conversation"})

StreamEvent = tuple[str, Any]


def stream_events(
    graph: AgentGraph,
    payload: AgentState | Command[Any],
    config: RunnableConfig,
    context: AgentContext,
) -> Iterator[StreamEvent]:
    """Stream a turn as ``(mode, chunk)`` pairs: model tokens plus node updates."""

    stream = graph.stream(  # pyright: ignore[reportUnknownMemberType]
        payload,
        config,
        context=context,
        stream_mode=["messages", "updates"],
    )
    yield from cast(Iterable[StreamEvent], stream)


def stream_text(chunk: Any) -> str:
    """Visible text of one ``messages``-mode chunk, or ``""`` to skip it.

    Only token chunks from answer nodes count. Whole messages (emitted for
    node outputs), tool messages, router tokens and tool-call chunks are dropped.
    """

    message, metadata = cast(tuple[Any, Mapping[str, Any]], chunk)
    if not isinstance(message, AIMessageChunk):
        return ""
    if metadata.get("langgraph_node") not in ANSWER_NODES:
        return ""
    return message.text


def turn_error_message(error: BaseException) -> str:
    """User-facing text for a turn that raised, naming the exception type.

    Kept pure and separate from the Streamlit call site so it is unit
    -testable without a Streamlit runtime.
    """

    return (
        f"Something went wrong while running the agent: {type(error).__name__}. "
        "Check the terminal logs."
    )


def interrupt_payload(update: Any) -> dict[str, Any] | None:
    """The first interrupt's value from an ``updates``-mode chunk, if any."""

    if not isinstance(update, dict):
        return None
    interrupts = cast(dict[str, Any], update).get("__interrupt__")
    if not interrupts:
        return None
    value = cast(tuple[Any, ...], interrupts)[0].value
    return cast(dict[str, Any], value) if isinstance(value, dict) else {"value": value}


@dataclass
class TurnStream:
    """Turns graph stream events into chat text, capturing the interrupt aside.

    ``st.write_stream`` only consumes strings, so the approval request that
    pauses the turn is recorded on :attr:`interrupt` while text is yielded.
    """

    interrupt: dict[str, Any] | None = None
    _last_message_id: str | None = field(default=None, repr=False)
    _wrote_text: bool = field(default=False, repr=False)

    def text(self, events: Iterable[StreamEvent]) -> Iterator[str]:
        for mode, chunk in events:
            if mode == "updates":
                payload = interrupt_payload(chunk)
                if payload is not None:
                    self.interrupt = payload
                continue
            if mode != "messages":
                continue
            text = stream_text(chunk)
            if not text:
                continue
            message_id = cast(tuple[AIMessageChunk, Any], chunk)[0].id
            if self._wrote_text and message_id != self._last_message_id:
                # A new model call in the same turn: start a new paragraph.
                yield "\n\n"
            self._last_message_id = message_id
            self._wrote_text = True
            yield text


@dataclass(frozen=True)
class ApprovalCard:
    """Human-readable view of an ``approval_required`` interrupt payload."""

    title: str
    question: str
    fields: list[tuple[str, str]]
    reason: str | None
    evidence: list[str]
    raw: dict[str, Any]


_ACTION_TITLES = {
    "make_good_invoice": "Make-good invoice",
    "credit_memo": "Credit memo",
    "plan_amendment": "Plan amendment",
}
_DEFAULT_QUESTION = "Approve this sandbox mutation?"


def approval_card(payload: Mapping[str, Any]) -> ApprovalCard:
    """Build the approval card fields from an interrupt payload.

    Tolerates missing keys so an unexpected payload still renders (the raw
    action is always available for an expander).
    """

    action_raw = payload.get("action")
    action = cast(dict[str, Any], action_raw) if isinstance(action_raw, dict) else {}
    action_type = str(action.get("action_type") or "")
    title = _ACTION_TITLES.get(action_type, action_type.replace("_", " ").capitalize())
    title = title or "Sandbox action"
    if action.get("status") == "rollback_requested":
        title = f"Roll back: {title}"

    fields: list[tuple[str, str]] = []
    for key, label in (
        ("action_id", "Action ID"),
        ("plan_id", "Plan"),
        ("invoice_id", "Invoice"),
    ):
        if action.get(key):
            fields.append((label, str(action[key])))
    if action.get("amount") is not None:
        amount = f"{action['amount']} {action.get('currency') or ''}".strip()
        fields.append(("Amount", amount))
    change_set = action.get("change_set")
    if isinstance(change_set, dict) and change_set:
        changes = cast(dict[str, Any], change_set)
        fields.append(
            ("Change set", "; ".join(f"{k}: {v}" for k, v in changes.items()))
        )
    if action.get("sandbox_record_id"):
        fields.append(("Sandbox record", str(action["sandbox_record_id"])))

    evidence_raw = action.get("evidence")
    evidence = (
        [str(item) for item in cast(list[Any], evidence_raw)]
        if isinstance(evidence_raw, list)
        else []
    )
    reason = action.get("reason")
    return ApprovalCard(
        title=title,
        question=str(payload.get("question") or _DEFAULT_QUESTION),
        fields=fields,
        reason=str(reason) if reason else None,
        evidence=evidence,
        raw=action,
    )
