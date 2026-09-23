"""Streamlit chat UI. Run with ``uv run task ui``.

The graph (and its checkpointer, e.g. one SQLite connection) is built once per
process; each browser session only keeps its own thread id and chat log.
Stream/approval logic lives in :mod:`revenue_leakage_agent.interfaces.streaming`.
"""

from __future__ import annotations

import warnings
from typing import Any, cast
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph, build_graph
from revenue_leakage_agent.interfaces.streaming import (
    TurnStream,
    approval_card,
    stream_events,
)
from revenue_leakage_agent.persistence import build_checkpointer
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)

EXAMPLE_PROMPTS = (
    "What can you do?",
    "Investigate SUB-2001",
    "Is the SUB-2014-A1 invoice billed in the right currency?",
    "Are there payments without a contract?",
)

SETUP_HELP = """
**The agent could not start:** {error}

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Restart the app (`uv run task ui`, or `docker compose up` for containers).
"""


@st.cache_resource(show_spinner="Starting the agent...")
def _load_graph() -> AgentGraph:
    return build_graph(checkpointer=build_checkpointer(get_settings()))


@st.cache_resource
def _load_store() -> JsonStore:
    return JsonStore(get_settings())


def _new_conversation() -> None:
    st.session_state.thread_id = f"streamlit-{uuid4()}"
    st.session_state.chat_messages = []
    st.session_state.pending_interrupt = None


def _thread_id() -> str:
    return cast(str, st.session_state.thread_id)


def _chat_messages() -> list[dict[str, str]]:
    return cast(list[dict[str, str]], st.session_state.chat_messages)


def _pending_interrupt() -> dict[str, Any] | None:
    return cast(dict[str, Any] | None, st.session_state.pending_interrupt)


def _config() -> RunnableConfig:
    return {"configurable": {"thread_id": _thread_id()}}


def _run_turn(graph: AgentGraph, payload: AgentState | Command[Any]) -> None:
    """Stream one turn into an assistant bubble, then record reply + interrupt."""

    turn = TurnStream()
    context = AgentContext(store=_load_store())
    with st.chat_message("assistant"), st.spinner("Working..."):
        streamed = st.write_stream(
            turn.text(stream_events(graph, payload, _config(), context))
        )
    reply = streamed if isinstance(streamed, str) else ""
    if reply.strip():
        _chat_messages().append({"role": "assistant", "content": reply})

    interrupt = turn.interrupt
    if interrupt is None:
        # Fallback: the checkpoint is the source of truth for a paused turn.
        interrupts = graph.get_state(_config()).interrupts
        if interrupts:
            value = interrupts[0].value
            interrupt = (
                cast(dict[str, Any], value)
                if isinstance(value, dict)
                else {"value": value}
            )
    st.session_state.pending_interrupt = interrupt
    st.rerun()


def _render_sidebar() -> None:
    store = _load_store()
    with st.sidebar:
        st.header("Session")
        st.caption("Thread ID")
        st.code(_thread_id(), language=None)
        if st.button("New conversation", width="stretch"):
            _new_conversation()
            st.rerun()

        entries = store.load_audit_log()
        with st.expander(f"Audit log ({len(entries)})"):
            if not entries:
                st.caption("No sandbox actions yet.")
            for entry in reversed(entries):
                action_raw = entry.get("action")
                action = (
                    cast(dict[str, Any], action_raw)
                    if isinstance(action_raw, dict)
                    else entry
                )
                st.markdown(
                    f"**{entry.get('audit_id', '?')}** · "
                    f"`{entry.get('event_type', '?')}` · "
                    f"{action.get('action_id', '')}"
                )
                st.caption(str(entry.get("created_at", "")))

        if st.button(
            "Reset sandbox",
            width="stretch",
            help="Delete sandbox ledgers and the audit log, then start a new "
            "conversation. The read-only dataset is untouched.",
        ):
            removed = store.reset_sandbox()
            _new_conversation()
            st.toast(f"Sandbox reset: removed {len(removed)} file(s).")
            st.rerun()


def _render_approval(graph: AgentGraph, payload: dict[str, Any]) -> None:
    card = approval_card(payload)
    with st.container(border=True):
        st.subheader(f"Approval required: {card.title}")
        st.write(card.question)
        for label, value in card.fields:
            st.markdown(f"**{label}:** {value}")
        if card.reason:
            st.markdown(f"**Reason:** {card.reason}")
        if card.evidence:
            st.markdown("**Evidence:**")
            st.markdown("\n".join(f"- {item}" for item in card.evidence))
        with st.expander("Raw action"):
            st.json(card.raw)
        approve_col, reject_col = st.columns(2)
        decision: str | None = None
        with approve_col:
            if st.button("Approve", type="primary", width="stretch"):
                decision = "approve"
        with reject_col:
            if st.button("Reject", width="stretch"):
                decision = "reject"
    if decision is not None:
        st.session_state.pending_interrupt = None
        _run_turn(graph, Command(resume={"decision": decision}))


def main() -> None:
    st.set_page_config(page_title="Revenue Leakage Agent", page_icon="$")
    st.title("Revenue Leakage Agent")

    try:
        graph = _load_graph()
    except RuntimeError as error:
        st.error(SETUP_HELP.format(error=error))
        st.stop()

    if "thread_id" not in st.session_state:
        _new_conversation()
    _render_sidebar()

    for message in _chat_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = _pending_interrupt()
    if pending:
        _render_approval(graph, pending)

    prompt = st.chat_input(
        "Ask about a plan, e.g. 'Investigate SUB-2001'",
        disabled=pending is not None,
    )
    if not _chat_messages() and pending is None:
        st.caption("Try one of these:")
        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, example in zip(columns, EXAMPLE_PROMPTS, strict=True):
            if column.button(example, width="stretch"):
                prompt = example

    if prompt:
        _chat_messages().append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        _run_turn(graph, {"messages": [HumanMessage(content=prompt)]})


main()
