from __future__ import annotations

import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.graph import build_graph  # noqa: E402
from agents.messages import extract_ai_text  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
)

st.set_page_config(page_title="Revenue Leakage Agent", page_icon="$")
st.title("Revenue Leakage Agent")


def _init_session() -> None:
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid4()}"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "pending_interrupt" not in st.session_state:
        st.session_state.pending_interrupt = None


def _config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": str(st.session_state.thread_id)}}


def _chat_messages() -> list[dict[str, str]]:
    return cast(list[dict[str, str]], st.session_state.chat_messages)


def _graph() -> Any:
    return st.session_state.graph


def _run_stream(stream: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    interrupt_payload: dict[str, Any] | None = None
    for chunk in stream:
        if "__interrupt__" in chunk:
            first_interrupt = chunk["__interrupt__"][0]
            value = first_interrupt.value
            interrupt_payload = (
                cast(dict[str, Any], value)
                if isinstance(value, dict)
                else {"value": value}
            )
            continue

        for update in chunk.values():
            if not isinstance(update, dict):
                continue
            update_map = cast(dict[str, Any], update)
            messages_raw = update_map.get("messages", [])
            if not isinstance(messages_raw, list):
                continue
            messages = cast(list[Any], messages_raw)
            for message in messages:
                if isinstance(message, AIMessage):
                    text = extract_ai_text(message)
                    if text:
                        _chat_messages().append({"role": "assistant", "content": text})
    return interrupt_payload


def _run_user_message(content: str) -> dict[str, Any] | None:
    return _run_stream(
        _graph().stream(
            {"messages": [HumanMessage(content=content)]},
            _config(),
            stream_mode="updates",
        )
    )


def _resume(decision: str) -> None:
    interrupt_payload = _run_stream(
        _graph().stream(
            Command(resume={"decision": decision}),
            _config(),
            stream_mode="updates",
        )
    )
    st.session_state.pending_interrupt = interrupt_payload


_init_session()

for message in _chat_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_interrupt:
    payload = cast(dict[str, Any], st.session_state.pending_interrupt)
    action_raw = payload.get("action", {})
    action = cast(dict[str, Any], action_raw) if isinstance(action_raw, dict) else {}
    st.subheader("Approval Required")
    st.write(payload.get("question", "Approve this sandbox mutation?"))
    st.json(action)
    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button("Approve", type="primary"):
            _resume("approve")
            st.rerun()
    with reject_col:
        if st.button("Reject"):
            _resume("reject")
            st.rerun()

if prompt := st.chat_input("Ask about a plan, e.g. 'Investigate SUB-2001'"):
    _chat_messages().append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with (
        st.chat_message("assistant"),
        st.status(
            "Thinking...",
            expanded=False,
        ) as status,
    ):
        interrupt_payload = _run_user_message(prompt)
        if interrupt_payload:
            status.update(label="Waiting for approval...", state="complete")
        else:
            status.update(label="Done", state="complete")
    st.session_state.pending_interrupt = interrupt_payload
    st.rerun()
