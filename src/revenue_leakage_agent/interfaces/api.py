"""Thin HTTP API over the graph: threads, turns and approval resumes.

Run with ``uv run task api``. Endpoints are sync because the graph is sync;
FastAPI runs them in its threadpool.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel, Field

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import AgentGraph, build_graph
from revenue_leakage_agent.messages import extract_ai_text
from revenue_leakage_agent.persistence import build_checkpointer
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.store import JsonStore


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ThreadCreated(BaseModel):
    thread_id: str


class MessageRequest(BaseModel):
    content: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    decision: Literal["approve", "reject"]


class TurnResponse(BaseModel):
    thread_id: str
    replies: list[str] = Field(description="AI replies produced during this turn.")
    interrupt: dict[str, Any] | None = Field(
        description="Pending approval request, if the turn paused for one."
    )


class ThreadState(BaseModel):
    active_scope: dict[str, Any] | None
    findings: list[dict[str, Any]]
    pending_action: dict[str, Any] | None
    applied_actions: list[dict[str, Any]]
    interrupt: dict[str, Any] | None


def create_app(
    graph: AgentGraph | None = None, store: JsonStore | None = None
) -> FastAPI:
    """Build the API app.

    Missing dependencies are built in the lifespan (not at import), so importing
    this module never needs ``OPENAI_API_KEY``. The default graph uses
    :func:`build_checkpointer`; a SQLite connection it opens is closed on
    shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        settings = get_settings()
        checkpointer = None if graph is not None else build_checkpointer(settings)
        try:
            app.state.graph = graph or build_graph(checkpointer=checkpointer)
            app.state.store = store or JsonStore(settings)
            yield
        finally:
            if isinstance(checkpointer, SqliteSaver):
                checkpointer.conn.close()

    app = FastAPI(title="Revenue Leakage Agent API", lifespan=lifespan)

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/threads")
    def create_thread() -> ThreadCreated:
        return ThreadCreated(thread_id=str(uuid4()))

    @app.post("/threads/{thread_id}/messages")
    def post_message(
        thread_id: str, body: MessageRequest, request: Request
    ) -> TurnResponse:
        payload: AgentState = {"messages": [HumanMessage(content=body.content)]}
        return _run_turn(request, thread_id, payload)

    @app.post("/threads/{thread_id}/resume")
    def resume(thread_id: str, body: ResumeRequest, request: Request) -> TurnResponse:
        snapshot = _known_snapshot(request, thread_id)
        if not snapshot.interrupts:
            raise HTTPException(status_code=409, detail="No pending approval.")
        return _run_turn(
            request, thread_id, Command(resume={"decision": body.decision})
        )

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str, request: Request) -> ThreadState:
        snapshot = _known_snapshot(request, thread_id)
        values = cast(AgentState, snapshot.values)
        return ThreadState(
            active_scope=values.get("active_scope"),
            findings=values.get("findings", []),
            pending_action=values.get("pending_action"),
            applied_actions=values.get("applied_actions", []),
            interrupt=_interrupt(snapshot),
        )

    return app


def _graph(request: Request) -> AgentGraph:
    return cast(AgentGraph, request.app.state.graph)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _known_snapshot(request: Request, thread_id: str) -> StateSnapshot:
    """State of a thread that has at least one checkpoint, else 404."""

    snapshot = _graph(request).get_state(_config(thread_id))
    if snapshot.created_at is None:
        raise HTTPException(status_code=404, detail="Unknown thread.")
    return snapshot


def _interrupt(snapshot: StateSnapshot) -> dict[str, Any] | None:
    if not snapshot.interrupts:
        return None
    value = snapshot.interrupts[0].value
    return cast(dict[str, Any], value) if isinstance(value, dict) else {"value": value}


def _messages(snapshot: StateSnapshot) -> list[AnyMessage]:
    return cast(AgentState, snapshot.values).get("messages", [])


def _run_turn(
    request: Request, thread_id: str, payload: AgentState | Command[Any]
) -> TurnResponse:
    graph = _graph(request)
    config = _config(thread_id)
    seen = {message.id for message in _messages(graph.get_state(config))}

    graph.invoke(  # pyright: ignore[reportUnknownMemberType]
        payload,
        config,
        context=AgentContext(store=cast(JsonStore, request.app.state.store)),
    )

    snapshot = graph.get_state(config)
    replies = [
        text
        for message in _messages(snapshot)
        if isinstance(message, AIMessage) and message.id not in seen
        if (text := extract_ai_text(message))
    ]
    return TurnResponse(
        thread_id=thread_id, replies=replies, interrupt=_interrupt(snapshot)
    )


app = create_app()
