"""Token streaming through the real graph, as the Streamlit chat consumes it."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from openai import OpenAI

from fakes import call, route, scripted
from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.interfaces.streaming import TurnStream, stream_events
from revenue_leakage_agent.llm import AgentModels, build_default_models
from revenue_leakage_agent.store import JsonStore

CONFIG: RunnableConfig = {"configurable": {"thread_id": "stream-test"}}


def test_conversation_reply_streams_token_chunks(seeded_store: JsonStore) -> None:
    models = AgentModels(
        router=scripted(route("conversation", "chit_chat")),
        agent=scripted(AIMessage(content="agent must not run")),
        conversational=scripted(AIMessage(content="Hello there, how can I help?")),
    )
    graph = build_graph(models, checkpointer=InMemorySaver())
    turn = TurnStream()

    pieces = list(
        turn.text(
            stream_events(
                graph,
                {"messages": [HumanMessage(content="hi")]},
                CONFIG,
                AgentContext(store=seeded_store),
            )
        )
    )

    # GenericFakeChatModel streams word by word; >1 piece proves real token
    # streaming reached the UI instead of one whole message at node end.
    assert len(pieces) > 1
    assert "".join(pieces) == "Hello there, how can I help?"
    assert turn.interrupt is None


def test_apply_turn_streams_text_then_captures_interrupt(
    seeded_store: JsonStore,
) -> None:
    models = AgentModels(
        router=scripted(route("investigation", "investigation")),
        agent=scripted(
            call(
                "propose_make_good_invoice",
                plan_id="SUB-2001",
                amount="10000",
                reason="Missing June invoice",
            ),
            call("apply"),
            AIMessage(content="Applied the make-good invoice."),
        ),
        conversational=scripted(AIMessage(content="unused")),
    )
    graph = build_graph(models, checkpointer=InMemorySaver())
    context = AgentContext(store=seeded_store)
    turn = TurnStream()

    text = "".join(
        turn.text(
            stream_events(
                graph,
                {"messages": [HumanMessage(content="draft and apply it")]},
                CONFIG,
                context,
            )
        )
    )

    assert text == ""
    assert turn.interrupt is not None
    assert turn.interrupt["type"] == "approval_required"
    assert turn.interrupt["action"]["plan_id"] == "SUB-2001"

    resumed = TurnStream()
    reply = "".join(
        resumed.text(
            stream_events(
                graph, Command(resume={"decision": "approve"}), CONFIG, context
            )
        )
    )

    assert reply == "Applied the make-good invoice."
    assert resumed.interrupt is None


def _openai_router_transport() -> httpx.MockTransport:
    """Fake OpenAI chat-completions endpoint answering one RouteDecision.

    Serves SSE when the client asks to stream and JSON otherwise, so the test
    exercises whichever path the router model actually takes.
    """

    content = json.dumps(
        {
            "route": "conversation",
            "intent": "chit_chat",
            "reason": "greeting",
            "resolved_question": None,
        }
    )
    base = {"id": "c1", "created": 1, "model": "gpt-test"}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            deltas: list[tuple[dict[str, str], str | None]] = [
                ({"role": "assistant", "content": ""}, None),
                ({"content": content}, None),
                ({}, "stop"),
            ]
            sse = "".join(
                "data: "
                + json.dumps(
                    {
                        **base,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                )
                + "\n\n"
                for delta, finish in deltas
            )
            return httpx.Response(
                200,
                content=(sse + "data: [DONE]\n\n").encode(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    return httpx.MockTransport(handler)


@pytest.mark.filterwarnings("error")
def test_openai_router_streams_turn_without_pydantic_serializer_warnings(
    seeded_store: JsonStore,
) -> None:
    """Streaming a json_schema structured output makes langchain-openai dump the
    openai SDK's ParsedChatCompletion, whose ``parsed`` field warns
    ("Pydantic serializer warnings"). The router is built non-streaming, so the
    UI/CLI need no warning filter."""

    router = build_default_models(AppSettings(openai_api_key="sk-test")).router
    assert isinstance(router, ChatOpenAI)
    root = OpenAI(
        api_key="sk-test",
        http_client=httpx.Client(transport=_openai_router_transport()),
    )
    router.root_client = root
    router.client = root.chat.completions
    models = AgentModels(
        router=router,
        agent=scripted(AIMessage(content="agent must not run")),
        conversational=scripted(AIMessage(content="Hello there")),
    )
    graph = build_graph(models, checkpointer=InMemorySaver())

    text = "".join(
        TurnStream().text(
            stream_events(
                graph,
                {"messages": [HumanMessage(content="hi")]},
                CONFIG,
                AgentContext(store=seeded_store),
            )
        )
    )

    assert text == "Hello there"
