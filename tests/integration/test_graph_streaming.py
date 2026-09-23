"""Token streaming through the real graph, as the Streamlit chat consumes it."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fakes import call, route, scripted
from revenue_leakage_agent.context import AgentContext
from revenue_leakage_agent.graph import build_graph
from revenue_leakage_agent.interfaces.streaming import TurnStream, stream_events
from revenue_leakage_agent.llm import AgentModels
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
