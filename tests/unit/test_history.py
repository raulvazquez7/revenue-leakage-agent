from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult
from pydantic import Field

from fakes import ScriptedChatModel
from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.history import recent_history
from revenue_leakage_agent.nodes import agent as agent_module
from revenue_leakage_agent.nodes.agent import make_agent_node


def _tool_turn(n: int) -> list[AnyMessage]:
    call_id = f"call-{n}"
    return [
        HumanMessage(content=f"question {n}"),
        AIMessage(
            content="",
            tool_calls=[{"name": "load_plan", "args": {}, "id": call_id}],
        ),
        ToolMessage(content="{}", tool_call_id=call_id),
        AIMessage(content=f"answer {n}"),
    ]


def _assert_no_orphan_tool_messages(messages: list[AnyMessage]) -> None:
    seen_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            seen_call_ids.update(str(tc["id"]) for tc in message.tool_calls)
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in seen_call_ids


def test_recent_history_starts_on_human_and_keeps_tool_pairs() -> None:
    history = [*_tool_turn(1), *_tool_turn(2)]

    result = recent_history(history, 3)

    assert result
    assert isinstance(result[0], HumanMessage)
    _assert_no_orphan_tool_messages(result)


def test_recent_history_never_drops_the_current_turn() -> None:
    """A turn longer than the budget is kept whole rather than emptied."""

    history = [*_tool_turn(1), *_tool_turn(2)]

    result = recent_history(history, 3)

    assert result == history[4:]


def test_recent_history_keeps_whole_turn_when_it_fits() -> None:
    history = [*_tool_turn(1), *_tool_turn(2)]

    result = recent_history(history, 5)

    assert result == history[4:]


def test_recent_history_returns_everything_within_budget() -> None:
    history = _tool_turn(1)

    assert recent_history(history, 40) == history


class _RecordingModel(ScriptedChatModel):
    """Scripted model that records the messages it was invoked with."""

    seen: list[list[BaseMessage]] = Field(
        default_factory=lambda: list[list[BaseMessage]]()
    )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_agent_node_sends_bounded_history(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings(agent_history_messages=5)
    monkeypatch.setattr(agent_module, "get_settings", lambda: settings)
    model = _RecordingModel(messages=iter([AIMessage(content="ok")]))
    node = make_agent_node(model, [])
    history = [*_tool_turn(1), *_tool_turn(2), HumanMessage(content="next")]

    node({"messages": history})

    sent = [m for m in model.seen[0] if not isinstance(m, SystemMessage)]
    assert sent == history[-5:]
