from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.prompts import load_prompt
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tracing import get_langfuse_callbacks


def make_agent_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
) -> Callable[[AgentState], dict[str, object]]:
    """Build the investigator node with ``tools`` bound to ``model``."""

    prompt = load_prompt("agent")
    llm = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict[str, object]:
        config: RunnableConfig = {"callbacks": get_langfuse_callbacks(get_settings())}
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                SystemMessage(content=_state_context(state)),
                *state.get("messages", []),
            ],
            config=config,
        )
        return {"messages": [response]}

    return agent_node


def _state_context(state: AgentState) -> str:
    context: dict[str, Any] = {
        "active_scope": state.get("active_scope"),
        "findings": state.get("findings", []),
        "pending_action": state.get("pending_action"),
        "applied_actions": state.get("applied_actions", []),
        "last_error": state.get("last_error"),
    }
    return "Current graph state:\n" + json.dumps(context, default=str)
