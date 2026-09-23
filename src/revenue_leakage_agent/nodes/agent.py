from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from revenue_leakage_agent.config import build_chat_openai_kwargs, get_settings
from revenue_leakage_agent.prompts import load_prompt
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tools import get_tools
from revenue_leakage_agent.tracing import get_langfuse_callbacks


def agent_node(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    openai_api_key = settings.openai_api_key
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the investigator agent.")
    prompt = load_prompt("agent")

    base_llm: Any = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.agent_model,
            api_key=SecretStr(openai_api_key),
            reasoning_effort=settings.agent_reasoning_effort,
            reasoning_summary=settings.agent_reasoning_summary,
            timeout_seconds=settings.agent_timeout_seconds,
        )
    )
    llm: Any = base_llm.bind_tools(get_tools())
    config: RunnableConfig = {"callbacks": get_langfuse_callbacks(settings)}

    response = cast(
        AIMessage,
        llm.invoke(
            [
                SystemMessage(content=prompt),
                SystemMessage(content=_state_context(state)),
                *state.get("messages", []),
            ],
            config=config,
        ),
    )
    return {"messages": [response]}


def _state_context(state: AgentState) -> str:
    context: dict[str, Any] = {
        "active_scope": state.get("active_scope"),
        "findings": state.get("findings", []),
        "pending_action": state.get("pending_action"),
        "applied_actions": state.get("applied_actions", []),
        "last_error": state.get("last_error"),
    }
    return "Current graph state:\n" + json.dumps(context, default=str)
