from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from revenue_leakage_agent.config import build_chat_openai_kwargs, get_settings
from revenue_leakage_agent.prompts import load_prompt
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tracing import get_langfuse_callbacks


def conversational_node(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    openai_api_key = settings.openai_api_key
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the conversational node.")

    prompt = load_prompt("conversational")
    base_llm: Any = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.conversational_model,
            api_key=SecretStr(openai_api_key),
            reasoning_effort=settings.conversational_reasoning_effort,
            reasoning_summary=settings.conversational_reasoning_summary,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
    config: RunnableConfig = {"callbacks": get_langfuse_callbacks(settings)}
    response = cast(
        AIMessage,
        base_llm.invoke(
            [
                SystemMessage(content=prompt),
                SystemMessage(content=f"Route decision: {state.get('route_decision')}"),
                *state.get("messages", [])[-8:],
            ],
            config=config,
        ),
    )

    return {"messages": [response]}
