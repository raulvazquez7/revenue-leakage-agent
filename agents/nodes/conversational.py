from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from settings import get_settings
from settings.config import build_chat_openai_kwargs
from settings.tracing import get_langfuse_callbacks

from agents.prompts import load_prompt
from agents.state import AgentState

FALLBACK_CONVERSATIONAL_PROMPT = """You are the conversational surface for a
revenue leakage agent.

Handle greetings, capability questions, and out-of-scope turns. Keep answers
short, useful, and in the user's language. Do not perform financial analysis or
invent billing evidence; when analysis is needed, tell the user to provide a
plan ID so the investigation route can use tools.
"""


def conversational_node(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    openai_api_key = settings.openai_api_key
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the conversational node.")

    prompt = load_prompt(
        prompts_dir=settings.prompts_dir,
        prompt_name=settings.conversational_prompt_name,
        fallback=FALLBACK_CONVERSATIONAL_PROMPT,
    )
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
