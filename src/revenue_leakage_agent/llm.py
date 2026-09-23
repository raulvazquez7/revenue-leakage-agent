"""Chat model construction, kept apart from graph wiring so models are injectable."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from revenue_leakage_agent.config import (
    AppSettings,
    build_chat_openai_kwargs,
    get_settings,
)


@dataclass(frozen=True)
class AgentModels:
    """The chat models used by each model-backed graph node."""

    router: BaseChatModel
    agent: BaseChatModel
    conversational: BaseChatModel


def build_default_models(settings: AppSettings | None = None) -> AgentModels:
    """Build the default OpenAI models from settings.

    Raises ``RuntimeError`` when ``OPENAI_API_KEY`` is not configured.
    """

    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to build the default models. "
            "Set it in .env or pass explicit models to build_graph()."
        )
    api_key = SecretStr(settings.openai_api_key)

    router = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.router_model,
            api_key=api_key,
            reasoning_effort=settings.router_reasoning_effort,
            reasoning_summary=settings.router_reasoning_summary,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
    agent = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.agent_model,
            api_key=api_key,
            reasoning_effort=settings.agent_reasoning_effort,
            reasoning_summary=settings.agent_reasoning_summary,
            timeout_seconds=settings.agent_timeout_seconds,
        )
    )
    conversational = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.conversational_model,
            api_key=api_key,
            reasoning_effort=settings.conversational_reasoning_effort,
            reasoning_summary=settings.conversational_reasoning_summary,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
    return AgentModels(router=router, agent=agent, conversational=conversational)
