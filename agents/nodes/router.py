from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from settings import get_settings
from settings.config import build_chat_openai_kwargs
from settings.tracing import get_langfuse_callbacks

from agents.prompts import load_prompt
from agents.schemas import InvestigationScope, RouteDecision
from agents.state import AgentState

FALLBACK_ROUTER_PROMPT = """You route a revenue leakage assistant turn.

Routes:
- conversation: greetings, capability questions, out-of-scope requests, or answers
  that do not need billing tools.
- investigation: questions or follow-ups about billing plans, invoices, revenue
  leakage, corrective actions, drafts, approval, or sandbox mutations.

Use the conversation history and active scope to resolve follow-ups like "what
about that month?" into a standalone resolved_question when possible. Do not
choose financial tools here; only choose the route.
"""


def router_node(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    openai_api_key = _require_openai_key(settings.openai_api_key)
    prompt = load_prompt(
        prompts_dir=settings.prompts_dir,
        prompt_name=settings.router_prompt_name,
        fallback=FALLBACK_ROUTER_PROMPT,
    )

    base_llm: Any = ChatOpenAI(
        **build_chat_openai_kwargs(
            model=settings.router_model,
            api_key=SecretStr(openai_api_key),
            reasoning_effort=settings.router_reasoning_effort,
            reasoning_summary=settings.router_reasoning_summary,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    )
    llm: Any = base_llm.with_structured_output(
        RouteDecision,
        method="json_schema",
        strict=True,
    )
    config: RunnableConfig = {"callbacks": get_langfuse_callbacks(settings)}
    raw_decision = llm.invoke(
        [
            SystemMessage(content=prompt),
            SystemMessage(content=f"Active scope: {state.get('active_scope')}"),
            *state.get("messages", [])[-8:],
        ],
        config=config,
    )
    decision = (
        raw_decision
        if isinstance(raw_decision, RouteDecision)
        else RouteDecision.model_validate(cast(dict[str, Any], raw_decision))
    )

    update: dict[str, object] = {"route_decision": decision.model_dump(mode="json")}
    if decision.resolved_question:
        current_scope = state.get("active_scope") or {}
        scope = InvestigationScope.model_validate(
            {
                **current_scope,
                "resolved_question": decision.resolved_question,
            }
        )
        update["active_scope"] = scope.model_dump(mode="json")
    return update


def route_from_decision(state: AgentState) -> str:
    decision = state.get("route_decision") or {}
    route = decision.get("route")
    return "investigation" if route == "investigation" else "conversation"


def _require_openai_key(openai_api_key: str | None) -> str:
    if not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run the model-backed router. "
            "Set it in .env before starting the demo."
        )
    return openai_api_key
