from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.domain.models import InvestigationScope, RouteDecision
from revenue_leakage_agent.history import recent_history
from revenue_leakage_agent.prompts import load_prompt
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tracing import get_langfuse_callbacks


def make_router_node(model: BaseChatModel) -> Callable[[AgentState], dict[str, object]]:
    """Build the router node that classifies a turn with structured output."""

    prompt = load_prompt("router")
    llm = model.with_structured_output(
        RouteDecision,
        method="json_schema",
        strict=True,
    )

    def router_node(state: AgentState) -> dict[str, object]:
        settings = get_settings()
        config: RunnableConfig = {"callbacks": get_langfuse_callbacks(settings)}
        raw_decision = llm.invoke(
            [
                SystemMessage(content=prompt),
                SystemMessage(content=f"Active scope: {state.get('active_scope')}"),
                *recent_history(
                    state.get("messages", []), settings.router_history_messages
                ),
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

    return router_node


def route_from_decision(state: AgentState) -> Literal["conversation", "investigation"]:
    decision = state.get("route_decision") or {}
    route = decision.get("route")
    return "investigation" if route == "investigation" else "conversation"
