from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.prompts import load_prompt
from revenue_leakage_agent.state import AgentState
from revenue_leakage_agent.tracing import get_langfuse_callbacks


def make_conversational_node(
    model: BaseChatModel,
) -> Callable[[AgentState], dict[str, object]]:
    """Build the node that answers greetings and out-of-scope turns."""

    prompt = load_prompt("conversational")

    def conversational_node(state: AgentState) -> dict[str, object]:
        config: RunnableConfig = {"callbacks": get_langfuse_callbacks(get_settings())}
        response = model.invoke(
            [
                SystemMessage(content=prompt),
                SystemMessage(content=f"Route decision: {state.get('route_decision')}"),
                *state.get("messages", [])[-8:],
            ],
            config=config,
        )
        return {"messages": [response]}

    return conversational_node
