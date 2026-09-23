from __future__ import annotations

import json
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
from agents.tools import get_tools

FALLBACK_AGENT_PROMPT = """You are a financial detective for revenue leakage.

Core rules:
- Use tools for plan data, invoice data, FX conversion, draft creation, and
  sandbox writes. Never invent amounts, currencies, dates, IDs, or write results.
- For a plan investigation, call load_plan first, then query_invoices with the
  plan_id. Use the plan_comparison findings returned by query_invoices as the
  source of truth for calculations.
- Explain evidence compactly: plan ID, billing period, expected amount, actual
  amount, invoice IDs, currency, and FX rate when relevant.
- Draft the right correction from tool evidence: propose_make_good_invoice for
  missing/underbilled revenue, propose_credit_memo for overbilling on an invoice,
  propose_plan_amendment when the plan itself is outdated. Drafting does not
  write to sandbox.
- Apply only when the user explicitly asks to apply/approve a pending draft.
  The apply tool will pause for human approval before writing. Use rollback only
  when the user asks to undo an applied action; it also pauses for approval.
- Reply in the user's language.
"""


def agent_node(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    openai_api_key = settings.openai_api_key
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the investigator agent.")
    prompt = load_prompt(
        prompts_dir=settings.prompts_dir,
        prompt_name=settings.agent_prompt_name,
        fallback=FALLBACK_AGENT_PROMPT,
    )

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
