"""Run-scoped dependencies passed to the graph via ``context=``."""

from __future__ import annotations

from dataclasses import dataclass

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.store import JsonStore


@dataclass(frozen=True)
class AgentContext:
    """Runtime context injected into tools through ``ToolRuntime``."""

    store: JsonStore


def resolve_store(context: AgentContext | None) -> JsonStore:
    """Store from runtime context, or a default JsonStore(get_settings())."""

    if context is not None:
        return context.store
    return JsonStore(get_settings())
