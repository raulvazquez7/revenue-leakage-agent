"""Run-scoped dependencies passed to the graph via ``context=``."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import InstanceOf
from pydantic.json_schema import SkipJsonSchema

from revenue_leakage_agent.config import get_settings
from revenue_leakage_agent.store import JsonStore


@dataclass(frozen=True)
class AgentContext:
    """Runtime context injected into tools through ``ToolRuntime``."""

    # Optional so platform callers (LangGraph Studio / server), which build the
    # context from a JSON dict via ``AgentContext(**context)``, can pass ``{}``;
    # tools then fall back to the default store. SkipJsonSchema + InstanceOf keep
    # the non-JSON store out of the generated schema but accept it in Python.
    store: SkipJsonSchema[InstanceOf[JsonStore] | None] = None


def resolve_store(context: AgentContext | None) -> JsonStore:
    """Store from runtime context, or a default JsonStore(get_settings())."""

    if context is not None and context.store is not None:
        return context.store
    return JsonStore(get_settings())
