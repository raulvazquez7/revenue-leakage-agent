from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, merge_configs

from revenue_leakage_agent.config import AppSettings


def get_langfuse_callbacks(settings: AppSettings) -> list[Any]:
    """Return Langfuse callbacks when configured, otherwise keep tracing optional."""

    if not (
        settings.langfuse_secret_key
        and settings.langfuse_public_key
        and settings.langfuse_base_url
    ):
        return []

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return []

    handler_cls: Any = CallbackHandler
    return [handler_cls()]


def model_call_config(settings: AppSettings) -> RunnableConfig:
    """Config for a model call made inside a graph node.

    Starts from the node's inherited run config and *adds* the Langfuse
    handlers. Passing ``{"callbacks": [...]}`` alone would replace the graph's
    callback manager, which detaches the call from the run: LangGraph's
    ``messages`` stream mode would then see no tokens.
    """

    return merge_configs(
        ensure_config(), {"callbacks": get_langfuse_callbacks(settings)}
    )
