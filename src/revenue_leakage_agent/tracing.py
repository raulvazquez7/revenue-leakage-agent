from typing import Any

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
