import logging
from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, merge_configs

from revenue_leakage_agent.config import AppSettings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _ensure_langfuse_client(public_key: str, secret_key: str, base_url: str) -> bool:
    """Construct the Langfuse client for ``(public_key, secret_key, base_url)`` once.

    The SDK's ``CallbackHandler`` reads its client from a process-wide
    registry keyed by ``public_key`` (see ``langfuse.get_client``), not from
    ``AppSettings``. Explicitly constructing the client here — rather than
    relying on LANGFUSE_* process env vars, which this app never sets
    (``AppSettings`` parses ``.env`` into itself, not into ``os.environ``) —
    is what actually registers it. ``lru_cache`` keyed on the three
    credential strings (rather than on ``settings``, which also carries
    unrelated fields) makes this a one-time cost per distinct credential set:
    it spins up an HTTP client and a background flush thread, so repeating it
    on every node call would leak resources. Returns ``True``/raises so
    ``lru_cache`` only remembers a *successful* construction; a call that
    raises is not cached and will be retried by the next node call.
    """

    from langfuse import Langfuse

    Langfuse(public_key=public_key, secret_key=secret_key, host=base_url)
    return True


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
        logger.warning(
            "Langfuse keys are configured but the langfuse.langchain "
            "CallbackHandler could not be imported; tracing is disabled. "
            "Is the `langchain` package installed?",
            exc_info=True,
        )
        return []

    handler_cls: Any = CallbackHandler
    try:
        _ensure_langfuse_client(
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_base_url,
        )
        return [handler_cls()]
    except Exception:
        logger.warning(
            "Langfuse keys are configured but the client/CallbackHandler "
            "could not be constructed; tracing is disabled.",
            exc_info=True,
        )
        return []


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
