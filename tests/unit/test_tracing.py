"""Langfuse callback wiring: optional, and quiet unless something is wrong."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator

import pytest

from revenue_leakage_agent.config import AppSettings, get_settings
from revenue_leakage_agent.tracing import (
    _ensure_langfuse_client,  # pyright: ignore[reportPrivateUsage]
    get_langfuse_callbacks,
)

LOGGER_NAME = "revenue_leakage_agent.tracing"


@pytest.fixture(autouse=True)
def _clear_langfuse_client_cache() -> Iterator[None]:
    """Isolate ``_ensure_langfuse_client``'s ``lru_cache`` between tests.

    Several tests here reuse the same dummy ``("pk-test", "sk-test", ...)``
    credential triple. Without clearing the cache, whichever test runs first
    would silently decide whether the *real* (stubbed or not) client
    construction happens for every later test using that same triple.
    """

    _ensure_langfuse_client.cache_clear()
    yield
    _ensure_langfuse_client.cache_clear()


def test_ambient_test_session_has_no_langfuse_keys() -> None:
    """Guard for the autouse fixture in ``tests/conftest.py``.

    Graph/integration tests call the real, process-wide ``get_settings()``
    (not an injected settings object) from inside nodes, so if a developer's
    local ``.env`` has real LANGFUSE_* keys, every ScriptedChatModel test
    would otherwise export a real trace the moment tracing actually works.
    This asserts the autouse fixture is doing its job for this test's own
    settings snapshot, regardless of what is in ``.env`` on this machine.
    """

    settings = get_settings()

    assert not settings.langfuse_secret_key
    assert not settings.langfuse_public_key
    assert not settings.langfuse_base_url
    assert get_langfuse_callbacks(settings) == []


def _settings(**overrides: object) -> AppSettings:
    return AppSettings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def _configured_settings() -> AppSettings:
    return _settings(
        langfuse_secret_key="sk-test",
        langfuse_public_key="pk-test",
        langfuse_base_url="http://localhost:9",
    )


def test_returns_empty_list_and_logs_nothing_when_keys_unset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    callbacks = get_langfuse_callbacks(_settings())

    assert callbacks == []
    assert caplog.records == []


def test_returns_one_callback_handler_when_keys_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    settings = _configured_settings()

    callbacks = get_langfuse_callbacks(settings)

    try:
        from langfuse.langchain import CallbackHandler

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], CallbackHandler)
        # Scope to our own logger: the Langfuse SDK itself may log at
        # WARNING (e.g. rejecting the dummy keys) on its own logger, which
        # is not this function's concern.
        our_records = [r for r in caplog.records if r.name == LOGGER_NAME]
        assert our_records == []
    finally:
        # Langfuse's client starts a background flush thread; shut it down so
        # the test process exits cleanly and teardown emits no warnings.
        from langfuse import get_client

        get_client().shutdown()


def test_logs_warning_and_returns_empty_list_when_handler_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "langfuse.langchain", None)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    callbacks = get_langfuse_callbacks(_configured_settings())

    assert callbacks == []
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == LOGGER_NAME
    assert record.levelno == logging.WARNING


def test_client_construction_is_cached_per_credential_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls with the same settings must not construct the client twice.

    No network involved: ``langfuse.Langfuse`` itself is replaced with a
    counting stub, so this exercises only ``_ensure_langfuse_client``'s
    ``lru_cache`` behaviour.
    """

    calls: list[tuple[str, str, str]] = []

    class _StubLangfuse:
        def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
            calls.append((public_key, secret_key, host))

    monkeypatch.setattr("langfuse.Langfuse", _StubLangfuse)
    settings = _configured_settings()

    callbacks_1 = get_langfuse_callbacks(settings)
    callbacks_2 = get_langfuse_callbacks(settings)

    assert len(calls) == 1
    assert calls[0] == ("pk-test", "sk-test", "http://localhost:9")
    # A fresh, stateless CallbackHandler wrapper per call is fine; only the
    # underlying client construction needs to be cached.
    assert len(callbacks_1) == 1
    assert len(callbacks_2) == 1
