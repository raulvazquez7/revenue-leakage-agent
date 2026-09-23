from __future__ import annotations

from pathlib import Path

import pytest

from revenue_leakage_agent.config import AppSettings, get_settings


def test_tests_ignore_dotenv_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard for the autouse fixture in ``tests/conftest.py``: a developer's
    ``.env`` must not change what the test suite runs against."""

    (tmp_path / ".env").write_text(
        "ROUTER_MODEL=from-dotenv\nOPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    settings = AppSettings()

    assert settings.router_model == "gpt-5.4-mini"
    assert settings.openai_api_key == ""


def test_ambient_settings_are_code_defaults() -> None:
    settings = get_settings()

    assert settings.openai_api_key == ""
    assert settings.router_reasoning_effort == "none"
    assert settings.checkpoint_db is None
