from __future__ import annotations

from pathlib import Path

import pytest

from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.store import JsonStore


@pytest.fixture
def store(tmp_path: Path) -> JsonStore:
    """A JsonStore rooted in an isolated temporary data/sandbox tree."""

    settings = AppSettings(data_dir=tmp_path / "data", sandbox_dir=tmp_path / "sandbox")
    return JsonStore(settings)
