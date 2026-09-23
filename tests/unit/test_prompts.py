from __future__ import annotations

from pathlib import Path

import pytest

from revenue_leakage_agent import prompts
from revenue_leakage_agent.prompts import load_prompt


@pytest.mark.parametrize("name", ["router", "agent", "conversational"])
def test_load_prompt_returns_packaged_markdown(name: str) -> None:
    assert load_prompt(name).strip()


def test_load_prompt_raises_for_unknown_prompt() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nope")


def test_load_prompt_raises_for_empty_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "blank.md").write_text("  \n", encoding="utf-8")

    def fake_files(_package: str) -> Path:
        return tmp_path

    monkeypatch.setattr(prompts, "files", fake_files)

    with pytest.raises(FileNotFoundError):
        load_prompt("blank")
