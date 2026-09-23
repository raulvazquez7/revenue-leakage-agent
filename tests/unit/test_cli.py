from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from revenue_leakage_agent.config import AppSettings
from revenue_leakage_agent.interfaces import cli
from revenue_leakage_agent.persistence import build_checkpointer

CHUNK = {"agent": {"messages": [AIMessage(content="hello")]}}


def test_run_stream_hides_chunk_dump_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._run_stream([CHUNK])  # pyright: ignore[reportPrivateUsage]

    out = capsys.readouterr().out
    assert "[trace] chunk=" not in out
    assert "Agent: hello" in out


def test_run_stream_prints_chunk_dump_when_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._run_stream([CHUNK], verbose=True)  # pyright: ignore[reportPrivateUsage]

    assert "[trace] chunk=" in capsys.readouterr().out


def test_help_documents_verbose_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    assert "--verbose" in capsys.readouterr().out


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_main_exits_cleanly_and_closes_sqlite(
    error: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(checkpoint_db=tmp_path / "checkpoints.sqlite")
    opened: list[BaseCheckpointSaver[Any]] = []

    def _build_checkpointer(settings: AppSettings) -> BaseCheckpointSaver[Any]:
        saver = build_checkpointer(settings)
        opened.append(saver)
        return saver

    def _raise(prompt: str = "") -> str:
        raise error

    def _build_graph(**_: Any) -> object:
        return object()

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "build_checkpointer", _build_checkpointer)
    monkeypatch.setattr(cli, "build_graph", _build_graph)
    monkeypatch.setattr("builtins.input", _raise)

    cli.main([])

    saver = opened[0]
    assert isinstance(saver, SqliteSaver)
    with pytest.raises(sqlite3.ProgrammingError):
        saver.conn.execute("select 1")
