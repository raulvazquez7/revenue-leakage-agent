"""Checkpointer construction for interactive interfaces."""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from revenue_leakage_agent.config import AppSettings


def build_checkpointer(settings: AppSettings) -> BaseCheckpointSaver[Any]:
    """In-memory checkpoints by default; SQLite when ``CHECKPOINT_DB`` is set.

    The SQLite connection is shared across threads (``SqliteSaver`` serialises
    access with its own lock). Callers that own the process lifetime should
    close ``saver.conn`` on shutdown.
    """

    path = settings.checkpoint_db
    if path is None:
        return InMemorySaver()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
