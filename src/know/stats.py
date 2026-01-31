"""Usage statistics tracker for know-cli.

Tracks every query, search, and memory operation in a project-local
SQLite database (`.know/stats.db`).  Powers the `know stats` command.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


class StatsTracker:
    """Project-local usage statistics backed by SQLite."""

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.db_path = self.root / ".know" / "stats.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                query TEXT DEFAULT '',
                budget INTEGER DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Record events
    # ------------------------------------------------------------------
    def record_context(
        self,
        query: str,
        budget: int,
        tokens_used: int,
        duration_ms: int,
    ):
        """Record a `know context` call."""
        self._insert("context", query=query, budget=budget,
                      tokens_used=tokens_used, duration_ms=duration_ms)

    def record_search(self, query: str, results_count: int, duration_ms: int):
        """Record a `know search` call."""
        self._insert("search", query=query, duration_ms=duration_ms,
                      metadata=f'{{"results": {results_count}}}')

    def record_remember(self, text: str, source: str = "manual"):
        """Record a `know remember` call."""
        self._insert("remember", query=text,
                      metadata=f'{{"source": "{source}"}}')

    def record_recall(self, query: str, results_count: int, duration_ms: int):
        """Record a `know recall` call."""
        self._insert("recall", query=query, duration_ms=duration_ms,
                      metadata=f'{{"results": {results_count}}}')

    def _insert(self, event_type: str, **kwargs):
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO events (event_type, query, budget, tokens_used,
                                   duration_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_type,
                kwargs.get("query", ""),
                kwargs.get("budget", 0),
                kwargs.get("tokens_used", 0),
                kwargs.get("duration_ms", 0),
                kwargs.get("metadata", "{}"),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Aggregated stats
    # ------------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated statistics for this project."""
        conn = self._get_conn()

        # Context calls
        ctx = conn.execute("""
            SELECT COUNT(*) as cnt,
                   COALESCE(AVG(tokens_used), 0) as avg_tokens,
                   COALESCE(AVG(budget), 0) as avg_budget,
                   COALESCE(AVG(duration_ms), 0) as avg_ms
            FROM events WHERE event_type = 'context'
        """).fetchone()

        # Search calls
        srch = conn.execute("""
            SELECT COUNT(*) as cnt,
                   COALESCE(AVG(duration_ms), 0) as avg_ms
            FROM events WHERE event_type = 'search'
        """).fetchone()

        # Memory operations
        rem = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'remember'"
        ).fetchone()
        rec = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'recall'"
        ).fetchone()

        avg_budget = ctx["avg_budget"] if ctx["avg_budget"] else 1
        budget_util = (ctx["avg_tokens"] / avg_budget * 100) if avg_budget > 0 else 0

        return {
            "context_queries": ctx["cnt"],
            "context_avg_tokens": int(ctx["avg_tokens"]),
            "context_avg_budget": int(ctx["avg_budget"]),
            "context_budget_util": round(budget_util, 1),
            "context_avg_ms": int(ctx["avg_ms"]),
            "search_queries": srch["cnt"],
            "search_avg_ms": int(srch["avg_ms"]),
            "remember_count": rem[0],
            "recall_count": rec[0],
        }
