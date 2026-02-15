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

    # ------------------------------------------------------------------
    # Burn rate analytics
    # ------------------------------------------------------------------
    def get_burn_rate(self, days: int = 30) -> Dict[str, Any]:
        """Get token usage over time for burn rate calculations.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dict with daily/weekly/monthly breakdowns and projections
        """
        conn = self._get_conn()
        
        # Get daily token usage for the period
        rows = conn.execute("""
            SELECT 
                DATE(created_at) as day,
                SUM(tokens_used) as tokens,
                COUNT(*) as queries
            FROM events 
            WHERE event_type = 'context' 
              AND created_at >= DATE('now', ?)
            GROUP BY day
            ORDER BY day DESC
        """, (f"-{days} days",)).fetchall()
        
        daily_data = [
            {"day": r["day"], "tokens": r["tokens"], "queries": r["queries"]}
            for r in rows
        ]
        
        # Calculate totals and averages
        total_tokens = sum(r["tokens"] for r in rows)
        total_queries = sum(r["queries"] for r in rows)
        
        # Daily average (non-zero days only)
        active_days = len(rows) if rows else 1
        daily_avg = total_tokens / active_days if active_days > 0 else 0
        
        # Weekly projection
        weekly_tokens = daily_avg * 7
        monthly_tokens = daily_avg * 30
        
        # Cost projections (Claude pricing)
        PRICE_PER_1M = 15.0
        daily_cost = (daily_avg / 1_000_000) * PRICE_PER_1M
        weekly_cost = (weekly_tokens / 1_000_000) * PRICE_PER_1M
        monthly_cost = (monthly_tokens / 1_000_000) * PRICE_PER_1M
        
        return {
            "period_days": days,
            "total_tokens": total_tokens,
            "total_queries": total_queries,
            "active_days": active_days,
            "daily_avg_tokens": round(daily_avg),
            "daily_data": daily_data[:14],  # Last 14 days
            "projections": {
                "daily_tokens": round(daily_avg),
                "daily_cost": round(daily_cost, 2),
                "weekly_tokens": round(weekly_tokens),
                "weekly_cost": round(weekly_cost, 2),
                "monthly_tokens": round(monthly_tokens),
                "monthly_cost": round(monthly_cost, 2),
            },
        }
    
    def get_project_breakdown(self) -> list:
        """Get per-project token breakdown (for multi-project setups).
        
        Returns:
            List of dicts with project stats
        """
        # For single-project setups, return just this project
        conn = self._get_conn()
        
        total = conn.execute(
            "SELECT SUM(tokens_used) FROM events WHERE event_type = 'context'"
        ).fetchone()[0] or 0
        
        return [{
            "project": str(self.root.name),
            "tokens": total,
            "queries": conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'context'"
            ).fetchone()[0],
        }]
