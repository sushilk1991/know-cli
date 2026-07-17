"""Usage statistics tracker for know-cli.

Tracks every query, search, and memory operation in a project-local
SQLite database (`.know/stats.db`).  Powers the `know stats` command.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, Optional

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
        conn = getattr(self, "_conn", None)
        if conn:
            conn.close()
            self._conn = None

    def __del__(self):
        """Best-effort fallback for short-lived callers that omit ``close``."""
        try:
            self.close()
        except Exception:
            pass

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
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Record a `know context` call."""
        payload = dict(metadata or {})
        self._insert(
            "context",
            query=query,
            budget=budget,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            metadata=json.dumps(payload, ensure_ascii=True),
        )

    def record_search(self, query: str, results_count: int, duration_ms: int):
        """Record a `know search` call."""
        self._insert(
            "search",
            query=query,
            duration_ms=duration_ms,
            metadata=json.dumps({"results": int(results_count)}, ensure_ascii=True),
        )

    def record_map(
        self,
        query: str,
        results_count: int,
        duration_ms: int,
        tokens_used: int = 0,
        session_id: str = "",
    ):
        """Record a `know map` retrieval call."""
        self._insert(
            "map",
            query=query,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            metadata=json.dumps(
                {
                    "results": int(results_count),
                    "session_id": session_id or "",
                },
                ensure_ascii=True,
            ),
        )

    def record_deep(
        self,
        name: str,
        duration_ms: int,
        tokens_used: int,
        *,
        call_graph_available: Optional[bool] = None,
        callers_count: int = 0,
        callees_count: int = 0,
        session_id: str = "",
        error: str = "",
    ):
        """Record a `know deep` retrieval call."""
        self._insert(
            "deep",
            query=name,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            metadata=json.dumps(
                {
                    "call_graph_available": call_graph_available,
                    "callers_count": int(callers_count),
                    "callees_count": int(callees_count),
                    "session_id": session_id or "",
                    "error": error or "",
                },
                ensure_ascii=True,
            ),
        )

    def record_workflow(
        self,
        query: str,
        duration_ms: int,
        tokens_used: int,
        *,
        mode: str = "implement",
        degraded_by_latency: bool = False,
        call_graph_available: Optional[bool] = None,
        callers_count: int = 0,
        callees_count: int = 0,
        session_id: str = "",
    ):
        """Record a `know workflow` retrieval call."""
        self._insert(
            "workflow",
            query=query,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            metadata=json.dumps(
                {
                    "mode": mode,
                    "degraded_by_latency": bool(degraded_by_latency),
                    "call_graph_available": call_graph_available,
                    "callers_count": int(callers_count),
                    "callees_count": int(callees_count),
                    "session_id": session_id or "",
                },
                ensure_ascii=True,
            ),
        )

    def record_remember(self, text: str, source: str = "manual"):
        """Record a `know remember` call."""
        self._insert(
            "remember",
            query=text,
            metadata=json.dumps({"source": source}, ensure_ascii=True),
        )

    def record_recall(self, query: str, results_count: int, duration_ms: int):
        """Record a `know recall` call."""
        self._insert(
            "recall",
            query=query,
            duration_ms=duration_ms,
            metadata=json.dumps({"results": int(results_count)}, ensure_ascii=True),
        )

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

        avg_tokens = self._safe_nonnegative_int(ctx["avg_tokens"])
        avg_budget = self._safe_nonnegative_int(ctx["avg_budget"])
        avg_context_ms = self._safe_nonnegative_int(ctx["avg_ms"])
        avg_search_ms = self._safe_nonnegative_int(srch["avg_ms"])
        budget_util = (avg_tokens / avg_budget * 100) if avg_budget > 0 else 0.0

        return {
            "context_queries": ctx["cnt"],
            "context_avg_tokens": avg_tokens,
            "context_avg_budget": avg_budget,
            "context_budget_util": round(budget_util, 1),
            "context_avg_ms": avg_context_ms,
            "search_queries": srch["cnt"],
            "search_avg_ms": avg_search_ms,
            "remember_count": rem[0],
            "recall_count": rec[0],
        }

    @staticmethod
    def _percentile(values: list[int], p: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        if len(ordered) == 1:
            return int(ordered[0])

        rank = (len(ordered) - 1) * p
        low = int(rank)
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return int(round(ordered[low] * (1.0 - frac) + ordered[high] * frac))

    @staticmethod
    def _safe_nonnegative_int(raw: Any, default: int = 0) -> int:
        """Coerce persisted telemetry without trusting its SQLite affinity.

        SQLite permits text in INTEGER columns, and metadata JSON is user-
        writable.  A corrupt row should contribute zero, not make every stats
        query fail or inject negative/astronomically large measurements.
        """
        if isinstance(raw, bool) or raw is None:
            return default
        try:
            if isinstance(raw, float) and not math.isfinite(raw):
                return default
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            return default
        if value < 0 or value > 2**63 - 1:
            return default
        return value

    @staticmethod
    def _parse_metadata(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def get_retrieval_summary(self, days: int = 30) -> dict[str, Any]:
        """Return retrieval-focused analytics for map/context/deep/workflow."""
        conn = self._get_conn()
        event_types = ("map", "context", "deep", "workflow")

        rows = conn.execute(
            """
            SELECT event_type, tokens_used, duration_ms, metadata
            FROM events
            WHERE event_type IN ('map', 'context', 'deep', 'workflow')
              AND created_at >= DATETIME('now', ?)
            ORDER BY created_at DESC
            """,
            (f"-{int(days)} days",),
        ).fetchall()

        durations: dict[str, list[int]] = {k: [] for k in event_types}
        tokens: dict[str, list[int]] = {k: [] for k in event_types}
        counts: dict[str, int] = {k: 0 for k in event_types}

        workflow_total = 0
        workflow_degraded = 0
        workflow_call_graph_true = 0
        workflow_non_empty_edges = 0

        for row in rows:
            event_type = str(row["event_type"] or "")
            if event_type not in counts:
                continue

            duration_ms = self._safe_nonnegative_int(row["duration_ms"])
            token_count = self._safe_nonnegative_int(row["tokens_used"])
            counts[event_type] += 1
            durations[event_type].append(duration_ms)
            tokens[event_type].append(token_count)

            if event_type == "workflow":
                workflow_total += 1
                metadata = self._parse_metadata(row["metadata"])
                if bool(metadata.get("degraded_by_latency")):
                    workflow_degraded += 1
                if metadata.get("call_graph_available") is True:
                    workflow_call_graph_true += 1
                callers_count = self._safe_nonnegative_int(metadata.get("callers_count"))
                callees_count = self._safe_nonnegative_int(metadata.get("callees_count"))
                if callers_count + callees_count > 0:
                    workflow_non_empty_edges += 1

        command_summary: dict[str, dict[str, int]] = {}
        for event_type in event_types:
            cnt = counts[event_type]
            avg_tokens = int(round(sum(tokens[event_type]) / cnt)) if cnt else 0
            command_summary[event_type] = {
                "count": cnt,
                "p50_ms": self._percentile(durations[event_type], 0.50),
                "p95_ms": self._percentile(durations[event_type], 0.95),
                "avg_tokens": avg_tokens,
            }

        def _pct(numerator: int, denominator: int) -> float:
            if denominator <= 0:
                return 0.0
            return round((numerator / denominator) * 100.0, 1)

        return {
            "window_days": int(days),
            "commands": command_summary,
            "workflow_quality": {
                "degraded_by_latency_rate_pct": _pct(workflow_degraded, workflow_total),
                "call_graph_available_rate_pct": _pct(workflow_call_graph_true, workflow_total),
                "non_empty_edge_rate_pct": _pct(workflow_non_empty_edges, workflow_total),
            },
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
            {
                "day": r["day"],
                "tokens": self._safe_nonnegative_int(r["tokens"]),
                "queries": self._safe_nonnegative_int(r["queries"]),
            }
            for r in rows
        ]
        
        # Calculate totals and averages
        total_tokens = sum(row["tokens"] for row in daily_data)
        total_queries = sum(row["queries"] for row in daily_data)
        
        # Daily average (non-zero days only)
        active_days = len(rows)
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
