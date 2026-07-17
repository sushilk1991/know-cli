"""Adversarial regressions for stats and public memory-service boundaries."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from click.testing import CliRunner

from know.cli.knowledge import (
    decide,
    recall as recall_command,
    remember as remember_command,
)
from know.cli.stats import stats as stats_command
from know.config import Config
from know.stats import StatsTracker


@pytest.fixture
def config(tmp_path):
    cfg = Config.create_default(tmp_path)
    cfg.root = tmp_path
    return cfg


def test_retrieval_summary_treats_corrupt_numeric_fields_as_zero(config):
    """One malformed telemetry row must not break the whole stats command."""
    with StatsTracker(config) as tracker:
        conn = tracker._get_conn()
        conn.execute(
            """
            INSERT INTO events (event_type, tokens_used, duration_ms, metadata)
            VALUES ('workflow', ?, ?, ?)
            """,
            (
                "not-a-token-count",
                "NaN",
                json.dumps(
                    {
                        "callers_count": "many",
                        "callees_count": float("inf"),
                        "degraded_by_latency": True,
                    }
                ),
            ),
        )
        conn.commit()

        summary = tracker.get_retrieval_summary()

    assert summary["commands"]["workflow"] == {
        "count": 1,
        "p50_ms": 0,
        "p95_ms": 0,
        "avg_tokens": 0,
    }
    assert summary["workflow_quality"]["degraded_by_latency_rate_pct"] == 100.0
    assert summary["workflow_quality"]["non_empty_edge_rate_pct"] == 0.0


def test_context_summary_and_cli_survive_nonfinite_persisted_aggregates(config):
    with StatsTracker(config) as tracker:
        conn = tracker._get_conn()
        conn.execute(
            """INSERT INTO events
               (event_type, budget, tokens_used, duration_ms, metadata)
               VALUES ('context', ?, ?, ?, '{}')""",
            (float("inf"), float("inf"), float("inf")),
        )
        conn.commit()
        summary = tracker.get_summary()

    assert summary["context_avg_tokens"] == 0
    assert summary["context_avg_budget"] == 0
    assert summary["context_budget_util"] == 0.0
    assert summary["context_avg_ms"] == 0

    result = CliRunner().invoke(
        stats_command,
        [],
        obj={"config": config, "json": True, "quiet": True},
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["context_avg_tokens"] == 0
    assert payload["context_budget_util"] == 0.0


def test_empty_burn_rate_reports_zero_active_days(config):
    with StatsTracker(config) as tracker:
        summary = tracker.get_burn_rate()

    assert summary["active_days"] == 0
    assert summary["daily_avg_tokens"] == 0
    assert summary["projections"]["weekly_tokens"] == 0


@pytest.mark.parametrize(
    ("command", "arguments", "option"),
    [
        (remember_command, ["memory", "--type", "unknown"], "--type"),
        (remember_command, ["memory", "--status", "pending"], "--status"),
        (remember_command, ["memory", "--trust-level", "root"], "--trust-level"),
        (remember_command, ["memory", "--confidence", "1.01"], "--confidence"),
        (remember_command, ["memory", "--confidence", "nan"], "--confidence"),
        (decide, ["decision", "--status", "pending"], "--status"),
        (decide, ["decision", "--confidence", "-0.01"], "--confidence"),
        (recall_command, ["query", "--top-k", "0"], "--top-k"),
        (recall_command, ["query", "--top-k", "101"], "--top-k"),
    ],
)
def test_memory_cli_rejects_invalid_metadata_before_running(
    config, command, arguments, option,
):
    result = CliRunner().invoke(command, arguments, obj={"config": config})

    assert result.exit_code == 2
    assert f"Invalid value for '{option}'" in result.output


class _RecordingFastMCP:
    """Small registration double that exposes the actual decorated callables."""

    def __init__(self, name: str, **_kwargs: Any):
        self.name = name
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function

        return register

    def resource(self, uri: str):
        def register(function):
            self.resources[uri] = function
            return function

        return register


@pytest.fixture
def mcp_tools(config, monkeypatch):
    import know.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_server, "FastMCP", _RecordingFastMCP)
    return mcp_server.create_server(project_root=config.root).tools


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("get_context", {"query": "auth", "budget": 0}, "budget must be between"),
        ("get_context", {"query": "auth", "budget": 100_001}, "budget must be between"),
        ("get_context", {"query": "auth", "budget": True}, "budget must be an integer"),
        ("search_code", {"query": "auth", "top_k": 0}, "top_k must be between"),
        ("search_code", {"query": "auth", "top_k": 101}, "top_k must be between"),
        ("recall", {"query": "auth", "top_k": -1}, "top_k must be between"),
        ("recall", {"query": "auth", "top_k": 101}, "top_k must be between"),
    ],
)
def test_registered_mcp_tools_reject_unbounded_work_before_dispatch(
    mcp_tools, tool_name, arguments, message,
):
    with pytest.raises(ValueError, match=message):
        asyncio.run(mcp_tools[tool_name](**arguments))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"text": "memory", "memory_type": "unknown"}, "memory_type must be one of"),
        ({"text": "memory", "decision_status": "pending"}, "decision_status must be one of"),
        ({"text": "memory", "trust_level": "root"}, "trust_level must be one of"),
        ({"text": "memory", "confidence": float("nan")}, "confidence must be a finite"),
        ({"text": "   "}, "text must be a non-blank"),
    ],
)
def test_registered_mcp_remember_surfaces_memory_validation_errors(
    mcp_tools, arguments, message,
):
    with pytest.raises(ValueError, match=message):
        asyncio.run(mcp_tools["remember"](**arguments))
