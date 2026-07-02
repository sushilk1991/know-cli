"""TDD guardrails for context-intelligence-only upgrade scope."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config
from know.stats import StatsTracker


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "context-intelligence-upgrade-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def hello(name):\n"
        "    return f'hello {name}'\n",
        encoding="utf-8",
    )

    return tmp_path, config


def test_confidence_command_removed_from_registry():
    from know.cli import cli

    commands = cli.list_commands(None)
    assert "confidence" not in commands

    runner = CliRunner()
    result = runner.invoke(cli, ["commands", "--all"])
    assert result.exit_code == 0
    assert "confidence" not in result.output


def test_hooks_suggest_claude_json_contract(tmp_project):
    root, _ = tmp_project
    from know.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "--json",
            "hooks",
            "suggest",
            "--agent",
            "claude",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert payload["mode"] == "suggest"
    assert payload["agent"] == "claude"
    assert payload["mutation_safe"] is True
    assert "snippet" in payload and payload["snippet"]
    assert "updatedInput" not in json.dumps(payload)


def test_hooks_suggest_codex_plain_guidance(tmp_project):
    root, _ = tmp_project
    from know.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "hooks",
            "suggest",
            "--agent",
            "codex",
        ],
    )

    assert result.exit_code == 0
    assert "know workflow" in result.output
    assert "skill" in result.output.lower()
    assert "updatedInput" not in result.output


def test_retrieval_events_recorded_and_summary_contract(tmp_project):
    root, config = tmp_project
    from know.cli import cli

    map_client = MagicMock()
    map_client.call_sync.return_value = {
        "results": [
            {
                "file_path": "src/main.py",
                "chunk_name": "hello",
                "chunk_type": "function",
                "signature": "def hello(name):",
                "start_line": 1,
                "end_line": 2,
                "score": 1.0,
                "docstring": "",
            }
        ],
    }

    context_client = MagicMock()
    context_client.call_sync.return_value = {
        "query": "hello",
        "budget": 4000,
        "used_tokens": 123,
        "budget_utilization": "123 / 4,000 (3%)",
        "indexing_status": "complete",
        "warnings": [],
        "code": [],
        "dependencies": [],
        "tests": [],
        "summaries": [],
        "overview": "",
        "source_files": ["src/main.py"],
        "confidence": 0.7,
    }

    deep_client = MagicMock()
    deep_client.call_sync.return_value = {
        "target": {
            "file": "src/main.py",
            "name": "hello",
            "signature": "def hello(name):",
            "body": "def hello(name):\n    return f'hello {name}'",
            "line_start": 1,
            "line_end": 2,
            "tokens": 30,
        },
        "callees": [{"name": "print", "file": "src/main.py"}],
        "callers": [{"name": "main", "file": "src/main.py"}],
        "overflow_signatures": [],
        "call_graph_available": True,
        "call_graph_reason": None,
        "budget_used": 30,
        "budget": 3000,
    }

    workflow_client = MagicMock()
    workflow_client.call_sync.return_value = {
        "query": "hello",
        "session_id": "sess1234",
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 10},
        "context": {
            "query": "hello",
            "budget": 4000,
            "used_tokens": 80,
            "indexing_status": "complete",
            "warnings": [],
            "code": [],
            "dependencies": [],
            "tests": [],
            "summaries": [],
            "overview": "",
            "source_files": ["src/main.py"],
        },
        "deep": {
            "target": {"file": "src/main.py", "name": "hello", "line_start": 1, "line_end": 2},
            "call_graph_available": True,
            "callers": [{"name": "main", "file": "src/main.py"}],
            "callees": [{"name": "print", "file": "src/main.py"}],
            "budget_used": 20,
        },
        "latency_ms": {"map": 1, "context": 2, "deep": 2, "total": 5},
        "degraded_by_latency": False,
        "total_tokens": 110,
    }

    runner = CliRunner()

    with patch("know.cli.agent._get_daemon_client", return_value=map_client):
        map_result = runner.invoke(
            cli,
            ["--config", str(root / ".know" / "config.yaml"), "--json", "map", "hello"],
        )
    assert map_result.exit_code == 0

    with patch("know.cli.agent._get_daemon_client", return_value=context_client):
        context_result = runner.invoke(
            cli,
            ["--config", str(root / ".know" / "config.yaml"), "--json", "context", "hello"],
        )
    assert context_result.exit_code == 0

    with patch("know.cli.agent._get_daemon_client", return_value=deep_client):
        deep_result = runner.invoke(
            cli,
            ["--config", str(root / ".know" / "config.yaml"), "--json", "deep", "hello"],
        )
    assert deep_result.exit_code == 0

    with patch("know.cli.agent._get_daemon_client", return_value=workflow_client):
        workflow_result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "workflow",
                "hello",
                "--json-full",
            ],
        )
    assert workflow_result.exit_code == 0

    tracker = StatsTracker(config)
    retrieval = tracker.get_retrieval_summary(days=30)

    assert retrieval["window_days"] == 30
    for command_name in ("map", "context", "deep", "workflow"):
        assert retrieval["commands"][command_name]["count"] >= 1
        assert "p50_ms" in retrieval["commands"][command_name]
        assert "p95_ms" in retrieval["commands"][command_name]
        assert "avg_tokens" in retrieval["commands"][command_name]

    quality = retrieval["workflow_quality"]
    assert "degraded_by_latency_rate_pct" in quality
    assert "call_graph_available_rate_pct" in quality
    assert "non_empty_edge_rate_pct" in quality

    summary = tracker.get_summary()
    assert "context_queries" in summary
    assert "search_queries" in summary
    assert "remember_count" in summary
    assert "recall_count" in summary


def test_workflow_records_single_event_per_invocation(tmp_project):
    root, config = tmp_project
    from know.cli import cli

    workflow_client = MagicMock()
    workflow_client.call_sync.return_value = {
        "query": "hello",
        "session_id": "sess1234",
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
        "context": {
            "query": "hello",
            "budget": 4000,
            "used_tokens": 10,
            "indexing_status": "complete",
            "warnings": [],
            "code": [],
            "dependencies": [],
            "tests": [],
            "summaries": [],
            "overview": "",
            "source_files": [],
        },
        "deep": {"error": "no_target", "reason": "no_context_or_map_target"},
        "latency_ms": {"map": 1, "context": 1, "deep": 1, "total": 3},
        "degraded_by_latency": False,
        "total_tokens": 10,
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=workflow_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "workflow",
                "hello",
                "--json-full",
            ],
        )
    assert result.exit_code == 0

    db_path = root / ".know" / "stats.db"
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'workflow'",
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == 1


def test_stats_json_includes_retrieval_block(tmp_project):
    root, config = tmp_project
    from know.cli import cli

    tracker = StatsTracker(config)
    tracker.record_map("hello", results_count=1, duration_ms=11, tokens_used=12)
    tracker.record_context("hello", budget=4000, tokens_used=120, duration_ms=21)
    tracker.record_deep(
        "hello",
        duration_ms=31,
        tokens_used=32,
        call_graph_available=True,
        callers_count=1,
        callees_count=1,
    )
    tracker.record_workflow(
        "hello",
        duration_ms=41,
        tokens_used=140,
        mode="implement",
        degraded_by_latency=False,
        call_graph_available=True,
        callers_count=1,
        callees_count=1,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(root / ".know" / "config.yaml"), "--json", "stats"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)

    assert "retrieval" in payload
    assert payload["retrieval"]["window_days"] == 30
    assert set(payload["retrieval"]["commands"].keys()) == {
        "map", "context", "deep", "workflow",
    }


def test_stats_human_output_shows_retrieval_kpi_section(tmp_project):
    root, config = tmp_project
    from know.cli import cli

    tracker = StatsTracker(config)
    tracker.record_workflow(
        "hello",
        duration_ms=40,
        tokens_used=120,
        mode="implement",
        degraded_by_latency=False,
        call_graph_available=True,
        callers_count=1,
        callees_count=1,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(root / ".know" / "config.yaml"), "stats"],
    )

    assert result.exit_code == 0
    assert "Retrieval KPI" in result.output
    assert "workflow" in result.output
