"""Workflow mode and latency-SLA behavior tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()
    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "workflow-mode-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path, config


def _daemon_workflow_payload():
    return {
        "query": "billing",
        "session_id": "sess1234",
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
        "context": {
            "query": "billing",
            "budget": 4000,
            "used_tokens": 100,
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
        "latency_ms": {"map": 2, "context": 3, "deep": 0, "total": 5},
        "degraded_by_latency": False,
        "total_tokens": 100,
    }


def test_workflow_forwards_mode_and_latency_to_daemon(tmp_project):
    root, _ = tmp_project
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = _daemon_workflow_payload()

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "workflow",
                "agent providers",
                "--mode",
                "explore",
                "--max-latency-ms",
                "2500",
                "--json-full",
            ],
        )

    assert result.exit_code == 0
    args = fake_client.call_sync.call_args[0]
    assert args[0] == "workflow"
    params = args[1]
    assert params["mode"] == "explore"
    assert params["max_latency_ms"] == 2500


def test_workflow_explore_mode_skips_deep_in_fallback(tmp_project, monkeypatch):
    root, _ = tmp_project
    from know.cli import cli

    class _DB:
        def search_signatures(self, _query, _limit):
            return [{"file_path": "src/a.py", "chunk_name": "hello", "signature": "def hello()", "start_line": 1}]

        def close(self):
            return None

    class _Engine:
        def __init__(self):
            self.deep_calls = 0

        def build_context(self, *args, **kwargs):
            return {"query": "q", "budget": 4000, "used_tokens": 10}

        def format_agent_json(self, _result):
            return json.dumps(
                {
                    "query": "q",
                    "budget": 4000,
                    "used_tokens": 10,
                    "indexing_status": "complete",
                    "warnings": [],
                    "code": [
                        {
                            "file": "src/a.py",
                            "name": "hello",
                            "type": "function",
                            "signature": "def hello()",
                            "lines": [1, 3],
                            "tokens": 10,
                            "body": "def hello(): pass",
                        }
                    ],
                    "dependencies": [],
                    "tests": [],
                    "summaries": [],
                    "overview": "",
                    "source_files": ["src/a.py"],
                }
            )

        def build_deep_context(self, *args, **kwargs):
            self.deep_calls += 1
            raise AssertionError("deep should be skipped in explore mode")

    engine = _Engine()

    monkeypatch.setattr("know.context_engine.ContextEngine", lambda _cfg: engine, raising=True)
    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=None):
        with patch("know.cli.agent._get_db_fallback", return_value=_DB()):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "agent providers",
                    "--mode",
                    "explore",
                    "--json-full",
                ],
            )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["workflow_mode"] == "explore"
    assert payload["deep"]["error"] == "skipped_by_mode"
    assert engine.deep_calls == 0


def test_workflow_tiny_latency_budget_skips_context_and_deep(tmp_project, monkeypatch):
    root, _ = tmp_project
    from know.cli import cli

    class _DB:
        def search_signatures(self, _query, _limit):
            return [{"file_path": "src/a.py", "chunk_name": "hello", "signature": "def hello()", "start_line": 1}]

        def close(self):
            return None

    class _Engine:
        def __init__(self):
            self.context_calls = 0
            self.deep_calls = 0

        def build_context(self, *args, **kwargs):
            self.context_calls += 1
            raise AssertionError("context should be skipped when max-latency-ms is tiny")

        def format_agent_json(self, _result):
            return "{}"

        def build_deep_context(self, *args, **kwargs):
            self.deep_calls += 1
            raise AssertionError("deep should be skipped when max-latency-ms is tiny")

    engine = _Engine()
    monkeypatch.setattr("know.context_engine.ContextEngine", lambda _cfg: engine, raising=True)

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=None):
        with patch("know.cli.agent._get_db_fallback", return_value=_DB()):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "agent providers",
                    "--mode",
                    "implement",
                    "--max-latency-ms",
                    "1",
                    "--json-full",
                ],
            )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["context"]["error"] == "skipped_latency_budget"
    assert payload["deep"]["error"] == "skipped_latency_budget"
    assert payload["degraded_by_latency"] is True
    assert engine.context_calls == 0
    assert engine.deep_calls == 0


def test_workflow_stale_daemon_response_falls_back_locally(tmp_project, monkeypatch):
    root, _ = tmp_project
    from know.cli import cli

    # Simulate old daemon payload (missing workflow_mode / latency_ms fields).
    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "query": "legacy daemon",
        "session_id": "sess1234",
        "map": {"results": [], "count": 0, "truncated": False},
        "context": {"query": "legacy daemon", "budget": 4000, "used_tokens": 0, "code": []},
        "deep": {"error": "no_target"},
    }

    class _DB:
        def search_signatures(self, _query, _limit):
            return [{"file_path": "src/a.py", "chunk_name": "hello", "signature": "def hello()", "start_line": 1}]

        def close(self):
            return None

    class _Engine:
        def build_context(self, *args, **kwargs):
            return {"query": "q", "budget": 4000, "used_tokens": 10}

        def format_agent_json(self, _result):
            return json.dumps(
                {
                    "query": "q",
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
                }
            )

        def build_deep_context(self, *args, **kwargs):
            return {"error": "no_target", "reason": "fallback"}

    monkeypatch.setattr("know.context_engine.ContextEngine", lambda _cfg: _Engine(), raising=True)
    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        with patch("know.cli.agent._get_db_fallback", return_value=_DB()):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "legacy daemon",
                    "--mode",
                    "implement",
                    "--json-full",
                ],
            )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    # If fallback happened, new-mode fields are present.
    assert payload["workflow_mode"] == "implement"
    assert "latency_ms" in payload


def test_workflow_non_positive_latency_disables_sla(tmp_project, monkeypatch):
    root, _ = tmp_project
    from know.cli import cli

    class _DB:
        def search_signatures(self, _query, _limit):
            return [{"file_path": "src/a.py", "chunk_name": "hello", "signature": "def hello()", "start_line": 1}]

        def close(self):
            return None

    class _Engine:
        def __init__(self):
            self.context_calls = 0

        def build_context(self, *args, **kwargs):
            self.context_calls += 1
            return {"query": "q", "budget": 4000, "used_tokens": 10}

        def format_agent_json(self, _result):
            return json.dumps(
                {
                    "query": "q",
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
                }
            )

        def build_deep_context(self, *args, **kwargs):
            return {"error": "no_target"}

    engine = _Engine()
    monkeypatch.setattr("know.context_engine.ContextEngine", lambda _cfg: engine, raising=True)
    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=None):
        with patch("know.cli.agent._get_db_fallback", return_value=_DB()):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "non positive latency",
                    "--max-latency-ms",
                    "0",
                    "--json-full",
                ],
            )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["latency_budget_ms"] is None
    assert engine.context_calls == 1
