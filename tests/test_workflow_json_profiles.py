"""Phase 2 TDD: workflow JSON compact/full profile behavior."""

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
    config.project.name = "workflow-profile-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path, config


def _full_workflow_payload():
    return {
        "query": "billing",
        "session_id": "abcd1234",
        "daemon_api_version": 2,
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "map": {
            "results": [
                {
                    "file_path": "src/billing/service.py",
                    "chunk_name": "check_cloud_access",
                    "chunk_type": "function",
                    "start_line": 10,
                    "score": 2.1,
                }
            ],
            "count": 1,
            "truncated": False,
            "tokens": 40,
        },
        "context": {
            "query": "billing",
            "budget": 4000,
            "used_tokens": 900,
            "budget_utilization": "900 / 4,000 (22%)",
            "indexing_status": "complete",
            "confidence": 0.78,
            "warnings": [],
            "code": [
                {
                    "file": "src/billing/service.py",
                    "name": "check_cloud_access",
                    "type": "function",
                    "signature": "def check_cloud_access(workspace):",
                    "lines": [10, 40],
                    "score": 2.3,
                    "tokens": 200,
                    "body": "def check_cloud_access(workspace):\n    return True",
                }
            ],
            "dependencies": [],
            "tests": [],
            "summaries": [],
            "overview": "",
            "source_files": ["src/billing/service.py"],
        },
        "deep": {
            "target": {
                "file": "src/billing/service.py",
                "name": "check_cloud_access",
                "line_start": 10,
                "line_end": 40,
            },
            "callers": [{"name": "handle_request", "file": "src/api/router.py"}],
            "callees": [{"name": "count_active_sessions", "file": "src/billing/service.py"}],
            "call_graph_available": True,
            "call_graph_reason": None,
            "budget_used": 700,
        },
        "selected_deep_target": "check_cloud_access",
        "total_tokens": 1640,
        "latency_ms": {"map": 2, "context": 8, "deep": 4, "total": 14},
        "degraded_by_latency": False,
    }


class TestWorkflowProfiles:
    def test_json_default_tty_prefers_compact_profile(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = _full_workflow_payload()

        monkeypatch.setattr("know.cli.agent._stdout_is_tty", lambda: True, raising=False)

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                ["--config", str(root / ".know" / "config.yaml"), "--json", "workflow", "billing"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["metrics"]["profile"] == "compact"
        assert "map" not in payload
        assert "context" in payload
        assert "deep" in payload
        assert "targets" in payload
        candidates = payload["targets"]["candidates"]
        assert candidates
        assert "file_path" in candidates[0]
        assert candidates[0]["file_path"] == "src/billing/service.py"
        assert payload["targets"]["selected_file_path"] == "src/billing/service.py"
        assert payload["deep"]["target"]["file_path"] == "src/billing/service.py"

    def test_json_default_non_tty_stays_full(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = _full_workflow_payload()

        monkeypatch.setattr("know.cli.agent._stdout_is_tty", lambda: False, raising=False)

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                ["--config", str(root / ".know" / "config.yaml"), "--json", "workflow", "billing"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "map" in payload and "context" in payload and "deep" in payload
        assert "metrics" not in payload or payload.get("metrics", {}).get("profile") != "compact"

    def test_json_full_flag_forces_full_on_tty(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = _full_workflow_payload()

        monkeypatch.setattr("know.cli.agent._stdout_is_tty", lambda: True, raising=False)

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "billing",
                    "--json-full",
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "map" in payload and "context" in payload and "deep" in payload

    def test_json_compact_flag_forces_compact_on_non_tty(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = _full_workflow_payload()

        monkeypatch.setattr("know.cli.agent._stdout_is_tty", lambda: False, raising=False)

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "workflow",
                    "billing",
                    "--json-compact",
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["metrics"]["profile"] == "compact"
        assert "map" not in payload

    def test_profile_flags_are_mutually_exclusive(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "workflow",
                "billing",
                "--json-compact",
                "--json-full",
            ],
        )

        assert result.exit_code != 0
