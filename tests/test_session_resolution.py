"""Session ID resolution tests for agent commands."""

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
    config.project.name = "session-resolution-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path, config


def test_map_session_auto_resolves_and_persists(tmp_project):
    root, _ = tmp_project
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "results": [
            {
                "file_path": "src/main.py",
                "chunk_name": "main",
                "chunk_type": "function",
                "signature": "def main()",
                "start_line": 1,
                "end_line": 2,
                "score": 1.0,
            }
        ]
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "map",
                "main",
                "--session",
                "auto",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    resolved = payload["session_id"]
    assert resolved != "auto"
    assert len(resolved) == 8

    params = fake_client.call_sync.call_args[0][1]
    assert params["session_id"] == resolved

    session_file = root / ".know" / "current_session"
    assert session_file.read_text(encoding="utf-8").strip() == resolved


def test_deep_session_auto_resolves_before_daemon_call(tmp_project):
    root, _ = tmp_project
    from know.cli import cli

    captured = {}

    def _fake_call(method, params):
        captured["method"] = method
        captured["params"] = params
        return {
            "target": {
                "file": "src/main.py",
                "name": "main",
                "signature": "def main()",
                "body": "def main():\n    pass",
                "line_start": 1,
                "line_end": 2,
                "tokens": 20,
            },
            "callees": [],
            "callers": [],
            "overflow_signatures": [],
            "call_graph_available": True,
            "call_graph_reason": None,
            "budget_used": 20,
            "budget": 3000,
        }

    fake_client = MagicMock()
    fake_client.call_sync.side_effect = _fake_call

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "deep",
                "main",
                "--session",
                "auto",
            ],
        )

    assert result.exit_code == 0
    assert captured["method"] == "deep"
    resolved = captured["params"]["session_id"]
    assert resolved != "auto"
    assert len(resolved) == 8

    session_file = root / ".know" / "current_session"
    assert session_file.read_text(encoding="utf-8").strip() == resolved

