"""Phase 3 TDD: warm command and non-blocking empty-index behavior."""

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
    config.project.name = "warmup-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path, config


class _ZeroStatsDb:
    def get_stats(self):
        return {"files": 0, "chunks": 0, "imports": 0, "total_tokens": 0}


class TestWarmupBehavior:
    def test_warm_command_json_reports_index_state(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "running": True,
            "stats": {"files": 0, "chunks": 0},
        }

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                ["--config", str(root / ".know" / "config.yaml"), "--json", "warm"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["daemon_running"] is True
        assert payload["indexing_status"] in {"warming", "complete"}

    def test_context_engine_returns_warming_without_inline_index(self, tmp_project, monkeypatch):
        _, config = tmp_project
        from know.context_engine import ContextEngine

        engine = ContextEngine(config)
        db = _ZeroStatsDb()

        called = {"populate_index": 0}

        def _should_not_run(*_args, **_kwargs):
            called["populate_index"] += 1
            raise AssertionError("populate_index should not run in request thread")

        monkeypatch.setattr("know.daemon.populate_index", _should_not_run, raising=True)

        result = engine._build_context_v3_inner(
            db,
            query="billing",
            budget=2000,
            include_tests=False,
            include_imports=True,
            include_patterns=None,
            exclude_patterns=None,
            chunk_types=None,
            session_id=None,
        )

        assert called["populate_index"] == 0
        assert result["indexing_status"] == "warming"
        assert result["code_chunks"] == []
        assert result["used_tokens"] == 0
