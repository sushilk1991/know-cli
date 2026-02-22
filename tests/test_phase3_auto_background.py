"""Phase 3 tests: background refresh + runtime session autofill."""

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "test-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text('def hello():\n    return "hello"\n')

    return tmp_path, config


def test_diff_file_mtimes_detects_changed_new_removed():
    from know.daemon import _diff_file_mtimes

    previous = {
        "src/a.py": 10,
        "src/b.py": 20,
    }
    current = {
        "src/a.py": 11,  # changed
        "src/c.py": 5,   # new
    }

    changed, removed = _diff_file_mtimes(previous, current)
    assert changed == ["src/a.py", "src/c.py"]
    assert removed == ["src/b.py"]


def test_daemon_refresh_env_defaults_and_minimum(tmp_project, monkeypatch):
    root, config = tmp_project
    from know.daemon import KnowDaemon

    monkeypatch.delenv("KNOW_DAEMON_REFRESH_INTERVAL", raising=False)
    daemon = KnowDaemon(root, config)
    assert daemon._auto_refresh_interval == 60
    daemon.db.close()

    monkeypatch.setenv("KNOW_DAEMON_REFRESH_INTERVAL", "1")
    daemon2 = KnowDaemon(root, config)
    assert daemon2._auto_refresh_interval == 15
    daemon2.db.close()


def test_incremental_refresh_pass_uses_changed_and_removed(tmp_project, monkeypatch):
    root, config = tmp_project
    from know.daemon import KnowDaemon
    import know.daemon as daemon_mod

    daemon = KnowDaemon(root, config)
    daemon._file_mtime_snapshot = {
        "src/main.py": 1,
        "src/old.py": 1,
    }

    monkeypatch.setattr(
        daemon_mod,
        "_collect_project_file_mtimes",
        lambda _cfg: {
            "src/main.py": 2,
            "src/new.py": 3,
        },
    )

    captured = {}

    def fake_refresh(_root, _config, _db, file_paths, **kwargs):
        captured["paths"] = list(file_paths)
        captured["remove_missing"] = kwargs.get("remove_missing", False)
        return {
            "refreshed": 2,
            "removed": 1,
            "skipped": 0,
            "results": [],
        }

    monkeypatch.setattr(daemon_mod, "refresh_files_if_stale", fake_refresh)
    monkeypatch.setattr(daemon.db, "compute_importance", lambda: {"ok": 1})

    summary = daemon._incremental_refresh_once_sync()

    assert summary["refreshed"] == 2
    assert summary["removed"] == 1
    assert set(captured["paths"]) == {"src/main.py", "src/new.py", "src/old.py"}
    assert captured["remove_missing"] is True

    daemon.db.close()


def test_incremental_refresh_skips_importance_for_tiny_updates(tmp_project, monkeypatch):
    root, config = tmp_project
    from know.daemon import KnowDaemon
    import know.daemon as daemon_mod

    daemon = KnowDaemon(root, config)
    daemon._file_mtime_snapshot = {
        "src/main.py": 1,
    }

    monkeypatch.setattr(
        daemon_mod,
        "_collect_project_file_mtimes",
        lambda _cfg: {
            "src/main.py": 2,
        },
    )
    monkeypatch.setattr(
        daemon_mod,
        "refresh_files_if_stale",
        lambda *_args, **_kwargs: {
            "refreshed": 1,
            "removed": 0,
            "skipped": 0,
            "results": [],
        },
    )

    calls = {"importance": 0}

    def _importance():
        calls["importance"] += 1
        return {}

    monkeypatch.setattr(daemon.db, "compute_importance", _importance)
    summary = daemon._incremental_refresh_once_sync()

    assert summary["refreshed"] == 1
    assert calls["importance"] == 0
    daemon.db.close()


class TestSessionAutofill:
    def test_remember_autofills_session_from_runtime_context(self, tmp_project):
        root, config = tmp_project
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase

        (root / ".know" / "current_session").write_text("sess_auto_1\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "remember",
                "Runtime session autofill",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)

        kb = KnowledgeBase(config)
        mem = kb.get(payload["id"])
        assert mem is not None
        assert mem.session_id == "sess_auto_1"
        assert mem.agent == "know-cli"

    def test_decide_autofills_session_from_runtime_context(self, tmp_project):
        root, config = tmp_project
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase

        (root / ".know" / "current_session").write_text("sess_auto_2\n", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "decide",
                "Use daemon auto-refresh",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)

        kb = KnowledgeBase(config)
        mem = kb.get(payload["id"])
        assert mem is not None
        assert mem.memory_type == "decision"
        assert mem.session_id == "sess_auto_2"
        assert mem.agent == "know-cli"


def test_generate_context_handles_text_memory_shape(tmp_project, monkeypatch):
    root, _config = tmp_project
    from know.cli import cli

    class _FakeClient:
        def call_sync(self, method, _params=None):
            if method == "status":
                return {"stats": {"files": 1, "chunks": 1, "total_tokens": 50}}
            if method == "signatures":
                return {
                    "signatures": [
                        {
                            "file_path": "src/main.py",
                            "start_line": 1,
                            "chunk_type": "function",
                            "signature": "def hello():",
                        }
                    ]
                }
            if method == "recall":
                return {"memories": [{"text": "Workflow chose src/main.py"}]}
            raise AssertionError(f"unexpected method: {method}")

    monkeypatch.setattr("know.cli.agent._get_daemon_client", lambda _cfg: _FakeClient())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "generate-context",
            "--budget",
            "600",
        ],
    )

    assert result.exit_code == 0
    context_path = root / ".know" / "CONTEXT.md"
    assert context_path.exists()
    content = context_path.read_text(encoding="utf-8")
    assert "Workflow chose src/main.py" in content
