"""Usage telemetry tests for workflow/context/grep commands."""

from __future__ import annotations

import json
import subprocess
import sys
import types
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
    config.project.name = "usage-telemetry-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("def auth():\n    return 'ok'\n", encoding="utf-8")
    (src / "b.py").write_text("def token():\n    return 'abc'\n", encoding="utf-8")
    return tmp_path


def test_workflow_json_includes_usage(tmp_project):
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "query": "billing",
        "session_id": "sess1234",
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
        "context": {"query": "billing", "budget": 4000, "used_tokens": 100, "code": []},
        "deep": {"error": "no_target"},
        "latency_ms": {"map": 2, "context": 3, "deep": 0, "total": 5},
        "degraded_by_latency": False,
        "total_tokens": 100,
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(tmp_project / ".know" / "config.yaml"),
                "--json",
                "workflow",
                "billing",
                "--json-full",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["usage"]["source"] == "workflow"
    assert payload["usage"]["tokens_used"] == 100
    assert payload["usage"]["elapsed_ms"] == 5


def test_context_json_includes_usage(tmp_project):
    from know.cli import cli

    fake_client = MagicMock()
    fake_client.call_sync.return_value = {
        "query": "auth",
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
        "source_files": [],
    }

    runner = CliRunner()
    with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
        result = runner.invoke(
            cli,
            [
                "--config",
                str(tmp_project / ".know" / "config.yaml"),
                "--json",
                "context",
                "auth",
            ],
        )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["usage"]["source"] == "context"
    assert payload["usage"]["tokens_used"] == 123
    assert payload["usage"]["elapsed_ms"] >= 0


def test_grep_json_includes_usage(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib
    search_module = importlib.import_module("know.cli.search")

    def _fake_run(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = "\n".join(
            [
                json.dumps({"type": "match", "data": {"path": {"text": "src/a.py"}}}),
                json.dumps({"type": "match", "data": {"path": {"text": "src/b.py"}}}),
            ],
        ) + "\n"
        proc.stderr = ""
        return proc

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_module.subprocess, "run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "--json",
            "grep",
            "auth token",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["strategy"] == "grep_read"
    assert payload["usage"]["source"] == "grep_read"
    assert payload["usage"]["tokens_used"] > 0
    assert payload["usage"]["elapsed_ms"] >= 0


def test_grep_parses_paths_with_numeric_colons_in_match_text(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib

    search_module = importlib.import_module("know.cli.search")

    def _fake_run(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = (
            json.dumps(
                {
                    "type": "match",
                    "data": {
                        "path": {"text": "src/a.py"},
                        "line_number": 1,
                        "lines": {"text": "value:42:tail"},
                    },
                },
            )
            + "\n"
        )
        proc.stderr = ""
        return proc

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_module.subprocess, "run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "--json",
            "grep",
            "value",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["files_read"] == 1
    assert payload["results"][0]["file_path"] == "src/a.py"
    assert payload["usage"]["tokens_used"] > 0


def test_grep_reports_rg_failure(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib

    search_module = importlib.import_module("know.cli.search")

    def _fake_run(*_args, **_kwargs):
        proc = MagicMock()
        proc.returncode = 2
        proc.stdout = ""
        proc.stderr = "bad ripgrep args"
        return proc

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_module.subprocess, "run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_project / ".know" / "config.yaml"), "grep", "auth"],
    )

    assert result.exit_code != 0
    assert "ripgrep failed" in result.output


def test_grep_reports_timeout(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib

    search_module = importlib.import_module("know.cli.search")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="rg", timeout=60)

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_module.subprocess, "run", _timeout)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_project / ".know" / "config.yaml"), "grep", "auth"],
    )

    assert result.exit_code != 0
    assert "timed out" in result.output


def test_grep_requires_ripgrep_binary(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib

    search_module = importlib.import_module("know.cli.search")
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: None)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_project / ".know" / "config.yaml"), "grep", "auth"],
    )

    assert result.exit_code != 0
    assert "ripgrep (rg) is required" in result.output


def test_grep_include_adds_to_default_globs(tmp_project, monkeypatch):
    from know.cli import cli
    import importlib

    search_module = importlib.import_module("know.cli.search")
    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, *_args, **_kwargs):
        captured_cmds.append(list(cmd))
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = json.dumps({"type": "match", "data": {"path": {"text": "src/a.py"}}}) + "\n"
        proc.stderr = ""
        return proc

    monkeypatch.setattr(search_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_module.subprocess, "run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "grep",
            "auth",
            "--include",
            "*.custom",
        ],
    )

    assert result.exit_code == 0
    assert captured_cmds
    cmd = captured_cmds[0]
    assert "--json" in cmd
    assert "--fixed-strings" in cmd
    assert "--" in cmd
    assert "-g" in cmd
    assert "*.py" in cmd
    assert "*.custom" in cmd


def test_grep_rejects_non_positive_limits(tmp_project):
    from know.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "grep",
            "auth",
            "--max-files",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--max-files'" in result.output


def test_search_semantic_json_includes_usage(tmp_project, monkeypatch):
    from know.cli import cli

    class _FakeSearcher:
        def __init__(self, project_root):
            self.project_root = project_root

        def search_code(self, query, root, top_k, auto_index=True):
            return [{"path": "src/a.py", "score": 0.9, "token_count": 11}]

        def search_chunks(self, query, root, top_k, auto_index=True):
            return [{"path": "src/a.py", "name": "auth", "score": 0.8, "token_count": 7}]

    fake_semantic_module = types.ModuleType("know.semantic_search")
    fake_semantic_module.SemanticSearcher = _FakeSearcher
    monkeypatch.setitem(sys.modules, "know.semantic_search", fake_semantic_module)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_project / ".know" / "config.yaml"), "--json", "search", "auth"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["usage"]["source"] == "semantic_search"
    assert payload["usage"]["tokens_used"] == 11


def test_search_bm25_json_includes_usage(tmp_project, monkeypatch):
    from know.cli import cli

    class _RaiseSearcher:
        def __init__(self, project_root):
            raise ImportError("disabled semantic runtime")

    class _DB:
        def search_chunks(self, query, limit):
            return [{"file_path": "src/a.py", "chunk_name": "auth", "token_count": 9}]

    fake_semantic_module = types.ModuleType("know.semantic_search")
    fake_semantic_module.SemanticSearcher = _RaiseSearcher
    monkeypatch.setitem(sys.modules, "know.semantic_search", fake_semantic_module)
    monkeypatch.setattr("know.cli.agent._get_daemon_client", lambda _cfg: None)
    monkeypatch.setattr("know.cli.agent._get_db_fallback", lambda _cfg: _DB())

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_project / ".know" / "config.yaml"), "--json", "search", "auth"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["usage"]["source"] == "bm25_search"
    assert payload["usage"]["tokens_used"] == 9


def test_extract_path_from_rg_event_supports_windows_drive_letter():
    from know.cli.search import _extract_path_from_rg_event

    event = {"type": "match", "data": {"path": {"text": r"C:\repo\src\main.py"}}}
    assert _extract_path_from_rg_event(event) == r"C:\repo\src\main.py"


def test_extract_path_from_rg_event_supports_colon_digit_in_path():
    from know.cli.search import _extract_path_from_rg_event

    event = {"type": "match", "data": {"path": {"text": "dir:1:segment/file.py"}}}
    assert _extract_path_from_rg_event(event) == "dir:1:segment/file.py"
