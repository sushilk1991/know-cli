"""Tests for search fallback diagnostics/remediation messaging."""

import importlib
import sys
from pathlib import Path

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
    config.project.name = "search-fallback-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    return tmp_path


def test_search_fallback_missing_runtime_tip(tmp_project, monkeypatch):
    from know.cli import cli
    search_module = importlib.import_module("know.cli.search")

    class _EmptyDB:
        def search_chunks(self, query, limit):
            return []

    monkeypatch.setattr("know.cli.agent._get_daemon_client", lambda _config: None)
    monkeypatch.setattr("know.cli.agent._get_db_fallback", lambda _config: _EmptyDB())
    monkeypatch.setattr(
        search_module,
        "_embedding_runtime_diagnostics",
        lambda: {
            "fastembed_installed": False,
            "onnxruntime_installed": False,
            "distribution_version": "0.8.7",
            "module_version": "0.8.7",
            "editable_install": False,
            "version_mismatch": False,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "search",
            "hello",
        ],
    )

    assert result.exit_code == 2
    assert "python -m pip install -U know-cli" in result.output
    assert "know doctor --repair --reindex" in result.output
    assert "know-cli[search]" not in result.output


def test_search_fallback_editable_tip(tmp_project, monkeypatch):
    from know.cli import cli
    search_module = importlib.import_module("know.cli.search")

    class _ResultDB:
        def search_chunks(self, query, limit):
            return [
                {
                    "file_path": "src/main.py",
                    "chunk_name": "hello",
                    "signature": "def hello()",
                }
            ]

    monkeypatch.setattr("know.cli.agent._get_daemon_client", lambda _config: None)
    monkeypatch.setattr("know.cli.agent._get_db_fallback", lambda _config: _ResultDB())
    monkeypatch.setattr(
        search_module,
        "_embedding_runtime_diagnostics",
        lambda: {
            "fastembed_installed": True,
            "onnxruntime_installed": True,
            "distribution_version": "0.5.2",
            "module_version": "0.8.7",
            "editable_install": True,
            "version_mismatch": True,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "search",
            "hello",
        ],
    )

    assert result.exit_code == 0
    assert "editable install detected" in result.output
