"""Regression tests for `know update` command behavior."""

import json

from click.testing import CliRunner

from know.cli import cli
from know.config import Config


def _make_project(tmp_path):
    (tmp_path / ".know").mkdir()
    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "test-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        'def hello():\n'
        '    return "hello"\n',
        encoding="utf-8",
    )
    return tmp_path


def test_update_only_system_is_scoped_and_json_clean(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = _make_project(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "--json",
            "update",
            "--only",
            "system",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert [item["type"] for item in data["updated"]] == ["system"]


def test_update_without_only_updates_all(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = _make_project(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "--json",
            "update",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    types = [item["type"] for item in data["updated"]]
    assert "system" in types
    assert "diagram" in types
    assert "onboarding" in types


def test_update_all_flag_updates_all(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = _make_project(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "--json",
            "update",
            "--all",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    types = [item["type"] for item in data["updated"]]
    assert "system" in types
    assert "diagram" in types
    assert "onboarding" in types


def test_update_rejects_all_and_only_together(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    root = _make_project(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(root / ".know" / "config.yaml"),
            "update",
            "--all",
            "--only",
            "system",
        ],
    )

    assert result.exit_code != 0
    assert "Cannot combine --all with --only" in result.output
