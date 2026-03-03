"""Tests for simplified CLI surface with backward compatibility."""

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
    """Create a minimal project structure for CLI tests."""
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "test-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        'def hello():\n'
        '    """Say hello."""\n'
        '    return "hello"\n'
    )

    return tmp_path, config


class TestSimplifiedSurface:
    """Default UX should remain easy while keeping full discoverability."""

    def test_default_help_shows_common_and_all(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        # Human-first commands should be visible.
        assert "ask" in result.output
        assert "docs" in result.output
        assert "recall" in result.output
        assert "decide" in result.output
        assert "done" in result.output
        assert "status" in result.output
        # Advanced commands should be visible in default help too.
        assert "Advanced Commands" in result.output
        assert "workflow" in result.output
        assert "hooks" in result.output
        assert "watch" in result.output

    def test_simple_commands_registered(self):
        from know.cli import cli

        commands = cli.list_commands(None)
        assert "ask" in commands
        assert "docs" in commands
        assert "recall" in commands
        assert "decide" in commands
        assert "done" in commands
        assert "status" in commands
        assert "commands" in commands

    def test_commands_default_lists_simple_set(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["commands"])

        assert result.exit_code == 0
        names = [
            line.split()[0]
            for line in result.output.splitlines()
            if line.strip()
        ]
        assert "workflow" in names
        assert "ask" in names
        assert "docs" in names
        assert "status" in names
        assert "context" not in names
        assert "hooks" not in names

    def test_commands_simple_lists_curated_set(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["commands", "--simple"])

        assert result.exit_code == 0
        assert "workflow" in result.output
        assert "ask" in result.output
        assert "docs" in result.output
        assert "status" in result.output

    def test_commands_json_preserves_all_field(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--json", "commands"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["all"] is False

        result_simple = runner.invoke(cli, ["--json", "commands", "--simple"])
        assert result_simple.exit_code == 0
        payload_simple = json.loads(result_simple.output)
        assert payload_simple["all"] is False

        result_all = runner.invoke(cli, ["--json", "commands", "--all"])
        assert result_all.exit_code == 0
        payload_all = json.loads(result_all.output)
        assert payload_all["all"] is True

    def test_commands_all_lists_legacy(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["commands", "--all"])

        assert result.exit_code == 0
        assert "workflow" in result.output
        assert "context" in result.output
        assert "map" in result.output
        assert "deep" in result.output

    def test_commands_all_is_exhaustive_and_no_duplicates(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["commands", "--all"])

        assert result.exit_code == 0
        listed = [
            line.split()[0]
            for line in result.output.splitlines()
            if line.strip()
        ]
        assert len(listed) == len(set(listed))
        assert set(listed) == set(cli.list_commands(None))

    def test_docs_command_json_output(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "docs",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "digest" in data
        assert "diagram" in data


class TestBackwardCompatibility:
    """Legacy top-level commands must continue to work."""

    def test_legacy_commands_still_registered(self):
        from know.cli import cli

        commands = cli.list_commands(None)
        assert "workflow" in commands
        assert "context" in commands
        assert "diagram" in commands
        assert "digest" in commands
        assert "remember" in commands
        assert "memories" in commands

    def test_legacy_workflow_help_still_works(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["workflow", "--help"])
        assert result.exit_code == 0

    def test_done_alias_resolves_memory(self, tmp_project):
        root, config = tmp_project
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(config)
        memory_id = kb.remember(
            "Use daemon workflow by default",
            memory_type="decision",
            decision_status="active",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(root / ".know" / "config.yaml"), "done", str(memory_id)],
        )

        assert result.exit_code == 0

        updated = kb.get(memory_id)
        assert updated is not None
        assert updated.decision_status == "resolved"
