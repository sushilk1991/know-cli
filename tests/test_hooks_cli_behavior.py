"""CLI behavior tests for hooks install/uninstall/status."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


@pytest.fixture
def tmp_git_project(tmp_path):
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "hooks-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path


@pytest.fixture
def tmp_non_git_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "non-git-hooks-project"
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path


def test_hooks_install_json_is_machine_safe(tmp_git_project):
    from know.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "install"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "installed"
    assert payload["installed_hooks"] == ["post-commit"]
    assert payload["results"][0]["status"] == "installed"


def test_hooks_install_reports_already_installed(tmp_git_project):
    from know.cli import cli

    runner = CliRunner()
    first = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "install"],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "install"],
    )
    assert second.exit_code == 0
    payload = json.loads(second.output)
    assert payload["status"] == "no_change"
    assert payload["installed_hooks"] == []
    assert payload["results"][0]["status"] == "already_installed"


def test_hooks_uninstall_reports_missing_vs_removed(tmp_git_project):
    from know.cli import cli

    runner = CliRunner()
    install = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "install"],
    )
    assert install.exit_code == 0

    uninstall = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "uninstall"],
    )
    assert uninstall.exit_code == 0
    payload = json.loads(uninstall.output)
    assert payload["status"] == "uninstalled"
    assert payload["removed_hooks"] == ["post-commit"]
    assert payload["results"][0]["status"] == "removed"

    uninstall_again = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "uninstall"],
    )
    assert uninstall_again.exit_code == 0
    payload2 = json.loads(uninstall_again.output)
    assert payload2["status"] == "no_change"
    assert payload2["removed_hooks"] == []
    assert payload2["results"][0]["status"] == "missing"


def test_hooks_install_index_hooks_opt_in(tmp_git_project):
    from know.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_git_project / ".know" / "config.yaml"),
            "--json",
            "hooks",
            "install",
            "--index-hooks",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["index_hooks"] is True
    assert set(payload["installed_hooks"]) == {"post-commit", "post-merge", "post-checkout"}


def test_hooks_status_reports_states_plain_and_json(tmp_git_project):
    from know.cli import cli

    hooks_dir = tmp_git_project / ".git" / "hooks"
    (hooks_dir / "post-merge").write_text("#!/bin/bash\necho custom\n", encoding="utf-8")

    runner = CliRunner()
    install = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "hooks", "install"],
    )
    assert install.exit_code == 0

    plain = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "hooks", "status"],
    )
    assert plain.exit_code == 0
    assert "post-commit: installed by know" in plain.output
    assert "post-merge: exists (not managed by know)" in plain.output
    assert "post-checkout: not installed" in plain.output
    assert "pre-commit: not installed" in plain.output

    json_result = runner.invoke(
        cli,
        ["--config", str(tmp_git_project / ".know" / "config.yaml"), "--json", "hooks", "status"],
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["hooks"]["post-commit"] == "installed_by_know"
    assert payload["hooks"]["post-merge"] == "present_other"
    assert payload["hooks"]["post-checkout"] == "missing"
    assert payload["hooks"]["pre-commit"] == "missing"


def test_hooks_status_non_git_repo(tmp_non_git_project):
    from know.cli import cli

    runner = CliRunner()
    plain = runner.invoke(
        cli,
        ["--config", str(tmp_non_git_project / ".know" / "config.yaml"), "hooks", "status"],
    )
    assert plain.exit_code == 0
    assert "Not a git repository" in plain.output

    json_result = runner.invoke(
        cli,
        ["--config", str(tmp_non_git_project / ".know" / "config.yaml"), "--json", "hooks", "status"],
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload == {"hooks": {}}


def test_hooks_uninstall_index_hooks_removes_all_index_hooks(tmp_git_project):
    from know.cli import cli

    runner = CliRunner()
    install = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_git_project / ".know" / "config.yaml"),
            "--json",
            "hooks",
            "install",
            "--index-hooks",
        ],
    )
    assert install.exit_code == 0

    uninstall = runner.invoke(
        cli,
        [
            "--config",
            str(tmp_git_project / ".know" / "config.yaml"),
            "--json",
            "hooks",
            "uninstall",
            "--index-hooks",
        ],
    )
    assert uninstall.exit_code == 0
    payload = json.loads(uninstall.output)
    assert payload["status"] == "uninstalled"
    assert set(payload["removed_hooks"]) == {"post-commit", "post-merge", "post-checkout"}

    by_hook = {row["hook"]: row["status"] for row in payload["results"]}
    assert by_hook["post-commit"] == "removed"
    assert by_hook["post-merge"] == "removed"
    assert by_hook["post-checkout"] == "removed"
