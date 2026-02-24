"""Tests for agent skill installation and bootstrap behavior."""

import sys
from pathlib import Path

from click.testing import CliRunner

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know import __version__
from know.skill_installer import (
    auto_bootstrap_skill_install,
    install_skill_file,
    skill_bootstrap_marker_path,
    skill_target_paths,
)


def test_skill_target_paths_cover_common_agents(tmp_path):
    targets = skill_target_paths(home=tmp_path)
    assert "codex" in targets
    assert "claude" in targets
    assert "agents" in targets

    assert str(targets["codex"]).endswith(".codex/skills/know-cli/SKILL.md")
    assert str(targets["claude"]).endswith(".claude/skills/know-cli/SKILL.md")
    assert str(targets["agents"]).endswith(".agents/skills/know-cli/SKILL.md")


def test_packaged_skill_template_is_in_sync():
    repo_skill = Path(__file__).parent.parent / "KNOW_SKILL.md"
    packaged_skill = Path(__file__).parent.parent / "src" / "know" / "resources" / "KNOW_SKILL.md"
    assert packaged_skill.exists()
    assert repo_skill.read_text(encoding="utf-8") == packaged_skill.read_text(encoding="utf-8")


def test_install_skill_file_installs_to_all_targets(tmp_path):
    result = install_skill_file(home=tmp_path, force=False)
    assert result["template_available"] is True
    assert result["installed_count"] >= 1
    assert result["error_count"] == 0

    for target in skill_target_paths(home=tmp_path).values():
        assert target.exists()
        text = target.read_text(encoding="utf-8")
        assert "know-cli skill" in text.lower()
        assert "know workflow" in text.lower()


def test_install_skill_file_does_not_overwrite_without_force(tmp_path):
    targets = skill_target_paths(home=tmp_path)
    target = targets["codex"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("custom user skill\n", encoding="utf-8")

    result = install_skill_file(home=tmp_path, force=False)
    assert result["error_count"] == 0
    assert target.read_text(encoding="utf-8") == "custom user skill\n"


def test_install_skill_file_force_overwrites(tmp_path):
    targets = skill_target_paths(home=tmp_path)
    target = targets["codex"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("custom user skill\n", encoding="utf-8")

    result = install_skill_file(home=tmp_path, force=True)
    assert result["error_count"] == 0
    assert "know workflow" in target.read_text(encoding="utf-8").lower()
    assert result["installed_count"] >= 1


def test_auto_bootstrap_writes_marker_and_is_idempotent(tmp_path, monkeypatch):
    marker = skill_bootstrap_marker_path(home=tmp_path)
    assert not marker.exists()

    first = auto_bootstrap_skill_install(home=tmp_path)
    assert first["attempted"] is True
    assert marker.exists()

    second = auto_bootstrap_skill_install(home=tmp_path)
    assert second["attempted"] is False
    assert second["reason"] == "already_bootstrapped"

    marker_payload = marker.read_text(encoding="utf-8")
    assert __version__ in marker_payload


def test_auto_bootstrap_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KNOW_AUTO_INSTALL_SKILL", "0")
    result = auto_bootstrap_skill_install(home=tmp_path)
    assert result["attempted"] is False
    assert result["reason"] == "disabled"


def test_cli_help_bootstraps_skill_install(tmp_path, monkeypatch):
    from know.cli import cli

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    marker = skill_bootstrap_marker_path(home=tmp_path)
    assert not marker.exists()

    runner = CliRunner()
    result = runner.invoke(cli, ["commands"])
    assert result.exit_code == 0
    assert marker.exists()

    for target in skill_target_paths(home=tmp_path).values():
        assert target.exists()
