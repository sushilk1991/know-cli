"""Adversarial regressions for CLI-generated files and optional integrations."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner

from know.cli.core import onboard
from know.cli.mcp import mcp
from know.cli.search import search
from know.config import Config
from know.generator import DocGenerator
from know.generator import atomic_write_text
import know.generator as generator_module
import know.skill_installer as skill_installer


@pytest.fixture
def config(tmp_path: Path) -> Config:
    result = Config.create_default(tmp_path)
    result.output.directory = "docs"
    return result


def _generate_plantuml(generator: DocGenerator, output: str) -> Path:
    generator.config.diagrams.format = "plantuml"
    return generator.generate_c4_diagram({}, output)


@pytest.mark.parametrize(
    "generate",
    [
        lambda generator, output: generator.generate_c4_diagram({}, output),
        _generate_plantuml,
        lambda generator, output: generator.generate_dependency_graph({}, output),
        lambda generator, output: generator.generate_openapi([], output),
        lambda generator, output: generator.generate_postman([], output),
        lambda generator, output: generator.generate_api_markdown([], output),
    ],
    ids=["mermaid", "plantuml", "dependencies", "openapi", "postman", "api-markdown"],
)
def test_explicit_generator_outputs_create_nested_parent_directories(
    config: Config,
    tmp_path: Path,
    generate,
) -> None:
    output = tmp_path / "new" / "nested" / "artifact.txt"

    result = generate(DocGenerator(config), str(output))

    assert result == output
    assert output.is_file()


def test_generator_overwrite_is_atomic_when_replace_fails(
    config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "arc.md"
    output.write_text("irreplaceable old content", encoding="utf-8")

    def fail_replace(_source, _destination) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(generator_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated rename failure"):
        DocGenerator(config).generate_system_doc({}, str(output))

    assert output.read_text(encoding="utf-8") == "irreplaceable old content"
    assert list(tmp_path.glob(".arc.md.*.tmp")) == []


def test_atomic_write_preserves_existing_mode(tmp_path: Path) -> None:
    output = tmp_path / "private.txt"
    output.write_text("old", encoding="utf-8")
    output.chmod(0o640)

    atomic_write_text(output, "new")

    assert output.read_text(encoding="utf-8") == "new"
    assert output.stat().st_mode & 0o7777 == 0o640


def test_atomic_write_new_file_respects_process_umask(tmp_path: Path) -> None:
    output = tmp_path / "private.txt"
    previous_umask = os.umask(0o077)
    try:
        atomic_write_text(output, "private")
    finally:
        os.umask(previous_umask)

    assert output.stat().st_mode & 0o7777 == 0o600


def test_onboarding_audience_cannot_escape_output_directory_or_inject_title_markup(
    config: Config,
    tmp_path: Path,
) -> None:
    output_dir = (tmp_path / "docs").resolve()
    audience = "../../<script>alert(1)</script>"

    path = DocGenerator(config).save_onboarding("Welcome", audience, "html")

    assert path.resolve().parent == output_dir
    assert path.suffix == ".html"
    assert "/" not in path.name
    content = path.read_text(encoding="utf-8")
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


def test_save_onboarding_rejects_unimplemented_pdf_format(config: Config) -> None:
    with pytest.raises(ValueError, match="Unsupported onboarding format: pdf"):
        DocGenerator(config).save_onboarding("Welcome", "new developers", "pdf")


def test_onboard_cli_rejects_pdf_before_running_generation(config: Config) -> None:
    result = CliRunner().invoke(
        onboard,
        ["--for", "new developers", "--format", "pdf"],
        obj={"config": config, "quiet": True, "json": False},
    )

    assert result.exit_code == 2
    assert "Invalid value for '--format'" in result.output
    assert not (config.root / config.output.directory).exists()


def test_missing_mcp_dependency_is_an_actionable_click_failure(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import know.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server, "_MCP_AVAILABLE", False)

    result = CliRunner().invoke(
        mcp,
        ["serve"],
        obj={"config": config},
    )

    assert result.exit_code == 1
    assert "pip install know-cli[mcp]" in result.output
    assert not isinstance(result.exception, ImportError)


@pytest.mark.parametrize("port", ["0", "65536"])
def test_mcp_serve_rejects_ports_outside_the_tcp_range(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    import know.mcp_server as mcp_server

    called = False

    def run_server(**_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(mcp_server, "run_server", run_server)
    result = CliRunner().invoke(
        mcp,
        ["serve", "--sse", "--port", port],
        obj={"config": config},
    )

    assert result.exit_code == 2
    assert "65535" in result.output
    assert called is False


def test_mcp_serve_does_not_misdiagnose_unrelated_import_errors(
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import know.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server, "_MCP_AVAILABLE", True)

    def fail_server(**_kwargs) -> None:
        raise ImportError("No module named project_plugin", name="project_plugin")

    monkeypatch.setattr(mcp_server, "run_server", fail_server)
    result = CliRunner().invoke(
        mcp,
        ["serve"],
        obj={"config": config},
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ImportError)
    assert "know-cli[mcp]" not in result.output


def test_failed_skill_bootstrap_does_not_mark_version_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail_install(**_kwargs) -> dict:
        nonlocal calls
        calls += 1
        return {
            "template_available": True,
            "installed_count": 2,
            "skipped_count": 0,
            "error_count": 1,
            "errors": [{"target": "agents", "error": "permission denied"}],
        }

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(skill_installer, "install_skill_file", fail_install)
    marker = skill_installer.skill_bootstrap_marker_path(home=tmp_path)

    first = skill_installer.auto_bootstrap_skill_install(home=tmp_path)
    second = skill_installer.auto_bootstrap_skill_install(home=tmp_path)

    assert first["attempted"] is True
    assert first["reason"] == "install_failed"
    assert second["attempted"] is True
    assert calls == 2
    assert not marker.exists()


@pytest.mark.parametrize("top_k", ["0", "-1", "101"])
def test_search_cli_rejects_unbounded_top_k_before_searching(
    config: Config,
    top_k: str,
) -> None:
    result = CliRunner().invoke(
        search,
        ["authentication", "--top-k", top_k],
        obj={"config": config, "quiet": True, "json": False},
    )

    assert result.exit_code == 2
    assert "1<=x<=100" in result.output


def test_json_time_keeps_stdout_as_one_json_document(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["KNOW_AUTO_INSTALL_SKILL"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from know.cli import main; main()",
            "--json",
            "--quiet",
            "--time",
            "init",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "initialized"
    assert "⏱" not in completed.stdout
    assert "⏱" in completed.stderr
