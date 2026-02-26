"""Phase 1 TDD: JSON compliance and doctor diagnostics."""

import importlib
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
    config.project.name = "phase1-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")

    return tmp_path, config


class _DummyDB:
    def close(self):
        return None


class TestJsonCompliance:
    def test_next_file_respects_global_json_flag(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "results": [
                {
                    "file_path": "src/main.py",
                    "chunk_name": "hello",
                    "score": 0.9,
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
                    "next-file",
                    "hello",
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["file"] == "src/main.py"

    def test_related_respects_global_json_flag(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "imports": ["pkg.a"],
            "imported_by": ["pkg.b"],
        }

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            with patch("know.cli.agent._get_db_fallback", return_value=_DummyDB()):
                with patch("know.daemon.refresh_file_if_stale", return_value=False):
                    with patch(
                        "know.import_graph.ImportGraph.related_files_from_modules",
                        return_value=([], []),
                    ):
                        result = runner.invoke(
                            cli,
                            [
                                "--config",
                                str(root / ".know" / "config.yaml"),
                                "--json",
                                "related",
                                "src/main.py",
                            ],
                        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["imports"] == ["pkg.a"]
        assert payload["imported_by"] == ["pkg.b"]

    def test_callers_respects_global_json_flag(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "callers": [{"file_path": "src/main.py", "containing_chunk": "foo", "line_number": 10}]
        }

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "callers",
                    "hello",
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 1
        assert payload["callers"][0]["containing_chunk"] == "foo"

    def test_callees_respects_global_json_flag(self, tmp_project):
        root, _ = tmp_project
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "callees": [{"ref_name": "bar", "ref_type": "function", "file_path": "src/main.py"}]
        }

        runner = CliRunner()
        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(
                cli,
                [
                    "--config",
                    str(root / ".know" / "config.yaml"),
                    "--json",
                    "callees",
                    "hello",
                ],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["count"] == 1
        assert payload["callees"][0]["ref_name"] == "bar"


class TestDoctorDiagnostics:
    def test_doctor_has_command_resolution_diagnostics(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        doctor_module = importlib.import_module("know.cli.doctor")
        monkeypatch.setattr(doctor_module, "_probe_embedding_model", lambda _m: (True, None))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "doctor",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        env = payload["environment"]
        assert "know_command" in env
        assert "know_module_file" in env
        assert "know_version" in env
        assert "repair_command" in payload
