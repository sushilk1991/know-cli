"""Tests for `know doctor` reliability command."""

import json
import sys
import importlib
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


class TestDoctorCommand:
    def test_doctor_command_exists(self):
        from know.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_doctor_json_report(self, tmp_project, monkeypatch):
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
        data = json.loads(result.output)
        assert data["ok"] is True
        assert "fastembed_cache" in data["checks"]
        assert "embedding_model" in data["checks"]

    def test_doctor_repair_retries_probe(self, tmp_project, monkeypatch):
        root, _ = tmp_project
        from know.cli import cli

        doctor_module = importlib.import_module("know.cli.doctor")
        state = {"count": 0}

        def fake_probe(_model):
            state["count"] += 1
            if state["count"] == 1:
                return False, "model_optimized.onnx missing"
            return True, None

        monkeypatch.setattr(doctor_module, "_probe_embedding_model", fake_probe)
        monkeypatch.setattr(
            doctor_module,
            "_repair_fastembed_cache",
            lambda _path: "cleared cache",
        )
        monkeypatch.setattr(
            doctor_module,
            "_run_reindex_silent",
            lambda _config: (True, 42, None),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(root / ".know" / "config.yaml"),
                "--json",
                "doctor",
                "--repair",
                "--reindex",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["checks"]["embedding_model"]["ok"] is False
        assert data["checks"]["embedding_model_after_repair"]["ok"] is True
        assert "cleared cache" in data["actions"]
        assert "reindex indexed 42 chunks" in data["actions"]
