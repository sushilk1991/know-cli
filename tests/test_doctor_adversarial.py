"""Adversarial tests for destructive and aggregate doctor behavior."""

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from know.config import Config

doctor_module = importlib.import_module("know.cli.doctor")


def _healthy_dependencies(**overrides):
    dependencies = {
        "fastembed_installed": True,
        "onnxruntime_installed": True,
        "distribution_version": "0.8.7",
        "module_version": "0.8.7",
        "editable_install": False,
        "version_mismatch": False,
        "declares_fastembed_dependency": True,
    }
    dependencies.update(overrides)
    return dependencies


@pytest.fixture
def tmp_project(tmp_path):
    project = tmp_path / "project"
    (project / ".know" / "cache").mkdir(parents=True)
    config = Config.create_default(project)
    config.root = project
    config.project.name = "doctor-adversarial"
    config.save(project / ".know" / "config.yaml")
    return project


def test_doctor_cache_repair_accepts_dedicated_cache_identity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    cache_root = home / ".cache" / "know-cli" / "fastembed"
    model = cache_root / "models--qdrant--broken"
    model.mkdir(parents=True)
    (model / "corrupt.bin").write_bytes(b"corrupt")

    action = doctor_module._repair_fastembed_cache(
        cache_root, project_root=tmp_path / "unrelated-project"
    )

    assert action == f"cleared cache at {cache_root.resolve()}"
    assert cache_root.is_dir()
    assert list(cache_root.iterdir()) == []


@pytest.mark.parametrize(
    "candidate_name",
    ["filesystem_root", "home", "project", "unrelated"],
)
def test_doctor_cache_repair_refuses_unsafe_roots(candidate_name, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = (
        home / ".cache" / "know-cli" / "fastembed"
        if candidate_name == "project"
        else tmp_path / "project"
    )
    project.mkdir(parents=True)
    unrelated = tmp_path / "custom" / "fastembed_cache"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    candidates = {
        "filesystem_root": Path(Path.cwd().anchor),
        "home": home,
        "project": project,
        "unrelated": unrelated,
    }
    candidate = candidates[candidate_name]

    with pytest.raises(ValueError, match="Refusing to clear unsafe FastEmbed cache"):
        doctor_module._repair_fastembed_cache(candidate, project_root=project)

    if candidate_name == "unrelated":
        assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_doctor_cache_repair_rejects_symlink_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    outside = tmp_path / "outside" / "know-cli" / "fastembed"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    cache_link = tmp_path / "home" / ".cache" / "know-cli" / "fastembed"
    cache_link.parent.mkdir(parents=True)
    cache_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Refusing to clear unsafe FastEmbed cache"):
        doctor_module._repair_fastembed_cache(cache_link, project_root=tmp_path / "project")

    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_doctor_reports_unsafe_configured_cache_without_deleting_it(
    tmp_project, tmp_path, monkeypatch
):
    from know import embeddings
    from know.cli import cli

    custom_cache = tmp_path / "custom" / "fastembed_cache"
    custom_cache.mkdir(parents=True)
    keep = custom_cache / "keep.txt"
    keep.write_text("valuable", encoding="utf-8")

    monkeypatch.setattr(
        doctor_module,
        "_probe_embedding_model",
        lambda _m: (False, "corrupt model"),
    )
    monkeypatch.setattr(doctor_module, "_dependency_integrity", _healthy_dependencies)
    monkeypatch.setattr(doctor_module, "_default_fastembed_cache_root", lambda: custom_cache)
    monkeypatch.setattr(embeddings, "_configure_fastembed_cache_dir", lambda: None)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "--json",
            "doctor",
            "--repair",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["ok"] is False
    assert report["checks"]["fastembed_cache_repair"]["ok"] is False
    assert "Refusing to clear unsafe FastEmbed cache" in report["actions"][0]
    assert keep.read_text(encoding="utf-8") == "valuable"


def test_doctor_repair_does_not_hide_dependency_failure(tmp_project, tmp_path, monkeypatch):
    from know import embeddings
    from know.cli import cli

    probes = iter([(False, "corrupt model"), (True, None)])
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    cache_root = home / ".cache" / "know-cli" / "fastembed"

    monkeypatch.setattr(doctor_module, "_probe_embedding_model", lambda _m: next(probes))
    monkeypatch.setattr(
        doctor_module,
        "_dependency_integrity",
        lambda: _healthy_dependencies(fastembed_installed=False),
    )
    monkeypatch.setattr(doctor_module, "_default_fastembed_cache_root", lambda: cache_root)
    monkeypatch.setattr(embeddings, "_configure_fastembed_cache_dir", lambda: None)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(tmp_project / ".know" / "config.yaml"),
            "--json",
            "doctor",
            "--repair",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["checks"]["embedding_model_after_repair"]["ok"] is True
    assert report["checks"]["dependency_integrity"]["fastembed_installed"] is False
    assert report["ok"] is False
