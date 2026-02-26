"""Hard-exclude and stale index cleanup tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    # Noise/runtime trees that should never be indexed.
    (tmp_path / ".venv-release-smoke" / "lib" / "python3.14" / "site-packages" / "pkg").mkdir(
        parents=True
    )
    (tmp_path / ".venv-release-smoke" / "lib" / "python3.14" / "site-packages" / "pkg" / "noise.py").write_text(
        "def noise():\n    return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "venv123").mkdir()
    (tmp_path / "venv123" / "noise.py").write_text("def noise2():\n    return 0\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("def generated():\n    return 0\n", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.py").write_text("def bundle():\n    return 0\n", encoding="utf-8")

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.include = []  # broad scan; hard excludes must still apply
    config.exclude = []
    config.save(tmp_path / ".know" / "config.yaml")
    return tmp_path, config


def test_scanner_hard_excludes_runtime_trees(tmp_project):
    root, config = tmp_project
    from know.scanner import CodebaseScanner

    scanner = CodebaseScanner(config)
    stats = scanner.scan()
    assert stats["files"] >= 1

    scanned_paths = [str(m.path).replace("\\", "/") for m in scanner.modules]
    assert "src/main.py" in scanned_paths
    assert not any(".venv-release-smoke/" in p for p in scanned_paths)
    assert not any("/site-packages/" in p for p in scanned_paths)
    assert not any(p.startswith("venv123/") for p in scanned_paths)
    assert not any(p.startswith("build/") for p in scanned_paths)
    assert not any(p.startswith("dist/") for p in scanned_paths)


def test_populate_index_purges_out_of_scope_files(tmp_project):
    root, config = tmp_project
    from know.daemon import populate_index
    from know.daemon_db import DaemonDB

    noise = ".venv-release-smoke/lib/python3.14/site-packages/pkg/noise.py"
    db = DaemonDB(root)
    db.upsert_chunks(
        noise,
        "python",
        [
            {
                "name": "noise",
                "type": "function",
                "start_line": 1,
                "end_line": 2,
                "signature": "def noise()",
                "body": "def noise():\n    return 0\n",
            }
        ],
    )
    db.update_file_index(noise, "legacyhash", "python", 1)
    assert db.get_file_hash(noise) == "legacyhash"

    populate_index(root, config, db)

    assert db.get_file_hash(noise) is None
    assert db.get_file_hash("src/main.py") is not None

