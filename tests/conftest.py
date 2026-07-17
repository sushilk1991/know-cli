"""Test configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def isolate_agent_homes(tmp_path, monkeypatch):
    """Prevent CLI bootstrap tests from mutating real user agent homes."""
    # Keep the fake home outside the project fixture: repair commands
    # intentionally refuse to delete cache paths contained by a project root.
    test_home = tmp_path.parent / f".{tmp_path.name}-home"
    monkeypatch.setenv("HOME", str(test_home))
    monkeypatch.setenv("CODEX_HOME", str(test_home / ".codex"))
    monkeypatch.setenv(
        "FASTEMBED_CACHE_PATH",
        str(test_home / ".cache" / "know-cli" / "fastembed"),
    )
