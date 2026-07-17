"""Adversarial regressions for local AI helpers."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import types
from contextlib import closing

import json
from click.testing import CliRunner

from know.ai import AIResponseCache, AISummarizer, TokenOptimizer
from know.config import Config


def test_optimizer_preserves_operators_and_comment_markers_inside_strings():
    source = '''
def render(n):
    url = "https://example.test/a#fragment"
    marker = "abc#def // still text"
    return n // 2  # actual comment
'''

    compressed = TokenOptimizer.compress_code(source)

    assert 'url = "https://example.test/a#fragment"' in compressed
    assert 'marker = "abc#def // still text"' in compressed
    assert "return n // 2" in compressed
    assert "actual comment" not in compressed


def test_optimizer_honors_nonpositive_and_tiny_character_limits():
    assert TokenOptimizer.compress_code("value = 1", max_chars=0) == ""
    assert len(TokenOptimizer.compress_code("value = 1", max_chars=4)) <= 4


def test_invalid_timeout_environment_cannot_break_module_import(tmp_path):
    env = os.environ.copy()
    env["KNOW_API_TIMEOUT"] = "not-an-integer"
    completed = subprocess.run(
        [sys.executable, "-c", "import know.ai; print('ok')"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_corrupt_cache_is_quarantined_and_rebuilt(tmp_path):
    db_path = tmp_path / "ai_cache.db"
    db_path.write_bytes(b"not a sqlite database")

    cache = AIResponseCache(cache_dir=tmp_path)
    cache.set("content", "explain", "model", "response", 3)

    assert cache.get("content", "explain", "model") == "response"
    assert list(tmp_path.glob("ai_cache.db.corrupt-*"))
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_successful_api_response_survives_cache_write_failure(monkeypatch):
    class BrokenCache:
        def get(self, *args):
            return None

        def set(self, *args):
            raise OSError("disk full")

    class Messages:
        @staticmethod
        def create(**kwargs):
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(output_tokens=2),
                content=[types.SimpleNamespace(text="useful response")],
            )

    class AnthropicClient:
        def __init__(self, **kwargs):
            self.messages = Messages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=AnthropicClient),
    )
    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(Timeout=lambda *args, **kwargs: object()),
    )

    summarizer = AISummarizer.__new__(AISummarizer)
    summarizer.api_key = "key"
    summarizer.model = AISummarizer.MODEL_HAIKU
    summarizer.cache = BrokenCache()

    result = summarizer._call_claude("prompt", cache_key="key")

    assert result == "useful response"


def test_json_explain_reserves_stdout_for_one_json_value(tmp_path, monkeypatch):
    (tmp_path / ".know").mkdir()
    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.save(tmp_path / ".know" / "config.yaml")

    class Scanner:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def find_component(self, _component):
            return [{"name": "target", "type": "function", "content": "def target(): pass"}]

    class Summarizer:
        def __init__(self, _config):
            pass

        def explain_component(self, _component, detailed=False):
            return "explanation"

    monkeypatch.setattr("know.cli.core.CodebaseScanner", Scanner)
    monkeypatch.setattr("know.cli.core.AISummarizer", Summarizer)
    monkeypatch.setattr("know.cli.auto_bootstrap_skill_install", lambda: None)

    from know.cli import cli

    result = CliRunner().invoke(
        cli,
        [
            "--config", str(tmp_path / ".know" / "config.yaml"),
            "--json", "explain", "--component", "target",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "component": "target",
        "explanation": "explanation",
    }


def test_json_init_emits_one_complete_document(tmp_path, monkeypatch):
    class Scanner:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def scan(self):
            return {"files": 1, "functions": 2, "classes": 0, "modules": 1}

    class Generator:
        def __init__(self, _config):
            pass

        def generate_all(self):
            return None

    monkeypatch.setattr("know.cli.core.CodebaseScanner", Scanner)
    monkeypatch.setattr("know.cli.core.DocGenerator", Generator)
    monkeypatch.setattr("know.cli.auto_bootstrap_skill_install", lambda: None)

    from know.cli import cli

    result = CliRunner().invoke(
        cli,
        ["--json", "init", "--path", str(tmp_path)],
    )

    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "status": "initialized",
        "config_path": str(tmp_path / ".know" / "config.yaml"),
        "scan": {"files": 1, "functions": 2, "classes": 0, "modules": 1},
    }
