"""Tests for v2 features: daemon_db, embeddings, new CLI commands, improved parsers."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure."""
    (tmp_path / ".know").mkdir()
    # Create a minimal config
    from know.config import Config
    config = Config.create_default(tmp_path)
    return tmp_path, config


# ---------------------------------------------------------------------------
# DaemonDB tests
# ---------------------------------------------------------------------------
class TestDaemonDB:
    """Tests for the unified daemon database."""

    def test_creates_database(self, tmp_project):
        root, config = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)
        assert (root / ".know" / "daemon.db").exists()
        db.close()

    def test_upsert_and_search_chunks(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        chunks = [
            {
                "name": "authenticate_user",
                "type": "function",
                "start_line": 10,
                "end_line": 25,
                "signature": "def authenticate_user(username, password)",
                "body": "def authenticate_user(username, password):\n    # validate credentials",
            },
            {
                "name": "UserModel",
                "type": "class",
                "start_line": 30,
                "end_line": 60,
                "signature": "class UserModel",
                "body": "class UserModel:\n    # user data model",
            },
        ]
        db.upsert_chunks("auth/service.py", "python", chunks)

        results = db.search_chunks("authenticate")
        assert len(results) >= 1
        assert results[0]["chunk_name"] == "authenticate_user"
        db.close()

    def test_upsert_replaces_old_chunks(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        db.upsert_chunks("file.py", "python", [
            {"name": "old_func", "type": "function", "start_line": 1,
             "end_line": 5, "signature": "old_func()", "body": "old_func"},
        ])
        db.upsert_chunks("file.py", "python", [
            {"name": "new_func", "type": "function", "start_line": 1,
             "end_line": 5, "signature": "new_func()", "body": "new_func"},
        ])

        chunks = db.get_chunks_for_file("file.py")
        assert len(chunks) == 1
        assert chunks[0]["chunk_name"] == "new_func"
        db.close()

    def test_get_signatures(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        db.upsert_chunks("app.py", "python", [
            {"name": "main", "type": "function", "start_line": 1,
             "end_line": 10, "signature": "def main()", "body": "def main(): pass"},
        ])

        sigs = db.get_signatures("app.py")
        assert len(sigs) == 1
        assert sigs[0]["chunk_name"] == "main"

        all_sigs = db.get_signatures()
        assert len(all_sigs) == 1
        db.close()

    def test_memory_store_and_recall(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        stored = db.store_memory("mem1", "JWT authentication uses RS256")
        assert stored is True

        # Duplicate content should return False
        stored2 = db.store_memory("mem2", "JWT authentication uses RS256")
        assert stored2 is False

        memories = db.recall_memories("JWT authentication")
        assert len(memories) >= 1
        assert "RS256" in memories[0]["content"]
        db.close()

    def test_file_index_tracking(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        assert db.get_file_hash("test.py") is None

        db.update_file_index("test.py", "abc123", "python", 3)
        assert db.get_file_hash("test.py") == "abc123"

        db.remove_file("test.py")
        assert db.get_file_hash("test.py") is None
        db.close()

    def test_import_graph(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        db.set_imports("auth.service", [
            ("auth.models", "import"),
            ("utils.crypto", "from"),
        ])

        imports = db.get_imports_of("auth.service")
        assert set(imports) == {"auth.models", "utils.crypto"}

        imported_by = db.get_imported_by("utils.crypto")
        assert "auth.service" in imported_by
        db.close()

    def test_stats(self, tmp_project):
        root, _ = tmp_project
        from know.daemon_db import DaemonDB
        db = DaemonDB(root)

        db.upsert_chunks("a.py", "python", [
            {"name": "f", "type": "function", "start_line": 1,
             "end_line": 5, "signature": "f()", "body": "def f(): pass"},
        ])
        db.store_memory("m1", "some memory content")

        stats = db.get_stats()
        assert stats["chunks"] == 1
        assert stats["files"] == 0  # file_index not updated via upsert_chunks
        assert stats["memories"] == 1
        db.close()


# ---------------------------------------------------------------------------
# Embeddings module tests
# ---------------------------------------------------------------------------
class TestEmbeddings:
    """Tests for centralized embedding management."""

    def test_get_model_returns_none_without_fastembed(self):
        from know.embeddings import _model_cache
        _model_cache.clear()  # reset cache

        with patch.dict("sys.modules", {"fastembed": None}):
            from know import embeddings
            # Clear cache to force re-import attempt
            embeddings._model_cache.clear()
            result = embeddings.get_model()
            assert result is None

    def test_embed_text_returns_none_without_fastembed(self):
        from know import embeddings
        embeddings._model_cache.clear()

        with patch.object(embeddings, "get_model", return_value=None):
            result = embeddings.embed_text("hello world")
            assert result is None

    def test_is_available_returns_false_without_fastembed(self):
        from know import embeddings
        with patch.object(embeddings, "get_model", return_value=None):
            assert embeddings.is_available() is False


# ---------------------------------------------------------------------------
# Token counter tests
# ---------------------------------------------------------------------------
class TestTokenCounter:
    """Tests for tiktoken-based token counting."""

    def test_count_tokens_basic(self):
        from know.token_counter import count_tokens
        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_provider_calibration(self):
        from know.token_counter import count_tokens
        anthropic_count = count_tokens("test string", provider="anthropic")
        openai_count = count_tokens("test string", provider="openai")
        # Anthropic has 1.10x calibration, OpenAI has 1.0x
        assert anthropic_count >= openai_count

    def test_truncate_to_budget(self):
        from know.token_counter import truncate_to_budget, count_tokens
        long_text = "word " * 1000
        result = truncate_to_budget(long_text, budget=100)
        # Allow small overshoot due to calibration rounding
        assert count_tokens(result) <= 110

    def test_empty_string(self):
        from know.token_counter import count_tokens
        assert count_tokens("") == 0


# ---------------------------------------------------------------------------
# Import graph tests
# ---------------------------------------------------------------------------
class TestImportGraph:
    """Tests for FQN-based import resolution."""

    def test_build_import_graph(self, tmp_project):
        root, config = tmp_project

        # Create files with imports
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").touch()
        (pkg / "models.py").write_text("class User: pass\n")
        (pkg / "service.py").write_text("from pkg.models import User\n")

        from know.import_graph import ImportGraph
        ig = ImportGraph(config)
        count = ig.build()
        assert isinstance(count, int)

    def test_imports_of(self, tmp_project):
        root, config = tmp_project

        pkg = root / "myapp"
        pkg.mkdir()
        (pkg / "__init__.py").touch()
        (pkg / "utils.py").write_text("def helper(): pass\n")
        (pkg / "main.py").write_text("from myapp.utils import helper\n")

        from know.import_graph import ImportGraph
        ig = ImportGraph(config)
        ig.build()
        result = ig.imports_of("myapp.main")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Parser tests (multi-language)
# ---------------------------------------------------------------------------
class TestMultiLanguageParsers:
    """Tests for Tree-sitter multi-language parsing."""

    def test_python_parser(self, tmp_path):
        code = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello {name}"

class Greeter:
    """A greeter class."""
    def say_hi(self):
        pass
'''
        f = tmp_path / "test.py"
        f.write_text(code)

        from know.parsers import ParserFactory
        parser = ParserFactory.get_parser_for_file(f)
        assert parser is not None
        mod = parser.parse(f, tmp_path)
        func_names = [fn.name for fn in mod.functions]
        class_names = [c.name for c in mod.classes]
        assert "greet" in func_names
        assert "Greeter" in class_names

    def test_extension_to_language_mapping(self):
        from know.parsers import EXTENSION_TO_LANGUAGE
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".go"] == "go"
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"
        assert EXTENSION_TO_LANGUAGE[".java"] == "java"
        assert EXTENSION_TO_LANGUAGE[".rb"] == "ruby"

    def test_parser_factory_returns_parser(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        from know.parsers import ParserFactory
        parser = ParserFactory.get_parser_for_file(f)
        assert parser is not None

    def test_parser_factory_returns_none_for_unknown(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("unknown content")

        from know.parsers import ParserFactory
        parser = ParserFactory.get_parser_for_file(f)
        assert parser is None

    def test_typescript_parser_available(self, tmp_path):
        code = 'function hello(name: string): string { return name; }\n'
        f = tmp_path / "test.ts"
        f.write_text(code)

        from know.parsers import ParserFactory
        parser = ParserFactory.get_parser_for_file(f)
        assert parser is not None
        mod = parser.parse(f, tmp_path)
        func_names = [fn.name for fn in mod.functions]
        assert "hello" in func_names

    def test_go_parser_available(self, tmp_path):
        code = 'package main\n\nfunc main() {\n}\n'
        f = tmp_path / "test.go"
        f.write_text(code)

        from know.parsers import ParserFactory
        parser = ParserFactory.get_parser_for_file(f)
        assert parser is not None
        mod = parser.parse(f, tmp_path)
        func_names = [fn.name for fn in mod.functions]
        assert "main" in func_names


# ---------------------------------------------------------------------------
# CLI new commands tests
# ---------------------------------------------------------------------------
class TestNewCLICommands:
    """Tests for Phase 5 CLI commands."""

    def test_next_file_command_exists(self):
        from click.testing import CliRunner
        from know.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["next-file", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower() or "QUERY" in result.output

    def test_signatures_command_exists(self):
        from click.testing import CliRunner
        from know.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["signatures", "--help"])
        assert result.exit_code == 0

    def test_related_command_exists(self):
        from click.testing import CliRunner
        from know.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["related", "--help"])
        assert result.exit_code == 0

    def test_generate_context_command_exists(self):
        from click.testing import CliRunner
        from know.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["generate-context", "--help"])
        assert result.exit_code == 0
