"""Tests for v3 search ranking: FTS5 migration, BM25F, file categories, ranking, metadata."""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from know.daemon_db import DaemonDB
from know.file_categories import categorize_file, get_demotion, apply_category_demotion
from know.ranking import fuse_rankings, apply_relevance_floor


# ---------------------------------------------------------------------------
# File categories tests
# ---------------------------------------------------------------------------

class TestFileCategories:
    def test_source_file(self):
        assert categorize_file("src/auth/session.py") == "source"
        assert categorize_file("app/models/user.py") == "source"

    def test_test_file(self):
        assert categorize_file("test_session.py") == "test"
        assert categorize_file("tests/test_auth.py") == "test"
        assert categorize_file("tests/unit/test_session.py") == "test"
        assert categorize_file("spec/auth_spec.py") == "test"

    def test_vendor_file(self):
        assert categorize_file("vendor/lib/utils.py") == "vendor"
        assert categorize_file("node_modules/pkg/index.js") == "vendor"
        assert categorize_file(".venv/lib/python3.14/site.py") == "vendor"

    def test_generated_file(self):
        assert categorize_file("proto/service_pb2.py") == "generated"
        assert categorize_file("src/schema_generated.py") == "generated"

    def test_demotion_values(self):
        assert get_demotion("src/main.py") == 1.0
        assert get_demotion("tests/test_main.py") == 0.3
        assert get_demotion("vendor/lib.py") == 0.1

    def test_apply_demotion(self):
        chunks = [
            {"file_path": "src/main.py", "score": 1.0},
            {"file_path": "tests/test_main.py", "score": 1.0},
            {"file_path": "vendor/lib.py", "score": 1.0},
        ]
        apply_category_demotion(chunks)
        assert chunks[0]["score"] == 1.0
        assert chunks[1]["score"] == 0.3
        assert chunks[2]["score"] == pytest.approx(0.1)

    def test_skip_test_demotion_when_query_has_test(self):
        chunks = [
            {"file_path": "tests/test_auth.py", "score": 1.0},
            {"file_path": "vendor/lib.py", "score": 1.0},
        ]
        apply_category_demotion(chunks, query="test authentication")
        assert chunks[0]["score"] == 1.0  # Not demoted
        assert chunks[1]["score"] == pytest.approx(0.1)  # Still demoted


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

class TestRanking:
    def test_fuse_single_list(self):
        ranked = [("a", 1.0), ("b", 0.5), ("c", 0.2)]
        result = fuse_rankings([ranked])
        assert result[0][0] == "a"
        assert result[1][0] == "b"
        assert result[2][0] == "c"

    def test_fuse_multiple_lists(self):
        list1 = [("a", 1.0), ("b", 0.5)]
        list2 = [("b", 1.0), ("c", 0.5)]
        result = fuse_rankings([list1, list2])
        # b appears in both lists, should rank highest
        ids = [r[0] for r in result]
        assert ids[0] == "b"

    def test_fuse_empty_lists(self):
        assert fuse_rankings([]) == []
        assert fuse_rankings([[]]) == []

    def test_relevance_floor(self):
        chunks = [
            {"score": 1.0, "body": "high"},
            {"score": 0.5, "body": "medium"},
            {"score": 0.2, "body": "low"},
            {"score": 0.1, "body": "very low"},
        ]
        result = apply_relevance_floor(chunks, top_score_ratio=0.3)
        # Floor = 1.0 * 0.3 = 0.3. Only chunks with score >= 0.3 kept.
        assert len(result) == 2
        assert result[0]["score"] == 1.0
        assert result[1]["score"] == 0.5

    def test_relevance_floor_empty(self):
        assert apply_relevance_floor([]) == []

    def test_relevance_floor_all_pass(self):
        chunks = [{"score": 1.0}, {"score": 0.8}, {"score": 0.5}]
        result = apply_relevance_floor(chunks, top_score_ratio=0.3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# DaemonDB FTS5 migration tests
# ---------------------------------------------------------------------------

class TestFTS5Migration:
    def test_new_db_has_4_column_fts(self):
        """Fresh database should create 4-column FTS5 table."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            conn = db._get_conn()
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            ).fetchone()
            assert row is not None
            assert "file_path" in row[0]
            assert "chunk_name" in row[0]
            assert "signature" in row[0]
            assert "body" in row[0]
            db.close()

    def test_fts5vocab_table_exists(self):
        """fts5vocab table should be created alongside FTS5."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            conn = db._get_conn()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE name='chunks_fts_vocab'"
            ).fetchone()
            assert row is not None
            db.close()

    def test_schema_version_table_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            conn = db._get_conn()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE name='schema_version'"
            ).fetchone()
            assert row is not None
            db.close()

    def test_module_importance_table_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            conn = db._get_conn()
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE name='module_importance'"
            ).fetchone()
            assert row is not None
            db.close()

    def test_bm25f_search_uses_weights(self):
        """BM25F search should use field weights on new schema."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            # Insert a chunk
            db.upsert_chunks("src/auth.py", "python", [
                {
                    "name": "verify_session",
                    "type": "function",
                    "start_line": 1,
                    "end_line": 10,
                    "signature": "def verify_session(token: str):",
                    "body": "def verify_session(token: str):\n    return True",
                },
            ])
            results = db.search_chunks("verify_session")
            assert len(results) > 0
            assert results[0]["chunk_name"] == "verify_session"
            # Should have a score field
            assert "score" in results[0]
            db.close()

    def test_migration_detects_old_schema(self):
        """_needs_fts_migration should detect 3-column FTS tables."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".know").mkdir(parents=True)
            db_path = root / ".know" / "daemon.db"
            # Create old-style DB manually
            conn = sqlite3.connect(str(db_path))
            conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT, chunk_name TEXT, chunk_type TEXT,
                language TEXT, start_line INTEGER, end_line INTEGER,
                signature TEXT DEFAULT '', body TEXT, body_hash TEXT,
                token_count INTEGER, updated_at REAL
            )""")
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_name, signature, body,
                content='chunks', content_rowid='id'
            )""")
            conn.commit()
            conn.close()

            # Now open with DaemonDB — should migrate
            db = DaemonDB(root)
            conn = db._get_conn()
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            ).fetchone()
            assert "file_path" in row[0]
            db.close()


# ---------------------------------------------------------------------------
# Batch import tests
# ---------------------------------------------------------------------------

class TestBatchImports:
    def test_get_imports_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.set_imports("src.auth", [("src.models", "import"), ("src.utils", "from")])
            db.set_imports("src.api", [("src.auth", "import")])

            result = db.get_imports_batch(["src.auth", "src.api"])
            assert "src.models" in result["src.auth"]
            assert "src.utils" in result["src.auth"]
            assert "src.auth" in result["src.api"]
            db.close()

    def test_get_imported_by_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.set_imports("src.auth", [("src.models", "import")])
            db.set_imports("src.api", [("src.models", "import")])

            result = db.get_imported_by_batch(["src.models"])
            assert "src.auth" in result["src.models"]
            assert "src.api" in result["src.models"]
            db.close()

    def test_empty_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            assert db.get_imports_batch([]) == {}
            assert db.get_imported_by_batch([]) == {}
            db.close()


# ---------------------------------------------------------------------------
# Module importance tests
# ---------------------------------------------------------------------------

class TestModuleImportance:
    def test_compute_importance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.set_imports("src.a", [("src.core", "import")])
            db.set_imports("src.b", [("src.core", "import")])
            db.set_imports("src.c", [("src.core", "import"), ("src.utils", "import")])

            scores = db.compute_importance()
            # src.core imported by 3 modules, src.utils by 1
            assert scores["src.core"] == 1.0  # highest in-degree
            assert scores["src.utils"] == pytest.approx(1/3, abs=0.01)
            db.close()

    def test_get_importance_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.set_imports("src.a", [("src.core", "import")])
            db.set_imports("src.b", [("src.core", "import")])
            db.compute_importance()

            batch = db.get_importance_batch(["src.core", "src.missing"])
            assert batch["src.core"] == 1.0
            assert "src.missing" not in batch
            db.close()


# ---------------------------------------------------------------------------
# Zero-result intelligence tests
# ---------------------------------------------------------------------------

class TestZeroResultIntelligence:
    def test_nearest_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.upsert_chunks("src/auth.py", "python", [
                {"name": "authenticate", "type": "function",
                 "start_line": 1, "end_line": 5,
                 "signature": "def authenticate():", "body": "def authenticate(): pass"},
                {"name": "authorize", "type": "function",
                 "start_line": 6, "end_line": 10,
                 "signature": "def authorize():", "body": "def authorize(): pass"},
            ])
            terms = db.get_nearest_terms("auth")
            assert len(terms) > 0
            # Should find terms starting with "aut"
            assert any("auth" in t for t in terms)
            db.close()

    def test_matching_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = DaemonDB(root)
            db.upsert_chunks("src/auth/handler.py", "python", [
                {"name": "handle", "type": "function",
                 "start_line": 1, "end_line": 5,
                 "signature": "def handle():", "body": "def handle(): pass"},
            ])
            db.upsert_chunks("src/api/routes.py", "python", [
                {"name": "route", "type": "function",
                 "start_line": 1, "end_line": 5,
                 "signature": "def route():", "body": "def route(): pass"},
            ])
            files = db.get_matching_file_names("auth handler")
            assert "src/auth/handler.py" in files
            db.close()
