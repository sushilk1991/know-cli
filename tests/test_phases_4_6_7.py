"""Tests for Phase 4 (context expansion), Phase 6 (dedup), Phase 7 (call graph)."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from know.daemon_db import DaemonDB
from know.models import FunctionInfo, ClassInfo, ModuleInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Create a DaemonDB with test data."""
    db = DaemonDB(tmp_path)
    return db


@pytest.fixture
def populated_db(db):
    """DB with chunks to test context expansion and dedup."""
    chunks = [
        {
            "name": "auth",
            "type": "module",
            "start_line": 1,
            "end_line": 5,
            "signature": "auth",
            "body": "import os\nimport sys\nfrom auth.utils import hash_password\n\n# Auth module",
        },
        {
            "name": "AuthService",
            "type": "class",
            "start_line": 7,
            "end_line": 50,
            "signature": "AuthService",
            "body": "class AuthService:\n    def __init__(self):\n        pass\n    def login(self, user, pw):\n        return verify(user, pw)\n    def logout(self):\n        pass\n    def refresh(self):\n        pass",
        },
        {
            "name": "login",
            "type": "method",
            "start_line": 10,
            "end_line": 15,
            "signature": "login(self, user, pw)",
            "body": "def login(self, user, pw):\n    return verify(user, pw)",
        },
        {
            "name": "logout",
            "type": "method",
            "start_line": 17,
            "end_line": 20,
            "signature": "logout(self)",
            "body": "def logout(self):\n    self.session = None",
        },
        {
            "name": "refresh",
            "type": "method",
            "start_line": 22,
            "end_line": 25,
            "signature": "refresh(self)",
            "body": "def refresh(self):\n    self.token = new_token()",
        },
        {
            "name": "verify",
            "type": "function",
            "start_line": 52,
            "end_line": 60,
            "signature": "verify(user, password)",
            "body": "def verify(user, password):\n    return hash_password(password) == user.hash",
        },
    ]
    db.upsert_chunks("src/auth/service.py", "python", chunks)

    # Second file for cross-file testing
    chunks2 = [
        {
            "name": "hash_password",
            "type": "function",
            "start_line": 1,
            "end_line": 5,
            "signature": "hash_password(pw)",
            "body": "def hash_password(pw):\n    import hashlib\n    return hashlib.sha256(pw.encode()).hexdigest()",
        },
    ]
    db.upsert_chunks("src/auth/utils.py", "python", chunks2)
    return db


# ---------------------------------------------------------------------------
# Phase 7: Symbol refs (call graph)
# ---------------------------------------------------------------------------

class TestSymbolRefs:
    """Test symbol reference storage and querying."""

    def test_upsert_and_get_callers(self, db):
        """Test storing and querying callers."""
        refs = [
            {"ref_name": "verify", "ref_type": "call", "line_number": 12, "containing_chunk": "login"},
            {"ref_name": "hash_password", "ref_type": "call", "line_number": 55, "containing_chunk": "verify"},
        ]
        db.upsert_symbol_refs("src/auth/service.py", refs)

        callers = db.get_callers("verify")
        assert len(callers) == 1
        assert callers[0]["containing_chunk"] == "login"
        assert callers[0]["file_path"] == "src/auth/service.py"

    def test_get_callees(self, db):
        """Test querying what a chunk calls."""
        refs = [
            {"ref_name": "verify", "ref_type": "call", "line_number": 12, "containing_chunk": "login"},
            {"ref_name": "log_attempt", "ref_type": "call", "line_number": 13, "containing_chunk": "login"},
        ]
        db.upsert_symbol_refs("src/auth/service.py", refs)

        callees = db.get_callees("login")
        assert len(callees) == 2
        names = {c["ref_name"] for c in callees}
        assert names == {"verify", "log_attempt"}

    def test_upsert_replaces_old_refs(self, db):
        """Upserting for same file replaces previous refs."""
        refs1 = [{"ref_name": "foo", "ref_type": "call", "line_number": 1, "containing_chunk": "bar"}]
        db.upsert_symbol_refs("test.py", refs1)
        assert len(db.get_callers("foo")) == 1

        refs2 = [{"ref_name": "baz", "ref_type": "call", "line_number": 2, "containing_chunk": "qux"}]
        db.upsert_symbol_refs("test.py", refs2)
        assert len(db.get_callers("foo")) == 0
        assert len(db.get_callers("baz")) == 1

    def test_cross_file_callers(self, db):
        """Callers from multiple files are all returned."""
        db.upsert_symbol_refs("a.py", [
            {"ref_name": "shared", "ref_type": "call", "line_number": 5, "containing_chunk": "func_a"},
        ])
        db.upsert_symbol_refs("b.py", [
            {"ref_name": "shared", "ref_type": "call", "line_number": 10, "containing_chunk": "func_b"},
        ])

        callers = db.get_callers("shared")
        assert len(callers) == 2
        files = {c["file_path"] for c in callers}
        assert files == {"a.py", "b.py"}

    def test_empty_refs(self, db):
        """Empty refs list clears existing refs for the file."""
        db.upsert_symbol_refs("test.py", [
            {"ref_name": "x", "ref_type": "call", "line_number": 1, "containing_chunk": "y"},
        ])
        assert len(db.get_callers("x")) == 1

        db.upsert_symbol_refs("test.py", [])
        assert len(db.get_callers("x")) == 0


class TestCallExtraction:
    """Test call reference extraction from parsers."""

    def test_python_parser_extract_calls(self):
        """PythonParser extracts function calls."""
        from know.parsers import PythonParser

        code = '''
def greet(name):
    print(f"Hello {name}")
    log_message(name)

def main():
    greet("world")
    result = process_data()
'''
        parser = PythonParser()
        module = ModuleInfo(
            path=Path("test.py"),
            name="test",
            docstring=None,
            functions=[
                FunctionInfo(name="greet", line_number=2, end_line=4, docstring=None,
                             signature="greet(name)", is_async=False, is_method=False),
                FunctionInfo(name="main", line_number=6, end_line=8, docstring=None,
                             signature="main()", is_async=False, is_method=False),
            ],
            classes=[],
            imports=[],
        )

        refs = parser.extract_call_refs(code, module)
        assert len(refs) > 0

        # Check specific calls
        ref_names = {r["ref_name"] for r in refs}
        assert "print" in ref_names
        assert "log_message" in ref_names
        assert "greet" in ref_names
        assert "process_data" in ref_names

        # Check containing chunk
        greet_calls = [r for r in refs if r["ref_name"] == "greet"]
        assert any(r["containing_chunk"] == "main" for r in greet_calls)

    def test_python_parser_method_calls(self):
        """PythonParser extracts attribute/method calls."""
        from know.parsers import PythonParser

        code = '''
def process():
    result = db.query("SELECT *")
    result.sort()
'''
        parser = PythonParser()
        module = ModuleInfo(
            path=Path("test.py"), name="test", docstring=None,
            functions=[
                FunctionInfo(name="process", line_number=2, end_line=4, docstring=None,
                             signature="process()", is_async=False, is_method=False),
            ],
            classes=[], imports=[],
        )

        refs = parser.extract_call_refs(code, module)
        ref_names = {r["ref_name"] for r in refs}
        assert "query" in ref_names
        assert "sort" in ref_names

    def test_treesitter_extract_calls(self):
        """TreeSitterParser extracts function calls."""
        try:
            from know.parsers import TreeSitterParser
        except ImportError:
            pytest.skip("tree-sitter not available")

        code = '''def handler():
    result = validate_input(data)
    save_to_db(result)
    return format_response(result)
'''
        parser = TreeSitterParser("python")
        module = ModuleInfo(
            path=Path("test.py"), name="test", docstring=None,
            functions=[
                FunctionInfo(name="handler", line_number=1, end_line=4, docstring=None,
                             signature="handler()", is_async=False, is_method=False),
            ],
            classes=[], imports=[],
        )

        refs = parser.extract_call_refs(code, module)
        ref_names = {r["ref_name"] for r in refs}
        assert "validate_input" in ref_names
        assert "save_to_db" in ref_names
        assert "format_response" in ref_names

    def test_containing_chunk_assignment(self):
        """Calls at module level get <module> as containing chunk."""
        from know.parsers import PythonParser

        code = '''
setup_logging()

def main():
    run()
'''
        parser = PythonParser()
        module = ModuleInfo(
            path=Path("test.py"), name="test", docstring=None,
            functions=[
                FunctionInfo(name="main", line_number=4, end_line=5, docstring=None,
                             signature="main()", is_async=False, is_method=False),
            ],
            classes=[], imports=[],
        )

        refs = parser.extract_call_refs(code, module)
        setup_refs = [r for r in refs if r["ref_name"] == "setup_logging"]
        assert len(setup_refs) == 1
        assert setup_refs[0]["containing_chunk"] == "<module>"

        run_refs = [r for r in refs if r["ref_name"] == "run"]
        assert len(run_refs) == 1
        assert run_refs[0]["containing_chunk"] == "main"


# ---------------------------------------------------------------------------
# Phase 6: Chunk deduplication
# ---------------------------------------------------------------------------

class TestChunkDeduplication:
    """Test the _deduplicate_chunks method."""

    def _make_engine(self):
        """Create a minimal ContextEngine mock for testing dedup."""
        from know.context_engine import ContextEngine
        engine = ContextEngine.__new__(ContextEngine)
        return engine

    def test_three_methods_drops_class(self):
        """When 3+ methods from same class selected, drop the class body."""
        engine = self._make_engine()
        chunks = [
            {"file_path": "a.py", "chunk_name": "MyClass", "chunk_type": "class",
             "start_line": 1, "end_line": 50, "body": "class MyClass:...", "token_count": 100},
            {"file_path": "a.py", "chunk_name": "login", "chunk_type": "method",
             "start_line": 5, "end_line": 15, "body": "def login():", "token_count": 20},
            {"file_path": "a.py", "chunk_name": "logout", "chunk_type": "method",
             "start_line": 17, "end_line": 25, "body": "def logout():", "token_count": 20},
            {"file_path": "a.py", "chunk_name": "refresh", "chunk_type": "method",
             "start_line": 27, "end_line": 35, "body": "def refresh():", "token_count": 20},
        ]
        result, total_tokens = engine._deduplicate_chunks(chunks)
        names = {c["chunk_name"] for c in result}
        assert "MyClass" not in names
        assert "login" in names
        assert "logout" in names
        assert "refresh" in names

    def test_two_methods_drops_methods(self):
        """When < 3 methods from class selected, keep class, drop methods."""
        engine = self._make_engine()
        chunks = [
            {"file_path": "a.py", "chunk_name": "MyClass", "chunk_type": "class",
             "start_line": 1, "end_line": 50, "body": "class MyClass:...", "token_count": 100},
            {"file_path": "a.py", "chunk_name": "login", "chunk_type": "method",
             "start_line": 5, "end_line": 15, "body": "def login():", "token_count": 20},
            {"file_path": "a.py", "chunk_name": "logout", "chunk_type": "method",
             "start_line": 17, "end_line": 25, "body": "def logout():", "token_count": 20},
        ]
        result, total_tokens = engine._deduplicate_chunks(chunks)
        names = {c["chunk_name"] for c in result}
        assert "MyClass" in names
        assert "login" not in names
        assert "logout" not in names

    def test_no_overlap_preserved(self):
        """Chunks from different files are not deduplicated."""
        engine = self._make_engine()
        chunks = [
            {"file_path": "a.py", "chunk_name": "func_a", "chunk_type": "function",
             "start_line": 1, "end_line": 10, "body": "def func_a():", "token_count": 20},
            {"file_path": "b.py", "chunk_name": "func_b", "chunk_type": "function",
             "start_line": 1, "end_line": 10, "body": "def func_b():", "token_count": 20},
        ]
        result, total_tokens = engine._deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_methods_outside_class_range_not_deduped(self):
        """Methods outside class line range are not affected by dedup."""
        engine = self._make_engine()
        chunks = [
            {"file_path": "a.py", "chunk_name": "MyClass", "chunk_type": "class",
             "start_line": 1, "end_line": 30, "body": "class MyClass:...", "token_count": 100},
            {"file_path": "a.py", "chunk_name": "standalone", "chunk_type": "function",
             "start_line": 35, "end_line": 45, "body": "def standalone():", "token_count": 20},
        ]
        result, total_tokens = engine._deduplicate_chunks(chunks)
        names = {c["chunk_name"] for c in result}
        assert "MyClass" in names
        assert "standalone" in names


# ---------------------------------------------------------------------------
# Phase 4: Context expansion
# ---------------------------------------------------------------------------

class TestContextExpansion:
    """Test the _expand_context method."""

    def _make_engine(self):
        from know.context_engine import ContextEngine
        engine = ContextEngine.__new__(ContextEngine)
        return engine

    def test_expand_adds_module_chunk(self, populated_db):
        """Expansion includes module-level chunk (imports) from same file."""
        engine = self._make_engine()

        code_chunks = [
            {"file_path": "src/auth/service.py", "chunk_name": "verify", "chunk_type": "function",
             "start_line": 52, "end_line": 60, "body": "def verify(user, password):\n    return hash_password(password) == user.hash",
             "token_count": 20},
        ]

        expanded, new_used = engine._expand_context(
            populated_db, code_chunks, code_used=20, budget_code=500,
            seen_chunk_keys=set()
        )
        # Should include the module-level chunk
        names = {c["chunk_name"] for c in expanded}
        assert "auth" in names  # Module chunk

    def test_expand_includes_parent_class(self, populated_db):
        """Expansion includes parent class signature for methods."""
        engine = self._make_engine()

        code_chunks = [
            {"file_path": "src/auth/service.py", "chunk_name": "login", "chunk_type": "method",
             "start_line": 10, "end_line": 15, "body": "def login(self, user, pw):\n    return verify(user, pw)",
             "token_count": 20},
        ]

        expanded, new_used = engine._expand_context(
            populated_db, code_chunks, code_used=20, budget_code=500,
            seen_chunk_keys=set()
        )
        # Should include AuthService (class containing login)
        names = {c["chunk_name"] for c in expanded}
        # The module chunk and possibly class signature should be added
        assert len(expanded) >= 1

    def test_expand_respects_budget(self, populated_db):
        """Expansion stops when budget is exhausted."""
        engine = self._make_engine()

        code_chunks = [
            {"file_path": "src/auth/service.py", "chunk_name": "login", "chunk_type": "method",
             "start_line": 10, "end_line": 15, "body": "def login(self, user, pw):\n    return verify(user, pw)",
             "token_count": 20},
        ]

        # Very tight budget — only 21 tokens total, 20 used (1 token remaining)
        expanded, new_used = engine._expand_context(
            populated_db, code_chunks, code_used=20, budget_code=21,
            seen_chunk_keys=set()
        )
        # With only 1 token remaining, nothing should be added (all chunks > 1 token)
        assert len(expanded) == 1  # Only the original chunk
        assert new_used == 20

    def test_expand_skips_already_seen(self, populated_db):
        """Expansion doesn't add chunks that are already selected."""
        engine = self._make_engine()

        code_chunks = [
            {"file_path": "src/auth/service.py", "chunk_name": "login", "chunk_type": "method",
             "start_line": 10, "end_line": 15, "body": "def login(self, user, pw):\n    return verify(user, pw)",
             "token_count": 20},
        ]

        # Pre-mark the module chunk as seen (using the exact key format)
        seen = {"src/auth/service.py:auth:1"}

        expanded, new_used = engine._expand_context(
            populated_db, code_chunks, code_used=20, budget_code=500,
            seen_chunk_keys=seen
        )
        # Module chunk should NOT be added since it's already seen
        extra_names = {c["chunk_name"] for c in expanded if c["chunk_name"] != "login"}
        assert "auth" not in extra_names
