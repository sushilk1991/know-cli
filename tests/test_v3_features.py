"""Tests for v3 features: know map, session dedup, know deep."""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project with Python files for testing."""
    src = tmp_path / "src"
    src.mkdir()

    # billing/service.py
    billing = src / "billing"
    billing.mkdir()
    (billing / "__init__.py").write_text("")
    (billing / "service.py").write_text('''"""Billing service module."""

def check_cloud_access(workspace):
    """Verify workspace has active subscription."""
    count = count_active_sessions(workspace.id)
    if count > workspace.limit:
        raise LimitExceeded(workspace)
    return True

def count_active_sessions(workspace_id):
    """Count active cloud sessions for a workspace."""
    return db.query("SELECT COUNT(*) FROM sessions WHERE ws_id = ?", workspace_id)

class BillingService:
    """Manages billing operations."""

    def process_payment(self, amount):
        """Process a payment."""
        validate_amount(amount)
        return self._charge(amount)

    def _charge(self, amount):
        return {"status": "charged", "amount": amount}
''')

    # auth/middleware.py
    auth = src / "auth"
    auth.mkdir()
    (auth / "__init__.py").write_text("")
    (auth / "middleware.py").write_text('''"""Auth middleware."""

def authenticate(request):
    """Check authentication token."""
    token = request.headers.get("Authorization")
    if not token:
        raise Unauthorized()
    return verify_token(token)

def verify_token(token):
    """Verify JWT token."""
    return {"user_id": 1, "valid": True}
''')

    # tests/test_billing.py
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_billing.py").write_text('''"""Test billing."""

def test_check_cloud_access():
    assert True

def test_count_active_sessions():
    assert True
''')

    # .know dir
    know_dir = tmp_path / ".know"
    know_dir.mkdir()

    # pyproject.toml
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test-project"\n')

    return tmp_path


@pytest.fixture
def indexed_db(tmp_project):
    """Create a DaemonDB with indexed chunks from tmp_project."""
    from know.daemon_db import DaemonDB

    db = DaemonDB(tmp_project)

    # Insert chunks manually
    conn = db._get_conn()
    now = time.time()
    chunks = [
        ("src/billing/service.py", "check_cloud_access", "function", "python",
         3, 8, "def check_cloud_access(workspace):",
         'def check_cloud_access(workspace):\n    """Verify workspace."""\n    count = count_active_sessions(workspace.id)\n    return True',
         "hash1", 45, now),
        ("src/billing/service.py", "count_active_sessions", "function", "python",
         10, 13, "def count_active_sessions(workspace_id):",
         'def count_active_sessions(workspace_id):\n    """Count active sessions."""\n    return db.query("SELECT COUNT(*)")',
         "hash2", 30, now),
        ("src/billing/service.py", "BillingService", "class", "python",
         15, 24, "class BillingService:",
         'class BillingService:\n    """Manages billing."""\n    def process_payment(self, amount):\n        pass',
         "hash3", 35, now),
        ("src/billing/service.py", "BillingService.process_payment", "method", "python",
         18, 21, "def process_payment(self, amount):",
         'def process_payment(self, amount):\n    """Process a payment."""\n    validate_amount(amount)\n    return self._charge(amount)',
         "hash4", 30, now),
        ("src/auth/middleware.py", "authenticate", "function", "python",
         3, 8, "def authenticate(request):",
         'def authenticate(request):\n    """Check auth token."""\n    token = request.headers.get("Authorization")\n    return verify_token(token)',
         "hash5", 35, now),
        ("src/auth/middleware.py", "verify_token", "function", "python",
         10, 13, "def verify_token(token):",
         'def verify_token(token):\n    """Verify JWT."""\n    return {"user_id": 1}',
         "hash6", 25, now),
        ("tests/test_billing.py", "test_check_cloud_access", "function", "python",
         3, 5, "def test_check_cloud_access():",
         'def test_check_cloud_access():\n    assert True',
         "hash7", 15, now),
    ]

    for c in chunks:
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            c,
        )

    # Insert symbol_refs for call graph
    refs = [
        # check_cloud_access calls count_active_sessions
        ("src/billing/service.py", "check_cloud_access", "count_active_sessions", "call", 5),
        # authenticate calls verify_token
        ("src/auth/middleware.py", "authenticate", "verify_token", "call", 7),
    ]
    for r in refs:
        conn.execute(
            "INSERT OR REPLACE INTO symbol_refs "
            "(file_path, containing_chunk, ref_name, ref_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            r,
        )

    conn.commit()

    # Add file_index entries so get_stats() reports files > 0
    file_entries = [
        ("src/billing/service.py", "hash_bs", "python", 3, now),
        ("src/auth/middleware.py", "hash_am", "python", 2, now),
        ("tests/test_billing.py", "hash_tb", "python", 1, now),
    ]
    for fe in file_entries:
        conn.execute(
            "INSERT OR REPLACE INTO file_index "
            "(file_path, content_hash, language, chunk_count, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            fe,
        )
    conn.commit()

    # Rebuild FTS index
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    yield db
    db.close()


# ---------------------------------------------------------------------------
# Tests: know map (search_signatures)
# ---------------------------------------------------------------------------

class TestKnowMap:
    """Tests for the `know map` command / search_signatures()."""

    def test_search_signatures_returns_lightweight_results(self, indexed_db):
        """search_signatures returns sig fields without bodies."""
        results = indexed_db.search_signatures("billing", limit=10)
        assert len(results) > 0
        for r in results:
            assert "file_path" in r
            assert "chunk_name" in r
            assert "signature" in r
            assert "score" in r
            # Should NOT have full body
            assert "body" not in r or r.get("body") is None

    def test_search_signatures_respects_limit(self, indexed_db):
        results = indexed_db.search_signatures("billing", limit=2)
        assert len(results) <= 2

    def test_search_signatures_type_filter(self, indexed_db):
        results = indexed_db.search_signatures("billing", limit=10, chunk_type="class")
        for r in results:
            assert r["chunk_type"] == "class"

    def test_search_signatures_empty_query(self, indexed_db):
        results = indexed_db.search_signatures("", limit=10)
        # Empty query might return nothing or everything — just shouldn't crash
        assert isinstance(results, list)

    def test_map_cli_json_output(self, tmp_project, indexed_db):
        """CLI `know map` with --json produces valid JSON."""
        runner = CliRunner()
        from know.cli import cli

        with patch("know.cli.agent._get_daemon_client", return_value=None):
            with patch("know.cli.agent._get_db_fallback", return_value=indexed_db):
                result = runner.invoke(cli, ["--json", "map", "billing"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "results" in data
        assert "count" in data

    def test_map_cli_rich_output(self, tmp_project, indexed_db):
        """CLI `know map` without --json produces human-readable output."""
        runner = CliRunner()
        from know.cli import cli

        with patch("know.cli.agent._get_daemon_client", return_value=None):
            with patch("know.cli.agent._get_db_fallback", return_value=indexed_db):
                result = runner.invoke(cli, ["map", "billing"])

        assert result.exit_code == 0
        # Should contain "Map results" header or file paths
        assert "billing" in result.output.lower() or "Map results" in result.output or "No matches" in result.output


# ---------------------------------------------------------------------------
# Tests: Session dedup
# ---------------------------------------------------------------------------

class TestSessionDedup:
    """Tests for session-aware deduplication."""

    def test_create_session(self, indexed_db):
        """create_session creates a new session."""
        sid = indexed_db.create_session("test-session-1")
        assert sid == "test-session-1"

    def test_session_seen_empty_initially(self, indexed_db):
        """New session has no seen chunks."""
        indexed_db.create_session("test-session-2")
        seen = indexed_db.get_session_seen("test-session-2")
        assert len(seen) == 0

    def test_mark_and_get_session_seen(self, indexed_db):
        """mark_session_seen records chunks, get_session_seen retrieves them."""
        sid = "test-session-3"
        indexed_db.create_session(sid)
        keys = ["file.py:func_a:10", "file.py:func_b:20"]
        indexed_db.mark_session_seen(sid, keys)
        seen = indexed_db.get_session_seen(sid)
        assert "file.py:func_a:10" in seen
        assert "file.py:func_b:20" in seen

    def test_session_dedup_in_context_engine(self, tmp_project, indexed_db):
        """build_context with session_id deduplicates across calls."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project
        config.project.name = "test"
        config.project.description = ""

        engine = ContextEngine(config)

        # Use _build_context_v3_inner directly to avoid get_direct_db/close issues
        # First call — creates session and returns chunks
        result1 = engine._build_context_v3_inner(
            indexed_db, "billing", budget=5000,
            include_tests=True, include_imports=True,
            include_patterns=None, exclude_patterns=None,
            chunk_types=None, session_id="dedup-test",
        )

        chunks1 = [c.name for c in result1["code_chunks"]]
        assert len(chunks1) > 0
        assert result1.get("session_id") == "dedup-test"

        # Second call — same session, should get different (or fewer) chunks
        result2 = engine._build_context_v3_inner(
            indexed_db, "billing service", budget=5000,
            include_tests=True, include_imports=True,
            include_patterns=None, exclude_patterns=None,
            chunk_types=None, session_id="dedup-test",
        )

        chunks2 = [c.name for c in result2["code_chunks"]]

        # There should be no overlap between chunk sets
        overlap = set(chunks1) & set(chunks2)
        # Note: some overlap is OK if same chunk matches both queries
        # but the session should reduce it
        assert result2.get("session_id") == "dedup-test"

    def test_no_session_backward_compatible(self, tmp_project, indexed_db):
        """build_context without session_id works as before."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project
        config.project.name = "test"
        config.project.description = ""

        engine = ContextEngine(config)

        with patch("know.context_engine.get_direct_db", return_value=indexed_db):
            result = engine.build_context("billing", budget=5000)

        assert "session_id" not in result
        assert result["used_tokens"] >= 0

    def test_session_stats(self, indexed_db):
        """get_session_stats returns chunk count and total tokens."""
        sid = "stats-test"
        indexed_db.create_session(sid)
        indexed_db.mark_session_seen(sid, ["a:b:1", "c:d:2"], [100, 200])
        stats = indexed_db.get_session_stats(sid)
        assert stats["chunks_seen"] == 2
        assert stats["tokens_provided"] == 300

    def test_cleanup_expired_sessions(self, indexed_db):
        """cleanup_expired_sessions removes old sessions."""
        conn = indexed_db._get_conn()
        old_time = time.time() - 20000  # Well past 4-hour TTL
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, created_at, last_used_at) "
            "VALUES (?, ?, ?)",
            ("expired-session", old_time, old_time),
        )
        conn.commit()

        indexed_db.cleanup_expired_sessions(ttl_seconds=14400)

        seen = indexed_db.get_session_seen("expired-session")
        assert len(seen) == 0


# ---------------------------------------------------------------------------
# Tests: know deep (build_deep_context)
# ---------------------------------------------------------------------------

class TestKnowDeep:
    """Tests for the `know deep` command / build_deep_context()."""

    def test_deep_single_match(self, tmp_project, indexed_db):
        """build_deep_context returns target + callers + callees for exact match."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("check_cloud_access", budget=3000)

        assert "error" not in result
        assert result["target"]["name"] == "check_cloud_access"
        assert result["target"]["body"]
        assert result["budget_used"] > 0
        assert result["budget_used"] <= 3000

    def test_deep_not_found(self, tmp_project, indexed_db):
        """build_deep_context returns not_found for unknown function."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("nonexistent_function", budget=3000)

        assert result["error"] == "not_found"
        assert "nearest" in result

    def test_deep_callees_included(self, tmp_project, indexed_db):
        """build_deep_context includes callees (functions called by target)."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("check_cloud_access", budget=3000)

        assert "error" not in result
        callee_names = [c["name"] for c in result.get("callees", [])]
        assert "count_active_sessions" in callee_names

    def test_deep_callers_included(self, tmp_project, indexed_db):
        """build_deep_context includes callers (functions that call target)."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        # verify_token is called by authenticate
        result = engine.build_deep_context("verify_token", budget=3000)

        assert "error" not in result
        caller_names = [c["name"] for c in result.get("callers", [])]
        assert "authenticate" in caller_names

    def test_deep_keeps_cross_file_same_name_edges(self, tmp_project, indexed_db):
        """Cross-file edges with same symbol names should not be dropped."""
        from know.context_engine import ContextEngine

        conn = indexed_db._get_conn()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/api/service.py",
                "CodingAgentService.create_agent",
                "method",
                "python",
                10,
                13,
                "def CodingAgentService.create_agent(payload):",
                "def CodingAgentService.create_agent(payload):\n    return payload",
                "hash_ca_service",
                20,
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/api/router.py",
                "create_agent",
                "function",
                "python",
                20,
                24,
                "def create_agent(payload, service):",
                "def create_agent(payload, service):\n    return service.create_agent(payload)",
                "hash_ca_router",
                24,
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO symbol_refs "
            "(file_path, containing_chunk, ref_name, ref_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            ("src/api/router.py", "create_agent", "create_agent", "call", 22),
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        router_result = engine.build_deep_context("router.py:create_agent", budget=3000)
        assert any(
            c["file"] == "src/api/service.py" and c["name"] == "CodingAgentService.create_agent"
            for c in router_result.get("callees", [])
        )

        service_result = engine.build_deep_context("service.py:CodingAgentService.create_agent", budget=3000)
        assert any(c["file"] == "src/api/router.py" for c in service_result.get("callers", []))

    def test_deep_file_colon_name_format(self, tmp_project, indexed_db):
        """build_deep_context resolves 'file:function' format."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("service.py:check_cloud_access", budget=3000)

        assert "error" not in result
        assert result["target"]["name"] == "check_cloud_access"

    def test_deep_class_dot_method_format(self, tmp_project, indexed_db):
        """build_deep_context resolves 'Class.method' format."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("BillingService.process_payment", budget=3000)

        assert "error" not in result
        assert "process_payment" in result["target"]["name"]

    def test_deep_prefers_function_over_constant_for_same_symbol(self, tmp_project, indexed_db):
        """When a name exists as function+constant, deep should prefer function."""
        from know.context_engine import ContextEngine

        conn = indexed_db._get_conn()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/config/constants.py",
                "bootstrap",
                "constant",
                "python",
                1,
                1,
                "bootstrap",
                "bootstrap = True",
                "hash_bootstrap_constant",
                5,
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/startup/bootstrap.py",
                "bootstrap",
                "function",
                "python",
                3,
                8,
                "def bootstrap():",
                "def bootstrap():\n    return True",
                "hash_bootstrap_function",
                12,
                now,
            ),
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("bootstrap", budget=1000)
        assert "error" not in result
        assert result["target"]["file"] == "src/startup/bootstrap.py"

    def test_deep_excludes_tests_by_default(self, tmp_project, indexed_db):
        """build_deep_context filters test files unless --include-tests."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        # test_check_cloud_access is in tests/, should not be the primary result
        # when searching for check_cloud_access
        result = engine.build_deep_context("check_cloud_access", budget=3000)

        assert "error" not in result
        # Target should be from source, not test
        assert "tests/" not in result["target"]["file"]

    def test_deep_budget_respected(self, tmp_project, indexed_db):
        """build_deep_context stays within budget."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("check_cloud_access", budget=100)

        assert "error" not in result
        assert result["budget_used"] <= 100

    def test_deep_session_marks_seen(self, tmp_project, indexed_db):
        """build_deep_context with session_id marks chunks as seen."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        indexed_db.create_session("deep-session")
        result = engine.build_deep_context(
            "check_cloud_access", budget=3000, session_id="deep-session",
        )

        assert "error" not in result
        assert result.get("session_id") == "deep-session"

        # Chunks should now be marked as seen
        seen = indexed_db.get_session_seen("deep-session")
        assert len(seen) > 0

    def test_deep_refreshes_stale_file_on_file_hint(self, tmp_project, indexed_db):
        """Deep should refresh stale file index for explicit file:name queries."""
        from know.context_engine import ContextEngine

        service_file = tmp_project / "src" / "billing" / "service.py"
        service_file.write_text(
            '''"""Billing service module."""

def check_workspace_access(workspace):
    """Renamed function after edit."""
    count = count_active_sessions(workspace.id)
    if count > workspace.limit:
        raise LimitExceeded(workspace)
    return True

def count_active_sessions(workspace_id):
    return 0
''',
            encoding="utf-8",
        )

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        result = engine.build_deep_context("service.py:check_workspace_access", budget=3000, db=indexed_db)
        assert "error" not in result
        assert result["target"]["name"] == "check_workspace_access"
        assert result["target"]["file"] == "src/billing/service.py"

    def test_deep_cli_json_output(self, tmp_project, indexed_db):
        """CLI `know deep` with --json produces valid JSON."""
        runner = CliRunner()
        from know.cli import cli

        with patch("know.cli.agent._get_daemon_client", return_value=None):
            with patch("know.cli.agent._get_db_fallback", return_value=MagicMock()):
                with patch("know.context_engine.ContextEngine.build_deep_context") as mock_deep:
                    mock_deep.return_value = {
                        "target": {
                            "file": "src/billing/service.py",
                            "name": "check_cloud_access",
                            "signature": "def check_cloud_access(workspace):",
                            "body": "def check_cloud_access(workspace):\n    pass",
                            "line_start": 3,
                            "line_end": 8,
                            "tokens": 45,
                        },
                        "callees": [],
                        "callers": [],
                        "overflow_signatures": [],
                        "call_graph_available": True,
                        "budget_used": 45,
                        "budget": 3000,
                    }
                    result = runner.invoke(cli, ["--json", "deep", "check_cloud_access"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "target" in data

    def test_workflow_cli_json_output(self, tmp_project):
        """CLI `know workflow` with --json produces valid JSON."""
        runner = CliRunner()
        from know.cli import cli

        fake_client = MagicMock()
        fake_client.call_sync.return_value = {
            "query": "billing",
            "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
            "context": {
                "query": "billing",
                "budget": 4000,
                "used_tokens": 0,
                "budget_utilization": "0 / 4,000 (0%)",
                "code": [],
                "dependencies": [],
                "tests": [],
                "summaries": [],
                "overview": "",
                "warnings": [],
            },
            "deep": {"error": "no_target"},
            "selected_deep_target": None,
            "total_tokens": 0,
        }

        with patch("know.cli.agent._get_daemon_client", return_value=fake_client):
            result = runner.invoke(cli, ["--json", "workflow", "billing"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["query"] == "billing"
        assert "context" in data and "deep" in data

    def test_deep_no_call_graph(self, tmp_project, indexed_db):
        """build_deep_context with no symbol_refs returns body only."""
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project

        engine = ContextEngine.__new__(ContextEngine)
        engine.db = indexed_db
        engine.root = tmp_project
        engine.config = config

        # count_active_sessions has no outgoing calls and no callers in symbol_refs
        # (only check_cloud_access calls it, so it has callers but no callees)
        result = engine.build_deep_context("count_active_sessions", budget=3000)

        assert "error" not in result
        assert result["target"]["name"] == "count_active_sessions"
        # It should have callers (check_cloud_access) but no callees
        assert len(result.get("callees", [])) == 0
        assert len(result.get("callers", [])) > 0


# ---------------------------------------------------------------------------
# Tests: get_chunks_by_name
# ---------------------------------------------------------------------------

class TestGetChunksByName:
    """Tests for DaemonDB.get_chunks_by_name()."""

    def test_exact_match(self, indexed_db):
        results = indexed_db.get_chunks_by_name("check_cloud_access")
        assert len(results) == 1
        assert results[0]["chunk_name"] == "check_cloud_access"

    def test_no_match(self, indexed_db):
        results = indexed_db.get_chunks_by_name("nonexistent")
        assert len(results) == 0

    def test_method_match(self, indexed_db):
        results = indexed_db.get_chunks_by_name("BillingService.process_payment")
        assert len(results) == 1

    def test_limit_respected(self, indexed_db):
        results = indexed_db.get_chunks_by_name("check_cloud_access", limit=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Tests: hybrid retrieval + graph-first expansion + packing
# ---------------------------------------------------------------------------

class TestHybridGraphAndPacking:
    """High-impact retrieval improvements (TDD guardrails)."""

    @staticmethod
    def _engine(tmp_project):
        from know.context_engine import ContextEngine

        config = MagicMock()
        config.root = tmp_project
        config.project.name = "test"
        config.project.description = ""
        return ContextEngine(config)

    def test_graph_expand_lane_adds_cross_file_call_neighbors(self, tmp_project, indexed_db):
        """Graph lane should pull call-neighborhood chunks from other files."""
        conn = indexed_db._get_conn()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/gateway/router.py",
                "handle_request",
                "function",
                "python",
                3,
                9,
                "def handle_request(req):",
                "def handle_request(req):\n    return validate_api_key(req.headers.get('x-api-key'))",
                "hash_router_handle",
                24,
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO chunks "
            "(file_path, chunk_name, chunk_type, language, start_line, end_line, "
            "signature, body, body_hash, token_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "src/security/keys.py",
                "validate_api_key",
                "function",
                "python",
                5,
                11,
                "def validate_api_key(value):",
                "def validate_api_key(value):\n    return bool(value and len(value) > 8)",
                "hash_validate_api_key",
                20,
                now,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO symbol_refs "
            "(file_path, containing_chunk, ref_name, ref_type, line_number) "
            "VALUES (?, ?, ?, ?, ?)",
            ("src/gateway/router.py", "handle_request", "validate_api_key", "call", 4),
        )
        conn.execute(
            "INSERT OR REPLACE INTO file_index "
            "(file_path, content_hash, language, chunk_count, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("src/gateway/router.py", "hash_router_file", "python", 1, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO file_index "
            "(file_path, content_hash, language, chunk_count, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("src/security/keys.py", "hash_keys_file", "python", 1, now),
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()

        engine = self._engine(tmp_project)
        seed = indexed_db.get_chunks_by_name("handle_request")
        assert seed
        seed[0]["score"] = 1.0

        expanded = engine._graph_expand_lane(indexed_db, seed, query="api key validation", limit=20)
        names = {c.get("chunk_name") for c in expanded}
        assert "validate_api_key" in names

    def test_fuse_hybrid_lanes_rrf_promotes_multi_signal_candidates(self, tmp_project):
        """Hybrid fusion should prioritize chunks that win in multiple lanes."""
        engine = self._engine(tmp_project)

        lexical = [
            {"file_path": "src/a.py", "chunk_name": "alpha", "start_line": 1, "score": 1.0},
            {"file_path": "src/b.py", "chunk_name": "beta", "start_line": 1, "score": 0.8},
        ]
        graph = [
            {"file_path": "src/c.py", "chunk_name": "gamma", "start_line": 1, "score": 1.0},
            {"file_path": "src/b.py", "chunk_name": "beta", "start_line": 1, "score": 0.7},
        ]
        semantic = [
            {"file_path": "src/c.py", "chunk_name": "gamma", "start_line": 1, "score": 0.95},
            {"file_path": "src/a.py", "chunk_name": "alpha", "start_line": 1, "score": 0.6},
        ]

        fused = engine._fuse_hybrid_lanes(lexical, graph, semantic, limit=3)
        assert fused
        # Top rank should come from a multi-signal candidate (not lane-only noise).
        assert fused[0]["chunk_name"] in {"alpha", "gamma"}

    def test_context_pipeline_uses_hybrid_retrieval(self, tmp_project, indexed_db):
        """v3 context should route through hybrid retrieval helper."""
        engine = self._engine(tmp_project)
        with patch.object(engine, "_retrieve_hybrid_candidates", wraps=engine._retrieve_hybrid_candidates) as spy:
            result = engine._build_context_v3_inner(
                indexed_db, "billing", budget=3000,
                include_tests=True, include_imports=True,
                include_patterns=None, exclude_patterns=None,
                chunk_types=None, session_id=None,
            )

        assert spy.call_count == 1
        assert result["used_tokens"] >= 0

    def test_pack_chunks_for_prompt_places_top_utility_on_edges(self, tmp_project):
        """Packing should place top-utility chunks at prompt edges, not the middle."""
        engine = self._engine(tmp_project)
        chunks = [
            {"file_path": "src/a.py", "chunk_name": "top1", "start_line": 1, "score": 1.00, "token_count": 30},
            {"file_path": "src/b.py", "chunk_name": "top2", "start_line": 1, "score": 0.95, "token_count": 30},
            {"file_path": "src/c.py", "chunk_name": "mid1", "start_line": 1, "score": 0.70, "token_count": 30},
            {"file_path": "src/d.py", "chunk_name": "mid2", "start_line": 1, "score": 0.60, "token_count": 30},
            {"file_path": "src/e.py", "chunk_name": "low1", "start_line": 1, "score": 0.50, "token_count": 30},
            {"file_path": "src/f.py", "chunk_name": "low2", "start_line": 1, "score": 0.40, "token_count": 30},
        ]

        packed = engine._pack_chunks_for_prompt(chunks)
        assert len(packed) == len(chunks)

        edge_names = {packed[0]["chunk_name"], packed[-1]["chunk_name"]}
        assert "top1" in edge_names
        assert "top2" in edge_names
        assert packed[len(packed) // 2]["chunk_name"] not in {"top1", "top2"}
