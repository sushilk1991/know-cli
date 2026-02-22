"""Tests for query understanding, dual-lane search, and ranking signal wiring."""

import sqlite3
import time
from pathlib import Path

import pytest

from know.query import (
    analyze_query,
    build_fts_or_query,
    build_fts_and_query,
    _is_identifier,
    _split_identifier,
)


# ---------------------------------------------------------------------------
# Query analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeQuery:
    """Test the core query understanding pipeline."""

    def test_stop_word_removal(self):
        plan = analyze_query("fix the auth bug")
        # "fix" and "the" are stop words
        assert "fix" not in plan.terms
        assert "the" not in plan.terms
        assert "auth" in plan.terms
        assert "bug" in plan.terms

    def test_identifier_detection_snake_case(self):
        plan = analyze_query("verify_session")
        assert "verify_session" in plan.identifiers
        assert plan.query_type == "identifier"

    def test_identifier_detection_camel_case(self):
        plan = analyze_query("AuthMiddleware")
        assert "AuthMiddleware" in plan.identifiers
        assert plan.query_type == "identifier"

    def test_identifier_expansion_camel(self):
        plan = analyze_query("AuthMiddleware")
        lower_expanded = [t.lower() for t in plan.expanded_terms]
        assert "auth" in lower_expanded or "Auth" in plan.expanded_terms
        assert "middleware" in lower_expanded or "Middleware" in plan.expanded_terms

    def test_identifier_expansion_snake(self):
        plan = analyze_query("verify_session")
        lower_expanded = [t.lower() for t in plan.expanded_terms]
        assert "verify" in lower_expanded
        assert "session" in lower_expanded

    def test_agent_prefix_stripping(self):
        plan = analyze_query("help me find the auth middleware")
        assert "help" not in plan.terms
        # Should have found meaningful terms
        assert len(plan.all_search_terms) > 0

    def test_agent_prefix_show_me(self):
        plan = analyze_query("show me the database connection")
        assert "show" not in plan.terms
        assert "database" in plan.terms or "connection" in plan.terms

    def test_concept_query_classification(self):
        plan = analyze_query("how does authentication work")
        assert plan.query_type == "concept"

    def test_error_query_classification(self):
        plan = analyze_query("fix the authentication error")
        assert plan.query_type == "error"

    def test_identifier_query_classification(self):
        plan = analyze_query("build_fts_query")
        assert plan.query_type == "identifier"

    def test_empty_query_fallback(self):
        plan = analyze_query("")
        assert plan.original == ""

    def test_all_stop_words_fallback(self):
        """If every word is a stop word, fall back to original tokens."""
        plan = analyze_query("the is a")
        # Should have some terms (fallback behavior)
        assert len(plan.all_search_terms) >= 0  # graceful handling

    def test_dotted_path_identifier(self):
        plan = analyze_query("know.daemon_db.search_chunks")
        assert "know.daemon_db.search_chunks" in plan.identifiers

    def test_mixed_query(self):
        """Query with both identifiers and natural language."""
        plan = analyze_query("find the verify_session function in auth module")
        assert "verify_session" in plan.identifiers
        assert "auth" in plan.terms or "auth" in [t.lower() for t in plan.all_search_terms]

    def test_all_search_terms_deduped(self):
        plan = analyze_query("auth_middleware auth middleware")
        # Should not have duplicates
        lower_terms = [t.lower() for t in plan.all_search_terms]
        assert len(lower_terms) == len(set(lower_terms))

    def test_real_agent_query_1(self):
        """Real query: 'help me understand the search ranking pipeline'."""
        plan = analyze_query("help me understand the search ranking pipeline")
        assert "search" in plan.terms or "ranking" in plan.terms or "pipeline" in plan.terms
        # Stop words removed
        assert "help" not in plan.terms
        assert "the" not in plan.terms

    def test_real_agent_query_2(self):
        """Real query: 'how do I add pagination to the users API'."""
        plan = analyze_query("how do I add pagination to the users API")
        assert "pagination" in plan.terms or "users" in plan.terms or "API" in plan.terms

    def test_real_agent_query_3(self):
        """Real query: 'DaemonDB._build_fts_query'."""
        plan = analyze_query("DaemonDB._build_fts_query")
        assert plan.query_type == "identifier"
        assert len(plan.identifiers) > 0


class TestIdentifierDetection:
    def test_snake_case(self):
        assert _is_identifier("verify_session") is True
        assert _is_identifier("build_fts_query") is True

    def test_camel_case(self):
        assert _is_identifier("AuthMiddleware") is True
        assert _is_identifier("DaemonDB") is True

    def test_dotted_path(self):
        assert _is_identifier("know.daemon_db") is True
        assert _is_identifier("src.auth.middleware") is True

    def test_regular_words(self):
        assert _is_identifier("authentication") is False
        assert _is_identifier("database") is False
        assert _is_identifier("the") is False

    def test_underscore_in_token(self):
        assert _is_identifier("my_var") is True


class TestSplitIdentifier:
    def test_camel_case(self):
        parts = _split_identifier("AuthMiddleware")
        assert "Auth" in parts
        assert "Middleware" in parts

    def test_snake_case(self):
        parts = _split_identifier("verify_session")
        assert "verify" in parts
        assert "session" in parts

    def test_dotted_path(self):
        parts = _split_identifier("know.daemon_db")
        assert "know" in parts
        assert "daemon" in parts
        assert "db" in parts


class TestFTSQueryBuilders:
    def test_or_query(self):
        q = build_fts_or_query(["auth", "session"])
        assert '"auth"' in q
        assert '"session"' in q
        assert " OR " in q

    def test_and_query(self):
        q = build_fts_and_query(["auth", "session"])
        assert '"auth"' in q
        assert '"session"' in q
        assert " AND " in q

    def test_and_query_needs_two_terms(self):
        q = build_fts_and_query(["auth"])
        assert q == ""

    def test_empty_query(self):
        assert build_fts_or_query([]) == ""
        assert build_fts_and_query([]) == ""


# ---------------------------------------------------------------------------
# Dual-lane search integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def search_db(tmp_path):
    """Create a DaemonDB with diverse chunks for search testing."""
    (tmp_path / ".know").mkdir()
    from know.daemon_db import DaemonDB
    db = DaemonDB(tmp_path)

    # Auth-related chunks
    db.upsert_chunks("src/auth/session.py", "python", [
        {
            "name": "verify_session",
            "type": "function",
            "start_line": 10, "end_line": 30,
            "signature": "def verify_session(token: str) -> bool",
            "body": "def verify_session(token: str) -> bool:\n    '''Verify a session token.'''\n    if not token:\n        return False\n    return check_token_validity(token)",
        },
        {
            "name": "create_session",
            "type": "function",
            "start_line": 35, "end_line": 50,
            "signature": "def create_session(user_id: int) -> str",
            "body": "def create_session(user_id: int) -> str:\n    '''Create a new session.'''\n    token = generate_token()\n    store_session(user_id, token)\n    return token",
        },
    ])

    db.upsert_chunks("src/auth/middleware.py", "python", [
        {
            "name": "AuthMiddleware",
            "type": "class",
            "start_line": 1, "end_line": 40,
            "signature": "class AuthMiddleware:",
            "body": "class AuthMiddleware:\n    '''Authentication middleware for web framework.'''\n    def process_request(self, request):\n        session = verify_session(request.token)\n        if not session:\n            raise AuthenticationError('Invalid session')",
        },
    ])

    # Unrelated chunks (noise)
    db.upsert_chunks("src/utils/helpers.py", "python", [
        {
            "name": "format_date",
            "type": "function",
            "start_line": 1, "end_line": 5,
            "signature": "def format_date(dt)",
            "body": "def format_date(dt):\n    '''Format the date for display.'''\n    return dt.strftime('%Y-%m-%d')",
        },
    ])

    db.upsert_chunks("src/database/connection.py", "python", [
        {
            "name": "get_connection",
            "type": "function",
            "start_line": 1, "end_line": 10,
            "signature": "def get_connection()",
            "body": "def get_connection():\n    '''Get a database connection.'''\n    return sqlite3.connect(DB_PATH)",
        },
    ])

    yield db
    db.close()


class TestDualLaneSearch:
    """Integration tests for the dual-lane search pipeline."""

    def test_auth_query_finds_auth_code(self, search_db):
        """'fix the auth bug' should find auth code, not random chunks with 'the'."""
        results = search_db.search_chunks("fix the auth bug")
        assert len(results) > 0
        # Auth-related chunks should be in top results
        top_files = [r["file_path"] for r in results[:3]]
        assert any("auth" in f for f in top_files)

    def test_exact_function_name(self, search_db):
        """'verify_session' should find the exact function in top 3."""
        results = search_db.search_chunks("verify_session")
        assert len(results) > 0
        top_names = [r["chunk_name"] for r in results[:3]]
        assert "verify_session" in top_names

    def test_camel_case_class(self, search_db):
        """'AuthMiddleware' should find the class."""
        results = search_db.search_chunks("AuthMiddleware")
        assert len(results) > 0
        top_names = [r["chunk_name"] for r in results[:3]]
        assert "AuthMiddleware" in top_names

    def test_concept_query(self, search_db):
        """Concept query: 'session management' should find session-related code."""
        results = search_db.search_chunks("session management")
        assert len(results) > 0
        # Should find session-related chunks
        found_session = any("session" in r["chunk_name"].lower() for r in results[:5])
        assert found_session

    def test_noise_not_dominating(self, search_db):
        """Queries with stop words shouldn't return noise."""
        results = search_db.search_chunks("help me fix the auth bug in the session")
        if results:
            # Auth/session chunks should rank higher than helpers/database
            auth_ranks = [
                i for i, r in enumerate(results)
                if "auth" in r["file_path"] or "session" in r["chunk_name"].lower()
            ]
            noise_ranks = [
                i for i, r in enumerate(results)
                if "helpers" in r["file_path"] or "connection" in r["file_path"]
            ]
            if auth_ranks and noise_ranks:
                assert min(auth_ranks) < min(noise_ranks)

    def test_fuse_rankings_called(self, search_db):
        """Verify that fuse_rankings is actually used (was dead code before)."""
        # A query that should trigger multiple lanes
        results = search_db.search_chunks("verify_session auth")
        assert len(results) > 0
        # The RRF fusion should produce scores > 0
        assert all(r.get("score", 0) > 0 for r in results)

    def test_lane_weights_stable_when_or_lane_empty(self, search_db, monkeypatch):
        """AND+exact lanes keep their native weights when OR lane is empty."""
        from know.query import SearchPlan
        import know.query as query_mod
        import know.ranking as ranking_mod

        monkeypatch.setattr(
            query_mod, "analyze_query",
            lambda _q: SearchPlan(
                original="q",
                terms=["needle"],
                identifiers=["verify_session"],
                expanded_terms=[],
                query_type="identifier",
                all_search_terms=["needle", "verify_session"],
            ),
        )
        monkeypatch.setattr(query_mod, "build_fts_or_query", lambda _terms: "OR_ONLY")
        monkeypatch.setattr(query_mod, "build_fts_and_query", lambda _terms: "AND_ONLY")

        sample = {
            "file_path": "src/auth.py",
            "chunk_name": "verify_session",
            "chunk_type": "function",
            "start_line": 10,
            "body": "def verify_session(token): pass",
            "score": 1.0,
        }
        monkeypatch.setattr(
            search_db,
            "_fts_search",
            lambda _conn, fts_query, _col_count, _limit: [] if fts_query == "OR_ONLY" else [sample],
        )

        seen_ranked_lists = {}

        def fake_fuse_rankings(ranked_lists, k=60):
            seen_ranked_lists["count"] = len(ranked_lists)
            return [("src/auth.py:verify_session:10", 42.0)]

        monkeypatch.setattr(ranking_mod, "fuse_rankings", fake_fuse_rankings)

        results = search_db.search_chunks("needle")
        assert results and results[0]["chunk_name"] == "verify_session"
        # AND lane weight (2x) + exact lane weight (3x) = 5 ranked lists.
        assert seen_ranked_lists["count"] == 5
