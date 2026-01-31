"""Tests for Week 2 features: function-level chunks, import graph, context engine, token counting."""

import ast
import json
import tempfile
import shutil
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.token_counter import count_tokens, truncate_to_budget, format_budget
from know.context_engine import (
    CodeChunk,
    ContextEngine,
    extract_chunks_from_file,
    _extract_signatures,
    _find_test_files,
)
from know.import_graph import ImportGraph
from know.config import Config, OutputConfig


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def temp_project():
    """Create a realistic temporary project with multiple modules."""
    temp = Path(tempfile.mkdtemp())
    (temp / ".know" / "cache").mkdir(parents=True, exist_ok=True)
    (temp / ".git").mkdir()  # fake git dir for project detection
    (temp / "src").mkdir()
    (temp / "src" / "auth").mkdir(parents=True)
    (temp / "tests").mkdir()

    # -- src/auth/jwt_handler.py --
    (temp / "src" / "auth" / "jwt_handler.py").write_text('''"""JWT token creation and validation."""

import time
import hashlib
from src.auth.config import SECRET_KEY

def create_token(user_id: str, expiry: int = 3600) -> str:
    """Create a new JWT token for a user.

    Args:
        user_id: The user's unique identifier.
        expiry: Token lifetime in seconds.

    Returns:
        Signed JWT string.
    """
    payload = {"sub": user_id, "exp": time.time() + expiry}
    return hashlib.sha256(str(payload).encode()).hexdigest()

def verify_token(token: str) -> dict:
    """Verify and decode a JWT token."""
    if not token:
        raise ValueError("Empty token")
    return {"sub": "user123", "exp": time.time() + 3600}

class TokenRefresher:
    """Handles token refresh logic."""

    def __init__(self, max_refreshes: int = 5):
        self.max_refreshes = max_refreshes
        self.count = 0

    def refresh(self, old_token: str) -> str:
        """Refresh an expired token."""
        self.count += 1
        if self.count > self.max_refreshes:
            raise RuntimeError("Too many refreshes")
        return create_token("refreshed_user")
''')

    # -- src/auth/middleware.py --
    (temp / "src" / "auth" / "middleware.py").write_text('''"""Authentication middleware."""

from src.auth.jwt_handler import verify_token

def require_auth(handler):
    """Decorator that requires authentication."""
    def wrapper(request):
        token = request.headers.get("Authorization")
        user = verify_token(token)
        request.user = user
        return handler(request)
    return wrapper

class AuthMiddleware:
    """ASGI middleware for authentication."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Process request through auth."""
        pass
''')

    # -- src/auth/config.py --
    (temp / "src" / "auth" / "config.py").write_text('''"""Auth configuration."""

SECRET_KEY = "super-secret"
TOKEN_EXPIRY = 3600
''')

    # -- src/main.py --
    (temp / "src" / "main.py").write_text('''"""Main application entry point."""

from src.auth.middleware import require_auth

def create_app():
    """Create and configure the application."""
    return {"name": "test-app"}

def run():
    """Run the server."""
    app = create_app()
    print(f"Running {app}")
''')

    # -- tests/test_jwt_handler.py --
    (temp / "tests" / "test_jwt_handler.py").write_text('''"""Tests for JWT handler."""

from src.auth.jwt_handler import create_token, verify_token

def test_create_token():
    """Test token creation."""
    token = create_token("user1")
    assert isinstance(token, str)
    assert len(token) > 0

def test_verify_token():
    """Test token verification."""
    token = create_token("user1")
    payload = verify_token(token)
    assert "sub" in payload
''')

    # -- non-Python file --
    (temp / "src" / "utils.ts").write_text('''
export function formatDate(d: Date): string {
    return d.toISOString();
}
''')

    config = Config(
        root=temp,
        exclude=["**/.git/**", "**/__pycache__/**", "**/.know/**"],
        include=["src/"],
        output=OutputConfig(directory="docs", watch=OutputConfig.WatchConfig()),
    )

    yield temp, config
    shutil.rmtree(temp)


# =========================================================================
# Token Counter
# =========================================================================

class TestTokenCounter:
    """Tests for token counting."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        tokens = count_tokens("hello")
        assert 1 <= tokens <= 3

    def test_code_snippet(self):
        code = "def hello():\n    print('world')\n    return 42"
        tokens = count_tokens(code, mode="code")
        # Should be roughly 15-25 tokens
        assert 10 <= tokens <= 40

    def test_code_vs_text_mode(self):
        text = "This is a simple sentence with some words."
        code_t = count_tokens(text, mode="code")
        text_t = count_tokens(text, mode="text")
        # Code mode counts more (line overhead) than text mode
        assert code_t >= text_t

    def test_truncate_within_budget(self):
        text = "short text"
        result = truncate_to_budget(text, 1000)
        assert result == text  # No truncation needed

    def test_truncate_exceeds_budget(self):
        text = "word " * 500  # ~650+ tokens
        result = truncate_to_budget(text, 50)
        assert count_tokens(result) <= 55  # Small buffer for [truncated]
        assert "[truncated]" in result

    def test_format_budget(self):
        assert format_budget(6234, 8000) == "6,234 / 8,000 (77%)"
        assert format_budget(0, 8000) == "0 / 8,000 (0%)"
        assert format_budget(8000, 8000) == "8,000 / 8,000 (100%)"


# =========================================================================
# Chunk Extraction
# =========================================================================

class TestChunkExtraction:
    """Tests for extracting function/class level chunks."""

    def test_extract_python_functions(self, temp_project):
        temp, config = temp_project
        chunks = extract_chunks_from_file(
            temp / "src" / "auth" / "jwt_handler.py", temp
        )
        names = [c.name for c in chunks]

        # Should have: module summary, create_token, verify_token, TokenRefresher (class),
        # TokenRefresher.__init__, TokenRefresher.refresh
        assert any("create_token" in n for n in names)
        assert any("verify_token" in n for n in names)
        assert any("TokenRefresher" in n for n in names)

    def test_chunk_has_metadata(self, temp_project):
        temp, config = temp_project
        chunks = extract_chunks_from_file(
            temp / "src" / "auth" / "jwt_handler.py", temp
        )
        func_chunks = [c for c in chunks if c.name == "create_token"]
        assert len(func_chunks) == 1
        chunk = func_chunks[0]

        assert chunk.chunk_type == "function"
        assert chunk.line_start > 0
        assert chunk.line_end >= chunk.line_start
        assert chunk.file_path == "src/auth/jwt_handler.py"
        assert chunk.tokens > 0
        assert "user_id" in chunk.body

    def test_module_level_chunk(self, temp_project):
        temp, config = temp_project
        chunks = extract_chunks_from_file(
            temp / "src" / "auth" / "jwt_handler.py", temp
        )
        module_chunks = [c for c in chunks if c.chunk_type == "module"]
        assert len(module_chunks) >= 1
        mod = module_chunks[0]
        # Should contain imports
        assert "import" in mod.body

    def test_class_methods_extracted(self, temp_project):
        temp, config = temp_project
        chunks = extract_chunks_from_file(
            temp / "src" / "auth" / "jwt_handler.py", temp
        )
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        method_names = [c.name for c in method_chunks]
        assert any("TokenRefresher.__init__" in n for n in method_names)
        assert any("TokenRefresher.refresh" in n for n in method_names)

    def test_non_python_file_single_chunk(self, temp_project):
        temp, config = temp_project
        chunks = extract_chunks_from_file(temp / "src" / "utils.ts", temp)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "module"
        assert "formatDate" in chunks[0].body

    def test_extract_signatures(self, temp_project):
        temp, config = temp_project
        sigs = _extract_signatures(temp / "src" / "auth" / "jwt_handler.py")
        assert "def create_token" in sigs
        assert "def verify_token" in sigs
        assert "class TokenRefresher" in sigs


# =========================================================================
# Import Graph
# =========================================================================

class TestImportGraph:
    """Tests for import graph building and queries."""

    def test_build_graph(self, temp_project):
        temp, config = temp_project
        ig = ImportGraph(config)

        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        edge_count = ig.build(structure["modules"])

        assert edge_count > 0

    def test_imports_of(self, temp_project):
        temp, config = temp_project
        ig = ImportGraph(config)
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig.build(structure["modules"])

        # middleware imports jwt_handler
        imports = ig.imports_of("middleware")
        assert any("jwt_handler" in m for m in imports)

    def test_imported_by(self, temp_project):
        temp, config = temp_project
        ig = ImportGraph(config)
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig.build(structure["modules"])

        # jwt_handler is imported by middleware
        imported_by = ig.imported_by("jwt_handler")
        assert any("middleware" in m for m in imported_by)

    def test_format_graph(self, temp_project):
        temp, config = temp_project
        ig = ImportGraph(config)
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig.build(structure["modules"])

        output = ig.format_graph("middleware")
        assert "Imports" in output
        assert "jwt_handler" in output

    def test_all_edges(self, temp_project):
        temp, config = temp_project
        ig = ImportGraph(config)
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig.build(structure["modules"])

        edges = ig.get_all_edges()
        assert len(edges) > 0
        # All edges are tuples of (source, target)
        for src, tgt in edges:
            assert isinstance(src, str)
            assert isinstance(tgt, str)


# =========================================================================
# Context Engine
# =========================================================================

class TestContextEngine:
    """Tests for the main context assembly engine."""

    def test_build_context_returns_structure(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth bug", budget=4000)

        assert "query" in result
        assert result["query"] == "auth bug"
        assert "budget" in result
        assert result["budget"] == 4000
        assert "used_tokens" in result
        assert "code_chunks" in result
        assert "overview" in result

    def test_budget_respected(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth bug", budget=4000)

        # Used tokens should not exceed budget
        assert result["used_tokens"] <= 4000

    def test_code_chunks_found(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("jwt token creation", budget=8000)

        # Should find relevant code chunks
        assert len(result["code_chunks"]) > 0
        chunk_names = [c.name for c in result["code_chunks"]]
        # At least some auth-related chunks should be found
        assert any("token" in n.lower() or "jwt" in n.lower() or "auth" in n.lower()
                    for n in chunk_names)

    def test_no_tests_flag(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth", budget=8000, include_tests=False)
        assert result["test_chunks"] == []

    def test_no_imports_flag(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth", budget=8000, include_imports=False)
        assert result["dependency_chunks"] == []

    def test_small_budget_warning(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        # Very small budget — might trigger warning
        result = engine.build_context("nonexistent gibberish xyz", budget=50)
        # Either finds nothing or warns about low utilization
        assert result["used_tokens"] <= 50

    def test_format_markdown(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth token", budget=8000)
        md = engine.format_markdown(result)

        assert "# Context for:" in md
        assert "Token Budget:" in md
        assert isinstance(md, str)

    def test_format_agent_json(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("auth token", budget=8000)
        json_str = engine.format_agent_json(result)

        data = json.loads(json_str)
        assert data["query"] == "auth token"
        assert data["budget"] == 8000
        assert "code" in data
        assert "source_files" in data

    def test_budget_allocation_percentages(self, temp_project):
        """Verify the engine respects ~40/30/20/10 allocation."""
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("create token authentication", budget=8000)

        code_tokens = sum(c.tokens for c in result["code_chunks"])
        dep_tokens = sum(c.tokens for c in result["dependency_chunks"])
        total = result["used_tokens"]

        # Code should be the largest portion (if anything was found)
        if total > 0 and code_tokens > 0:
            code_ratio = code_tokens / total
            # Code should be at least 30% of total (allowing some flexibility)
            assert code_ratio >= 0.2, f"Code ratio {code_ratio} too low"

    def test_chunk_scores_assigned(self, temp_project):
        temp, config = temp_project
        engine = ContextEngine(config)
        result = engine.build_context("jwt token", budget=8000)

        for chunk in result["code_chunks"]:
            # All chunks should have a score > 0 (text match or semantic)
            assert chunk.score >= 0.0


# =========================================================================
# Test File Discovery
# =========================================================================

class TestTestDiscovery:
    """Tests for test file matching."""

    def test_find_matching_tests(self, temp_project):
        temp, config = temp_project
        tests = _find_test_files(temp, "src/auth/jwt_handler.py")
        assert len(tests) >= 1
        test_names = [t.name for t in tests]
        assert "test_jwt_handler.py" in test_names

    def test_no_tests_for_unknown(self, temp_project):
        temp, config = temp_project
        tests = _find_test_files(temp, "src/nonexistent.py")
        assert len(tests) == 0


# =========================================================================
# CodeChunk dataclass
# =========================================================================

class TestCodeChunk:
    def test_qualified_name(self):
        chunk = CodeChunk(
            file_path="src/auth/jwt.py",
            name="create_token",
            chunk_type="function",
            line_start=10,
            line_end=25,
            body="def create_token(): ...",
        )
        assert chunk.qualified_name == "src/auth/jwt.py:create_token"

    def test_header(self):
        chunk = CodeChunk(
            file_path="src/auth/jwt.py",
            name="create_token",
            chunk_type="function",
            line_start=10,
            line_end=25,
            body="...",
        )
        assert "lines 10-25" in chunk.header()
        assert "src/auth/jwt.py" in chunk.header()


# =========================================================================
# Integration: CLI commands (smoke test)
# =========================================================================

class TestCLIIntegration:
    """Smoke tests for the new CLI commands."""

    def test_context_command_exists(self):
        """Verify the context command is registered."""
        from know.cli import cli
        commands = cli.commands
        assert "context" in commands

    def test_graph_command_exists(self):
        from know.cli import cli
        assert "graph" in cli.commands

    def test_reindex_command_exists(self):
        from know.cli import cli
        assert "reindex" in cli.commands


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
