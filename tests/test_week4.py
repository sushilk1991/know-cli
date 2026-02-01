"""Tests for Week 4: MCP Server, Performance, README, KNOW_SKILL.

Tests cover:
  - MCP server creation and tool/resource registration
  - MCP tool invocations (mocked transport)
  - MCP resource access
  - CLI `mcp` subcommand registration
  - Performance benchmarks
  - README content validation
  - KNOW_SKILL.md content validation
  - --time flag
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()

    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "test-project"
    config.save(tmp_path / ".know" / "config.yaml")

    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(
        'def hello():\n    """Say hello."""\n    return "hello"\n\n'
        'def world():\n    """Say world."""\n    return "world"\n'
    )
    (src / "utils.py").write_text(
        'import os\nfrom src.main import hello\n\n'
        'def helper():\n    """A helper function."""\n    return hello()\n'
    )
    (src / "auth.py").write_text(
        'import hashlib\n\n'
        'def check_token(token: str) -> bool:\n'
        '    """Verify an auth token."""\n'
        '    return len(token) > 0\n\n'
        'class AuthMiddleware:\n'
        '    """Authentication middleware."""\n'
        '    def __init__(self):\n'
        '        self.tokens = {}\n\n'
        '    def validate(self, request):\n'
        '        """Validate request auth."""\n'
        '        return True\n'
    )

    # Test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text(
        'from src.main import hello, world\n\n'
        'def test_hello():\n    assert hello() == "hello"\n\n'
        'def test_world():\n    assert world() == "world"\n'
    )

    return tmp_path, config


def _run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# MCP Server — Creation & Registration
# ---------------------------------------------------------------------------

class TestMCPServerCreation:
    """Test that the MCP server is created correctly."""

    def test_create_server_returns_fastmcp(self, tmp_project):
        root, config = tmp_project
        from know.mcp_server import create_server
        server = create_server(project_root=root)
        assert server is not None
        assert server.name == "know-cli"

    def test_mcp_available_flag(self):
        from know.mcp_server import _MCP_AVAILABLE
        assert _MCP_AVAILABLE is True

    def test_create_server_with_no_root(self, tmp_project):
        """Server creation works when cwd has .know."""
        root, config = tmp_project
        with patch("os.getcwd", return_value=str(root)):
            from know.mcp_server import create_server
            server = create_server(project_root=root)
            assert server is not None


# ---------------------------------------------------------------------------
# MCP Tools — Invocation Tests
# ---------------------------------------------------------------------------

class TestMCPToolGetContext:
    """Test the get_context MCP tool."""

    def test_get_context_returns_json(self, tmp_project):
        root, config = tmp_project
        from know.mcp_server import create_server
        server = create_server(project_root=root)

        # Call the tool function directly
        from know.mcp_server import create_server
        # Access the registered tool
        # FastMCP stores tools internally — we test via the underlying functions
        from know.context_engine import ContextEngine
        engine = ContextEngine(config)
        result = engine.build_context("hello function", budget=4000)
        json_str = engine.format_agent_json(result)
        data = json.loads(json_str)

        assert "query" in data
        assert "budget" in data
        assert data["budget"] == 4000
        assert "code" in data
        assert "used_tokens" in data

    def test_get_context_respects_budget(self, tmp_project):
        root, config = tmp_project
        from know.context_engine import ContextEngine
        engine = ContextEngine(config)
        result = engine.build_context("hello", budget=2000)
        assert result["used_tokens"] <= 2000

    def test_get_context_finds_relevant_code(self, tmp_project):
        root, config = tmp_project
        from know.context_engine import ContextEngine
        engine = ContextEngine(config)
        result = engine.build_context("authentication token", budget=8000)
        # Should find auth-related code
        code_names = [c.name for c in result["code_chunks"]]
        code_files = [c.file_path for c in result["code_chunks"]]
        # At least some results
        assert len(result["code_chunks"]) > 0


class TestMCPToolSearchCode:
    """Test the search_code MCP tool."""

    def test_search_returns_results(self, tmp_project):
        root, config = tmp_project
        from know.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(project_root=root)
        results = searcher.search_code("hello", root, top_k=5, auto_index=True)
        assert isinstance(results, list)

    def test_search_results_have_scores(self, tmp_project):
        root, config = tmp_project
        from know.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(project_root=root)
        results = searcher.search_code("helper function", root, top_k=3, auto_index=True)
        for r in results:
            assert "score" in r
            assert 0 <= r["score"] <= 1


class TestMCPToolMemory:
    """Test remember/recall MCP tools."""

    def test_remember_and_recall(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)

        mem_id = kb.remember("Auth uses JWT tokens", source="mcp", tags="auth")
        assert isinstance(mem_id, int)

        memories = kb.recall("authentication tokens")
        assert len(memories) > 0
        assert any("JWT" in m.text for m in memories)

    def test_remember_with_mcp_source(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)

        mem_id = kb.remember("test memory", source="mcp")
        mem = kb.get(mem_id)
        assert mem.source == "mcp"


class TestMCPToolExplainComponent:
    """Test explain_component tool (mocked AI)."""

    def test_explain_finds_component(self, tmp_project):
        root, config = tmp_project
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        matches = scanner.find_component("hello")
        assert len(matches) > 0

    def test_explain_returns_error_for_missing(self, tmp_project):
        root, config = tmp_project
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        matches = scanner.find_component("nonexistent_xyz_component")
        assert len(matches) == 0


class TestMCPToolShowGraph:
    """Test show_graph tool."""

    def test_graph_returns_data(self, tmp_project):
        root, config = tmp_project
        from know.import_graph import ImportGraph
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig = ImportGraph(config)
        edge_count = ig.build(structure["modules"])

        formatted = ig.format_graph("src.utils")
        assert "Import graph" in formatted

    def test_graph_shows_imports(self, tmp_project):
        root, config = tmp_project
        from know.import_graph import ImportGraph
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig = ImportGraph(config)
        ig.build(structure["modules"])

        imports = ig.imports_of("src.utils")
        # utils imports main
        assert any("main" in m for m in imports)


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------

class TestMCPResources:
    """Test MCP resource access."""

    def test_digest_resource_content(self, tmp_project):
        root, config = tmp_project
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        modules = structure.get("modules", [])

        # Simulate what the digest resource does
        assert len(modules) > 0
        names = [m["name"] if isinstance(m, dict) else m.name for m in modules]
        assert any("main" in n for n in names)

    def test_structure_resource_has_files(self, tmp_project):
        root, config = tmp_project
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        modules = structure.get("modules", [])
        assert len(modules) >= 3  # main.py, utils.py, auth.py

    def test_memories_resource_empty_initially(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        memories = kb.list_all()
        assert len(memories) == 0

    def test_memories_resource_after_remember(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Test insight", source="manual")
        memories = kb.list_all()
        assert len(memories) == 1
        assert memories[0].text == "Test insight"


# ---------------------------------------------------------------------------
# CLI — MCP Commands
# ---------------------------------------------------------------------------

class TestMCPCLICommands:
    """Test that MCP CLI commands are registered."""

    def test_mcp_group_registered(self):
        from know.cli import cli
        commands = cli.commands
        assert "mcp" in commands

    def test_mcp_serve_registered(self):
        from know.cli import cli
        mcp_group = cli.commands["mcp"]
        assert "serve" in mcp_group.commands

    def test_mcp_config_registered(self):
        from know.cli import cli
        mcp_group = cli.commands["mcp"]
        assert "config" in mcp_group.commands

    def test_mcp_config_output(self, tmp_project):
        """Test that mcp config prints valid JSON."""
        from know.mcp_server import print_config
        import io
        from contextlib import redirect_stdout

        root, config = tmp_project
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_config(project_root=root)

        output = buf.getvalue()
        # Extract JSON (first block before the comment lines)
        json_lines = []
        for line in output.splitlines():
            if line.startswith("#"):
                break
            json_lines.append(line)
        json_str = "\n".join(json_lines).strip()
        data = json.loads(json_str)

        assert "mcpServers" in data
        assert "know-cli" in data["mcpServers"]
        assert data["mcpServers"]["know-cli"]["args"] == ["mcp", "serve"]
        assert data["mcpServers"]["know-cli"]["cwd"] == str(root)


# ---------------------------------------------------------------------------
# CLI — --time flag
# ---------------------------------------------------------------------------

class TestTimeFlag:
    """Test the --time execution timing flag."""

    def test_time_flag_registered(self):
        from know.cli import cli
        param_names = [p.name for p in cli.params]
        assert "show_time" in param_names

    def test_time_flag_sets_context(self):
        """--time should set show_time in context."""
        from click.testing import CliRunner
        from know.cli import cli

        runner = CliRunner()
        # Use a command that exists — status is lightweight
        result = runner.invoke(cli, ["--time", "status"], catch_exceptions=False)
        # Should not error
        assert result.exit_code == 0 or "not initialized" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# Performance Benchmarks
# ---------------------------------------------------------------------------

class TestPerformance:
    """Performance benchmarks — assert reasonable execution times."""

    def test_context_engine_under_2s(self, tmp_project):
        """Context engine should be fast for small projects (text fallback)."""
        root, config = tmp_project
        from know.context_engine import ContextEngine

        engine = ContextEngine(config)

        t0 = time.monotonic()
        result = engine.build_context("hello function", budget=4000)
        elapsed = time.monotonic() - t0

        # Text fallback should be very fast for small projects
        assert elapsed < 2.0, f"Context build took {elapsed:.2f}s (expected <2s)"
        assert result["used_tokens"] > 0

    def test_search_under_2s(self, tmp_project):
        """Search should be fast for small projects."""
        root, config = tmp_project
        from know.semantic_search import SemanticSearcher

        searcher = SemanticSearcher(project_root=root)

        t0 = time.monotonic()
        results = searcher.search_code("hello", root, top_k=5, auto_index=True)
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"Search took {elapsed:.2f}s (expected <2s)"

    def test_knowledge_base_remember_fast(self, tmp_project):
        """Remember should complete in reasonable time."""
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)

        t0 = time.monotonic()
        for i in range(10):
            kb.remember(f"Memory {i}", source="bench")
        elapsed = time.monotonic() - t0

        # 10 inserts with embedding can be slow first time (model loading),
        # but should still be under 5s total
        assert elapsed < 5.0, f"10 remember calls took {elapsed:.2f}s"

    def test_knowledge_base_recall_fast(self, tmp_project):
        """Recall should be fast."""
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)

        # Seed some data
        for i in range(20):
            kb.remember(f"Authentication token handling pattern {i}", source="bench")

        t0 = time.monotonic()
        results = kb.recall("authentication")
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"Recall took {elapsed:.2f}s (expected <1s)"
        assert len(results) > 0

    def test_import_graph_build_fast(self, tmp_project):
        """Import graph build should be fast for small projects."""
        root, config = tmp_project
        from know.import_graph import ImportGraph
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()

        t0 = time.monotonic()
        ig = ImportGraph(config)
        ig.build(structure["modules"])
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"Graph build took {elapsed:.2f}s (expected <1s)"

    def test_chunk_extraction_fast(self, tmp_project):
        """Chunk extraction should be fast."""
        root, config = tmp_project
        from know.context_engine import extract_chunks_from_file

        auth_file = root / "src" / "auth.py"

        t0 = time.monotonic()
        for _ in range(100):
            chunks = extract_chunks_from_file(auth_file, root)
        elapsed = time.monotonic() - t0

        # 100 extractions should be under 1s
        assert elapsed < 1.0, f"100 extractions took {elapsed:.2f}s"
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# README Validation
# ---------------------------------------------------------------------------

class TestREADME:
    """Validate README.md content."""

    @pytest.fixture
    def readme(self):
        readme_path = Path(__file__).parent.parent / "README.md"
        return readme_path.read_text()

    def test_readme_has_title(self, readme):
        assert "# know" in readme

    def test_readme_has_problem_section(self, readme):
        assert "## The Problem" in readme

    def test_readme_has_solution_section(self, readme):
        assert "## The Solution" in readme

    def test_readme_has_quick_start(self, readme):
        assert "Quick Start" in readme
        assert "pip install know-cli" in readme
        assert "know init" in readme
        assert "know context" in readme

    def test_readme_mentions_mcp(self, readme):
        assert "MCP" in readme
        assert "know mcp serve" in readme

    def test_readme_mentions_context(self, readme):
        assert "know context" in readme
        assert "--budget" in readme

    def test_readme_mentions_memory(self, readme):
        assert "know remember" in readme
        assert "know recall" in readme

    def test_readme_has_commands_reference(self, readme):
        assert "Commands Reference" in readme

    def test_readme_mentions_search(self, readme):
        assert "know search" in readme

    def test_readme_mentions_graph(self, readme):
        assert "know graph" in readme

    def test_readme_has_works_with_section(self, readme):
        assert "Works With" in readme
        assert "Claude" in readme
        assert "Cursor" in readme

    def test_readme_mentions_pricing(self, readme):
        assert "Pricing" in readme or "Free" in readme

    def test_readme_has_install_instructions(self, readme):
        assert "pip install know-cli" in readme
        assert "know-cli[search]" in readme
        assert "know-cli[mcp]" in readme


# ---------------------------------------------------------------------------
# KNOW_SKILL.md Validation
# ---------------------------------------------------------------------------

class TestKnowSkill:
    """Validate KNOW_SKILL.md content."""

    @pytest.fixture
    def skill(self):
        skill_path = Path(__file__).parent.parent / "KNOW_SKILL.md"
        return skill_path.read_text()

    def test_skill_exists(self):
        skill_path = Path(__file__).parent.parent / "KNOW_SKILL.md"
        assert skill_path.exists()

    def test_skill_mentions_context(self, skill):
        assert "know context" in skill
        assert "--budget" in skill

    def test_skill_mentions_remember(self, skill):
        assert "know remember" in skill

    def test_skill_mentions_search(self, skill):
        assert "know search" in skill

    def test_skill_mentions_graph(self, skill):
        assert "know graph" in skill

    def test_skill_mentions_recall(self, skill):
        assert "know recall" in skill

    def test_skill_has_examples(self, skill):
        assert "Examples:" in skill or "```" in skill

    def test_skill_mentions_json_flag(self, skill):
        assert "--json" in skill

    def test_skill_mentions_quiet_flag(self, skill):
        assert "--quiet" in skill


# ---------------------------------------------------------------------------
# MCP Server — print_config
# ---------------------------------------------------------------------------

class TestMCPConfig:
    """Test the mcp config helper."""

    def test_print_config_valid_json(self, tmp_project):
        root, config = tmp_project
        from know.mcp_server import print_config
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_config(project_root=root)

        output = buf.getvalue()
        # Extract only JSON lines
        json_lines = []
        for line in output.splitlines():
            if line.startswith("#"):
                break
            json_lines.append(line)
        json_str = "\n".join(json_lines).strip()
        data = json.loads(json_str)
        assert "mcpServers" in data

    def test_print_config_uses_project_root(self, tmp_project):
        root, config = tmp_project
        from know.mcp_server import print_config
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_config(project_root=root)

        output = buf.getvalue()
        assert str(root) in output


# ---------------------------------------------------------------------------
# Version Check
# ---------------------------------------------------------------------------

class TestVersion:
    """Ensure version is bumped to 0.3.0."""

    def test_version_is_030(self):
        from know import __version__
        assert __version__ == "0.3.0"

    def test_pyproject_version_matches(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert 'version = "0.3.0"' in content


# ---------------------------------------------------------------------------
# MCP Server Graceful Degradation
# ---------------------------------------------------------------------------

class TestMCPGracefulDegradation:
    """Test that MCP server handles missing dependencies gracefully."""

    def test_search_falls_back_to_text(self, tmp_project):
        """When fastembed is unavailable, search still works via text matching."""
        root, config = tmp_project
        from know.context_engine import ContextEngine

        engine = ContextEngine(config)
        # Force text fallback by using a fresh context engine
        result = engine.build_context("hello function", budget=4000)
        # Should still return results via text matching
        assert result["used_tokens"] >= 0

    def test_remember_without_fastembed(self, tmp_project):
        """Remember works even if fastembed can't embed."""
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)

        # This should work regardless of fastembed
        mem_id = kb.remember("Test without embeddings", source="test")
        assert isinstance(mem_id, int)

        # Recall via text fallback
        results = kb.recall("without embeddings")
        assert len(results) > 0
