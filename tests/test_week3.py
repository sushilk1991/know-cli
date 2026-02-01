"""Tests for Week 3: Cross-session memory, stats, status, CLI polish."""

import json
import os
import sqlite3
import sys
import tempfile
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
    # Create .know dir
    (tmp_path / ".know").mkdir()
    (tmp_path / ".know" / "cache").mkdir()
    
    # Create config
    config = Config.create_default(tmp_path)
    config.root = tmp_path
    config.project.name = "test-project"
    config.save(tmp_path / ".know" / "config.yaml")
    
    # Create some Python files
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
    
    return tmp_path, config


# ---------------------------------------------------------------------------
# Knowledge Base Tests
# ---------------------------------------------------------------------------

class TestKnowledgeBaseCRUD:
    """Test remember, recall, forget, list, export, import."""

    def test_remember_returns_id(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        mid = kb.remember("The auth system uses JWT")
        assert isinstance(mid, int)
        assert mid > 0

    def test_remember_stores_text(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        mid = kb.remember("Test memory text")
        mem = kb.get(mid)
        assert mem is not None
        assert mem.text == "Test memory text"
        assert mem.source == "manual"

    def test_remember_with_source_and_tags(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        mid = kb.remember("Auto insight", source="auto-explain", tags="auth,jwt")
        mem = kb.get(mid)
        assert mem.source == "auto-explain"
        assert mem.tags == "auth,jwt"

    def test_forget_deletes_memory(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        mid = kb.remember("Temporary note")
        assert kb.forget(mid) is True
        assert kb.get(mid) is None

    def test_forget_nonexistent_returns_false(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        assert kb.forget(99999) is False

    def test_list_all_returns_all_memories(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Memory 1")
        kb.remember("Memory 2")
        kb.remember("Memory 3")
        mems = kb.list_all()
        assert len(mems) == 3

    def test_list_all_filter_by_source(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Manual note", source="manual")
        kb.remember("Auto explain", source="auto-explain")
        kb.remember("Auto digest", source="auto-digest")
        manual = kb.list_all(source="manual")
        assert len(manual) == 1
        assert manual[0].text == "Manual note"

    def test_count(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        assert kb.count() == 0
        kb.remember("One")
        kb.remember("Two")
        assert kb.count() == 2
        assert kb.count(source="manual") == 2

    def test_recall_text_fallback(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("The auth system uses JWT tokens with Redis session store")
        kb.remember("Database migrations run via alembic")
        kb.remember("Payment webhook handler must be updated with payments module")
        
        results = kb._recall_text("how does JWT auth work?", top_k=3)
        assert len(results) > 0
        # First result should be the auth-related memory (has "JWT" and "auth")
        assert "auth" in results[0].text.lower() or "jwt" in results[0].text.lower()

    def test_recall_empty_returns_empty(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        results = kb.recall("anything")
        assert results == []

    def test_export_json(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Export test 1")
        kb.remember("Export test 2", source="auto-explain")
        
        exported = kb.export_json()
        data = json.loads(exported)
        assert len(data) == 2
        assert data[0]["text"] in ("Export test 1", "Export test 2")
        assert "id" in data[0]
        assert "source" in data[0]

    def test_import_json(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        
        import_data = json.dumps([
            {"text": "Imported memory 1", "source": "manual", "tags": "test"},
            {"text": "Imported memory 2", "source": "auto-digest"},
        ])
        count = kb.import_json(import_data)
        assert count == 2
        assert kb.count() == 2

    def test_export_import_roundtrip(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Round trip test", source="manual", tags="rt")
        
        exported = kb.export_json()
        
        # Create a new KB (same project)
        kb2 = KnowledgeBase(config)
        # Clear existing
        kb2.forget(1)
        assert kb2.count() == 0
        
        count = kb2.import_json(exported)
        assert count == 1
        mems = kb2.list_all()
        assert mems[0].text == "Round trip test"

    def test_project_scoped(self, tmp_project):
        """Memories are scoped to project root."""
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(config)
        kb.remember("Scoped memory")
        
        # Create a different project
        other_root = root / "other"
        other_root.mkdir()
        (other_root / ".know").mkdir()
        other_config = Config.create_default(other_root)
        other_config.root = other_root
        
        kb2 = KnowledgeBase(other_config)
        assert kb2.count() == 0  # different project, different memories

    def test_get_relevant_context(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Auth uses JWT tokens")
        kb.remember("Database uses PostgreSQL")
        
        ctx = kb.get_relevant_context("authentication", max_tokens=200)
        assert "Relevant Memories" in ctx
        assert "JWT" in ctx or "Auth" in ctx

    def test_get_relevant_context_empty(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        ctx = kb.get_relevant_context("anything")
        assert ctx == ""

    def test_db_created_in_know_dir(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        kb.remember("Test")
        assert (root / ".know" / "knowledge.db").exists()

    def test_memory_to_dict(self, tmp_project):
        root, config = tmp_project
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        mid = kb.remember("Dict test", tags="a,b")
        mem = kb.get(mid)
        d = mem.to_dict()
        assert d["text"] == "Dict test"
        assert d["tags"] == "a,b"
        assert d["source"] == "manual"
        assert "created_at" in d


# ---------------------------------------------------------------------------
# Stats Tracker Tests
# ---------------------------------------------------------------------------

class TestStatsTracker:
    """Test stats recording and summary."""

    def test_record_context(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_context("test query", 8000, 5000, 450)
        
        summary = tracker.get_summary()
        assert summary["context_queries"] == 1
        assert summary["context_avg_tokens"] == 5000
        assert summary["context_avg_ms"] == 450

    def test_record_search(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_search("search query", 10, 200)
        
        summary = tracker.get_summary()
        assert summary["search_queries"] == 1
        assert summary["search_avg_ms"] == 200

    def test_record_remember(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_remember("test memory", "manual")
        
        summary = tracker.get_summary()
        assert summary["remember_count"] == 1

    def test_record_recall(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_recall("recall query", 3, 100)
        
        summary = tracker.get_summary()
        assert summary["recall_count"] == 1

    def test_multiple_context_calls_average(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_context("q1", 8000, 4000, 300)
        tracker.record_context("q2", 8000, 6000, 500)
        
        summary = tracker.get_summary()
        assert summary["context_queries"] == 2
        assert summary["context_avg_tokens"] == 5000
        assert summary["context_avg_ms"] == 400

    def test_budget_utilization(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_context("q", 10000, 7300, 200)
        
        summary = tracker.get_summary()
        assert summary["context_budget_util"] == 73.0

    def test_empty_summary(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        summary = tracker.get_summary()
        assert summary["context_queries"] == 0
        assert summary["search_queries"] == 0
        assert summary["remember_count"] == 0
        assert summary["recall_count"] == 0

    def test_stats_db_in_know_dir(self, tmp_project):
        root, config = tmp_project
        from know.stats import StatsTracker
        tracker = StatsTracker(config)
        tracker.record_search("test", 0, 0)
        assert (root / ".know" / "stats.db").exists()


# ---------------------------------------------------------------------------
# CLI Command Tests
# ---------------------------------------------------------------------------

class TestCLICommands:
    """Test new CLI commands are registered and callable."""

    def test_remember_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "remember" in commands

    def test_recall_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "recall" in commands

    def test_forget_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "forget" in commands

    def test_memories_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "memories" in commands

    def test_stats_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "stats" in commands

    def test_status_command_exists(self):
        from know.cli import cli
        commands = cli.list_commands(None)
        assert "status" in commands


class TestCLIMemoryIntegration:
    """Test memory commands via Click test runner."""

    def test_remember_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "remember", "CLI test memory"
        ])
        assert result.exit_code == 0
        assert "Remembered" in result.output

    def test_remember_json_output(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "--json", "remember", "JSON test"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "id" in data
        assert data["text"] == "JSON test"

    def test_recall_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase
        
        # Store a memory first
        kb = KnowledgeBase(config)
        kb.remember("The scanner uses parallel processing")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "recall", "scanner"
        ])
        assert result.exit_code == 0
        assert "parallel" in result.output.lower() or "scanner" in result.output.lower()

    def test_forget_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(config)
        mid = kb.remember("To be forgotten")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "forget", str(mid)
        ])
        assert result.exit_code == 0
        assert "Forgot" in result.output

    def test_memories_list_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(config)
        kb.remember("Listed memory")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "memories", "list"
        ])
        assert result.exit_code == 0
        assert "Listed memory" in result.output

    def test_memories_export_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        from know.knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(config)
        kb.remember("Export CLI test")
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "memories", "export"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_memories_import_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        # Create import file
        import_file = root / "import.json"
        import_file.write_text(json.dumps([
            {"text": "Imported via CLI", "source": "manual"}
        ]))
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "memories", "import", str(import_file)
        ])
        assert result.exit_code == 0
        assert "Imported 1" in result.output


class TestCLIStatsStatus:
    """Test stats and status commands via Click runner."""

    def test_stats_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "stats"
        ])
        assert result.exit_code == 0
        assert "Statistics" in result.output

    def test_stats_json_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "--json", "stats"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "context_queries" in data

    def test_status_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "status"
        ])
        assert result.exit_code == 0
        assert "know-cli" in result.output

    def test_status_json_via_cli(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "--json", "status"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "version" in data
        from know import __version__
        assert data["version"] == __version__


# ---------------------------------------------------------------------------
# Agent JSON format with memories
# ---------------------------------------------------------------------------

class TestAgentFormatWithMemories:
    """Test that agent JSON output includes memories."""

    def test_agent_json_contains_memories_field(self, tmp_project):
        root, config = tmp_project
        from know.context_engine import ContextEngine
        
        engine = ContextEngine(config)
        result = engine.build_context("test query", budget=4000)
        result["memories_context"] = "## Memories\n- test memory"
        
        agent_json = engine.format_agent_json(result)
        data = json.loads(agent_json)
        assert "memories" in data
        assert "test memory" in data["memories"]

    def test_markdown_includes_memories_section(self, tmp_project):
        root, config = tmp_project
        from know.context_engine import ContextEngine
        
        engine = ContextEngine(config)
        result = engine.build_context("test query", budget=4000)
        result["memories_context"] = "- [manual] Auth uses JWT"
        
        md = engine.format_markdown(result)
        assert "Memories" in md
        assert "Auth uses JWT" in md


# ---------------------------------------------------------------------------
# STDIN piping support
# ---------------------------------------------------------------------------

class TestSTDINSupport:
    """Test that context command reads from STDIN."""

    def test_context_reads_stdin(self, tmp_project):
        root, config = tmp_project
        from click.testing import CliRunner
        from know.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            "--config", str(root / ".know" / "config.yaml"),
            "--quiet",
            "context",
            "--budget", "1000",
        ], input="test query via stdin\n")
        # Should not error about missing query
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Search --chunk flag
# ---------------------------------------------------------------------------

class TestSearchChunkFlag:
    """Test that search supports --chunk for function-level search."""

    def test_search_chunk_flag_exists(self):
        from know.cli import search
        param_names = [p.name for p in search.params]
        assert "chunk" in param_names


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_matches(self):
        from know import __version__
        # Version should be at least 0.2.2 (may be bumped in later weeks)
        parts = [int(x) for x in __version__.split(".")]
        assert parts >= [0, 2, 2]

    def test_pyproject_version(self):
        from know import __version__
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert f'version = "{__version__}"' in content
