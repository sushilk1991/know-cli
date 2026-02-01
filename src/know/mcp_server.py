"""MCP (Model Context Protocol) server for know-cli.

Exposes know-cli's context engine, search, memory, and graph features
as MCP tools and resources.  Works with Claude Desktop, Cursor, and any
MCP-compatible client.

Transports:
  - stdio (default): ``know mcp serve``
  - SSE:            ``know mcp serve --sse --port 3000``

If the ``mcp`` package is not installed the CLI prints installation
instructions and exits gracefully.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from know.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Lazy MCP import so the rest of know-cli works without the mcp package.
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    pass


def _check_mcp():
    """Raise a friendly error when the mcp package is missing."""
    if not _MCP_AVAILABLE:
        msg = (
            "The 'mcp' package is required for the MCP server.\n"
            "Install it with:\n\n"
            "  pip install mcp\n\n"
            "Or install know-cli with the mcp extra:\n\n"
            "  pip install know-cli[mcp]\n"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Config helper — resolve the project root once
# ---------------------------------------------------------------------------

def _load_config():
    """Load the know-cli config for the current working directory."""
    from know.config import load_config
    return load_config()


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server(project_root: Optional[Path] = None) -> "FastMCP":
    """Create and configure the MCP server with all tools and resources.

    Args:
        project_root: Explicit project root.  When *None* the server
                      resolves the root from the current working directory
                      the same way the CLI does.
    """
    _check_mcp()

    server = FastMCP(
        "know-cli",
        instructions=(
            "know-cli provides intelligent code context for AI agents. "
            "Use get_context for smart, token-budgeted code retrieval. "
            "Use search_code for semantic search. "
            "Use remember/recall for cross-session memory."
        ),
    )

    # Shared lazy state -------------------------------------------------
    _state: dict[str, Any] = {}

    def _get_config():
        if "config" not in _state:
            if project_root:
                from know.config import Config, load_config
                try:
                    _state["config"] = load_config(project_root / ".know" / "config.yaml")
                except Exception:
                    _state["config"] = Config.create_default(project_root)
                    _state["config"].root = project_root
            else:
                _state["config"] = _load_config()
        return _state["config"]

    # ===================================================================
    # TOOLS
    # ===================================================================

    @server.tool()
    async def get_context(query: str, budget: int = 8000) -> str:
        """Get LLM-optimized context for a coding task.

        Returns relevant code, imports, tests, and memories within token
        budget.  This is know-cli's killer feature — smart, ranked context
        selection that minimises tokens while maximising relevance.

        Args:
            query: Natural language description of the task (e.g.
                   "help me fix the auth bug").
            budget: Maximum number of tokens to return (default 8000).
        """
        config = _get_config()

        from know.context_engine import ContextEngine

        engine = ContextEngine(config)
        result = engine.build_context(query, budget=budget)

        # Inject memories
        try:
            from know.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(config)
            mem_ctx = kb.get_relevant_context(query, max_tokens=min(500, budget // 10))
            if mem_ctx:
                result["memories_context"] = mem_ctx
        except Exception:
            pass

        return engine.format_agent_json(result)

    @server.tool()
    async def search_code(query: str, top_k: int = 5) -> str:
        """Search the codebase semantically.

        Returns relevant code snippets ranked by relevance using real
        embeddings (fastembed) with automatic text-match fallback.

        Args:
            query: What to search for (natural language or code terms).
            top_k: Maximum number of results to return (default 5).
        """
        config = _get_config()

        from know.semantic_search import SemanticSearcher

        searcher = SemanticSearcher(project_root=config.root)
        results = searcher.search_chunks(
            query, config.root, top_k, auto_index=True,
        )
        return json.dumps({"query": query, "results": results}, indent=2)

    @server.tool()
    async def remember(text: str, tags: str = "") -> str:
        """Store a piece of knowledge about the codebase for future sessions.

        Use this when you discover something important about the codebase
        that should persist across conversations (architectural decisions,
        gotchas, patterns, etc.).

        Args:
            text: The knowledge to store.
            tags: Optional comma-separated tags.
        """
        config = _get_config()

        from know.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(config)
        mem_id = kb.remember(text, source="mcp", tags=tags)

        # Track stats
        try:
            from know.stats import StatsTracker
            StatsTracker(config).record_remember(text, "mcp")
        except Exception:
            pass

        return json.dumps({"id": mem_id, "stored": True, "text": text})

    @server.tool()
    async def recall(query: str) -> str:
        """Recall stored knowledge about the codebase.

        Searches memories semantically — you don't need exact wording.

        Args:
            query: What to look up (e.g. "how does auth work?").
        """
        config = _get_config()

        from know.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(config)
        memories = kb.recall(query, top_k=5)

        results = [m.to_dict() for m in memories]

        # Track stats
        try:
            from know.stats import StatsTracker
            StatsTracker(config).record_recall(query, len(results), 0)
        except Exception:
            pass

        return json.dumps({"query": query, "results": results}, indent=2)

    @server.tool()
    async def explain_component(component: str, detailed: bool = False) -> str:
        """Explain a specific component or module using AI.

        Uses Anthropic Claude to generate an explanation of the requested
        component.  Requires ANTHROPIC_API_KEY to be set.

        Args:
            component: Name of the component/class/function to explain.
            detailed: Whether to generate a detailed explanation.
        """
        config = _get_config()

        from know.scanner import CodebaseScanner
        from know.ai import AISummarizer

        scanner = CodebaseScanner(config)
        matches = scanner.find_component(component)

        if not matches:
            return json.dumps({"error": f"Component '{component}' not found"})

        ai = AISummarizer(config)
        try:
            explanation = ai.explain_component(matches[0], detailed=detailed)
        except Exception as e:
            return json.dumps({"error": str(e)})

        # Auto-store as memory
        try:
            from know.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(config)
            kb.remember(
                f"[{component}] {explanation[:300]}",
                source="auto-explain",
                tags=component,
            )
        except Exception:
            pass

        return json.dumps({
            "component": component,
            "explanation": explanation,
        })

    @server.tool()
    async def show_graph(file_path: str) -> str:
        """Show import dependencies for a file.

        Returns what the file imports and what imports it.

        Args:
            file_path: Relative path to the file (e.g. "src/know/ai.py").
        """
        config = _get_config()

        from know.import_graph import ImportGraph
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig = ImportGraph(config)
        ig.build(structure["modules"])

        rel = str(Path(file_path).with_suffix("")).replace("/", ".").replace("\\", ".")

        return json.dumps({
            "module": rel,
            "imports": ig.imports_of(rel),
            "imported_by": ig.imported_by(rel),
            "formatted": ig.format_graph(rel),
        }, indent=2)

    # ===================================================================
    # RESOURCES
    # ===================================================================

    @server.resource("codebase://digest")
    async def get_digest() -> str:
        """LLM-optimized codebase summary.

        Returns a compact digest of the entire project: structure,
        key modules, statistics, and recent changes.
        """
        config = _get_config()

        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()

        modules = structure.get("modules", [])
        funcs = structure.get("function_count", 0)
        classes = structure.get("class_count", 0)

        lines = [
            f"# {config.project.name or config.root.name} — Codebase Digest",
            "",
            f"**Files:** {len(modules)} | **Functions:** {funcs} | **Classes:** {classes}",
            "",
            "## Modules",
            "",
        ]

        for mod in modules:
            name = mod["name"] if isinstance(mod, dict) else mod.name
            path = mod["path"] if isinstance(mod, dict) else str(mod.path)
            mod_funcs = mod.get("functions", []) if isinstance(mod, dict) else []
            mod_classes = mod.get("classes", []) if isinstance(mod, dict) else []
            lines.append(
                f"- **{name}** (`{path}`): "
                f"{len(mod_funcs)} functions, {len(mod_classes)} classes"
            )

        # Include memories summary
        try:
            from know.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(config)
            mem_count = kb.count()
            if mem_count > 0:
                lines.append("")
                lines.append(f"## Knowledge Base: {mem_count} memories stored")
        except Exception:
            pass

        return "\n".join(lines)

    @server.resource("codebase://structure")
    async def get_structure() -> str:
        """Project file structure and statistics."""
        config = _get_config()

        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()

        # Build tree view
        modules = structure.get("modules", [])
        tree_lines = [
            f"# {config.project.name or config.root.name} — Project Structure",
            "",
            f"Total files: {len(modules)}",
            f"Functions: {structure.get('function_count', 0)}",
            f"Classes: {structure.get('class_count', 0)}",
            "",
            "## File Tree",
            "",
        ]

        # Group by directory
        dirs: dict[str, list] = {}
        for mod in modules:
            path = mod["path"] if isinstance(mod, dict) else str(mod.path)
            parent = str(Path(path).parent)
            dirs.setdefault(parent, []).append(path)

        for d in sorted(dirs):
            tree_lines.append(f"📁 {d}/")
            for f in sorted(dirs[d]):
                tree_lines.append(f"  📄 {Path(f).name}")

        return "\n".join(tree_lines)

    @server.resource("codebase://memories")
    async def get_memories() -> str:
        """All stored knowledge/memories for this project."""
        config = _get_config()

        from know.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(config)
        memories = kb.list_all()

        if not memories:
            return "No memories stored yet. Use the `remember` tool to store knowledge."

        lines = [
            f"# Knowledge Base ({len(memories)} memories)",
            "",
        ]

        for m in memories:
            lines.append(f"- **#{m.id}** [{m.source}] {m.text}")
            if m.tags:
                lines.append(f"  Tags: {m.tags}")

        return "\n".join(lines)

    return server


# ---------------------------------------------------------------------------
# Runner — called from CLI
# ---------------------------------------------------------------------------

def run_server(*, sse: bool = False, port: int = 3000, project_root: Optional[Path] = None):
    """Start the MCP server.

    Args:
        sse: Use SSE transport instead of stdio.
        port: Port for SSE transport (default 3000).
        project_root: Explicit project root override.
    """
    _check_mcp()

    server = create_server(project_root=project_root)

    if sse:
        server.settings.port = port
        server.run(transport="sse")
    else:
        server.run(transport="stdio")


def print_config(project_root: Optional[Path] = None):
    """Print the Claude Desktop configuration snippet."""
    import shutil

    cwd = str(project_root or Path.cwd())
    know_path = shutil.which("know") or "know"

    config = {
        "mcpServers": {
            "know-cli": {
                "command": know_path,
                "args": ["mcp", "serve"],
                "cwd": cwd,
            }
        }
    }

    print(json.dumps(config, indent=2))
    print()
    print("# Add the above to your Claude Desktop config:")
    print("#   macOS: ~/Library/Application Support/Claude/claude_desktop_config.json")
    print("#   Linux: ~/.config/claude/claude_desktop_config.json")
    print("#   Windows: %APPDATA%\\Claude\\claude_desktop_config.json")
