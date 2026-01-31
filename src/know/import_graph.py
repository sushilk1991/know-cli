"""Import graph: tracks dependencies between project modules.

Stores import relationships as an adjacency list in index.db.
Provides queries for 'what does X import?' and 'what imports X?'.
"""

import ast
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import logging

from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


class ImportGraph:
    """Builds and queries the import dependency graph for a project.
    
    Edges are stored in SQLite alongside the existing index.db.
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.db_path = config.root / ".know" / "cache" / "index.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_table(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS import_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                import_type TEXT NOT NULL DEFAULT 'import',
                PRIMARY KEY (source, target)
            );
            CREATE INDEX IF NOT EXISTS idx_import_source ON import_edges(source);
            CREATE INDEX IF NOT EXISTS idx_import_target ON import_edges(target);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Build graph from codebase
    # ------------------------------------------------------------------
    def build(self, modules: Optional[list] = None) -> int:
        """Build the full import graph from the project modules.
        
        Args:
            modules: Optional list of ModuleInfo dicts (from scanner).
                     If None, scans Python files under self.root.
        
        Returns:
            Number of edges inserted.
        """
        # Collect all known module short-names -> full-name mapping
        known_modules: Dict[str, str] = {}
        py_files: Dict[str, Path] = {}  # short_name -> abs path

        if modules:
            for m in modules:
                path_str = m["path"] if isinstance(m, dict) else str(m.path)
                name = m["name"] if isinstance(m, dict) else m.name
                short = name.split(".")[-1]
                known_modules[short] = name
                known_modules[name] = name
                py_files[name] = self.root / path_str
        else:
            # Fall back: discover .py files
            for py in self.root.rglob("*.py"):
                if any(p.startswith(".") or p in {"venv", "node_modules", "__pycache__", ".git"}
                       for p in py.parts):
                    continue
                try:
                    rel = py.relative_to(self.root)
                except ValueError:
                    continue
                name = str(rel.with_suffix("")).replace("/", ".")
                short = name.split(".")[-1]
                known_modules[short] = name
                known_modules[name] = name
                py_files[name] = py

        edges: List[Tuple[str, str, str]] = []

        for mod_name, abs_path in py_files.items():
            if not abs_path.exists() or not str(abs_path).endswith(".py"):
                continue
            try:
                source = abs_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                targets: List[Tuple[str, str]] = []  # (resolved_name, type)

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        t = alias.name.split(".")[-1]
                        if t in known_modules and known_modules[t] != mod_name:
                            targets.append((known_modules[t], "import"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        t = node.module.split(".")[-1]
                        if t in known_modules and known_modules[t] != mod_name:
                            targets.append((known_modules[t], "from"))

                for resolved, imp_type in targets:
                    edges.append((mod_name, resolved, imp_type))

        # Persist
        conn = self._get_conn()
        conn.execute("DELETE FROM import_edges")
        if edges:
            conn.executemany(
                "INSERT OR REPLACE INTO import_edges (source, target, import_type) VALUES (?, ?, ?)",
                edges,
            )
        conn.commit()
        return len(edges)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def imports_of(self, module_name: str) -> List[str]:
        """What does *module_name* import? (outgoing edges)"""
        conn = self._get_conn()
        short = module_name.split(".")[-1]
        rows = conn.execute(
            "SELECT target FROM import_edges WHERE source = ? OR source LIKE ?",
            (module_name, f"%.{short}"),
        ).fetchall()
        return list({r[0] for r in rows})

    def imported_by(self, module_name: str) -> List[str]:
        """What modules import *module_name*? (incoming edges)"""
        conn = self._get_conn()
        short = module_name.split(".")[-1]
        rows = conn.execute(
            "SELECT source FROM import_edges WHERE target = ? OR target LIKE ?",
            (module_name, f"%.{short}"),
        ).fetchall()
        return list({r[0] for r in rows})

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Return all (source, target) edges."""
        conn = self._get_conn()
        rows = conn.execute("SELECT source, target FROM import_edges").fetchall()
        return [(r[0], r[1]) for r in rows]

    def file_for_module(self, module_name: str) -> Optional[Path]:
        """Resolve a module name to a file path under project root."""
        # Try direct path conversion
        rel = module_name.replace(".", "/") + ".py"
        candidate = self.root / rel
        if candidate.exists():
            return candidate
        # Try short name search
        short = module_name.split(".")[-1]
        for py in self.root.rglob(f"{short}.py"):
            if not any(p.startswith(".") or p in {"venv", "__pycache__"}
                       for p in py.relative_to(self.root).parts):
                return py
        return None

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    def format_graph(self, module_name: str) -> str:
        """Human-readable graph display for a single module."""
        imports = self.imports_of(module_name)
        imported = self.imported_by(module_name)

        lines = [f"# Import graph for: {module_name}", ""]

        if imports:
            lines.append("## Imports (dependencies)")
            for m in sorted(imports):
                lines.append(f"  → {m}")
        else:
            lines.append("## Imports: (none)")

        lines.append("")

        if imported:
            lines.append("## Imported by (dependents)")
            for m in sorted(imported):
                lines.append(f"  ← {m}")
        else:
            lines.append("## Imported by: (none)")

        return "\n".join(lines)
