"""Import graph: tracks dependencies between project modules.

Delegates storage to DaemonDB (the single source of truth).
Provides queries for 'what does X import?' and 'what imports X?'.

Uses fully-qualified module names to avoid false matches between
modules with the same leaf name (e.g., auth.config vs db.config).
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


class ImportGraph:
    """Builds and queries the import dependency graph for a project.

    Edges are stored in DaemonDB's imports table.
    All module names are stored as fully-qualified dotted paths
    (e.g., 'src.auth.config' not just 'config').
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        from know.daemon_db import DaemonDB
        self._db = DaemonDB(config.root)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
        # Build two lookup tables:
        #   fqn_set: set of all fully-qualified module names
        #   leaf_to_fqn: leaf_name -> list of FQNs (detects ambiguity)
        fqn_set: Set[str] = set()
        leaf_to_fqn: Dict[str, List[str]] = {}
        py_files: Dict[str, Path] = {}  # fqn -> abs path

        if modules:
            for m in modules:
                path_str = m["path"] if isinstance(m, dict) else str(m.path)
                name = m["name"] if isinstance(m, dict) else m.name
                fqn_set.add(name)
                leaf = name.split(".")[-1]
                leaf_to_fqn.setdefault(leaf, []).append(name)
                py_files[name] = self.root / path_str
        else:
            for py in self.root.rglob("*.py"):
                if any(p.startswith(".") or p in {"venv", "node_modules", "__pycache__", ".git"}
                       for p in py.parts):
                    continue
                try:
                    rel = py.relative_to(self.root)
                except ValueError:
                    continue
                name = str(rel.with_suffix("")).replace("/", ".")
                fqn_set.add(name)
                leaf = name.split(".")[-1]
                leaf_to_fqn.setdefault(leaf, []).append(name)
                py_files[name] = py

        def _resolve(import_name: str) -> Optional[str]:
            """Resolve an import string to a known FQN, or None."""
            # 1. Exact match against known FQNs
            if import_name in fqn_set:
                return import_name
            # 2. Try suffix match (e.g., 'auth.config' matches 'src.auth.config')
            for fqn in fqn_set:
                if fqn.endswith("." + import_name):
                    return fqn
            # 3. Leaf-name match ONLY if unambiguous
            leaf = import_name.split(".")[-1]
            candidates = leaf_to_fqn.get(leaf, [])
            if len(candidates) == 1:
                return candidates[0]
            # Ambiguous or not found
            return None

        # Collect edges per source module
        edges_by_source: Dict[str, List[Tuple[str, str]]] = {}

        for mod_name, abs_path in py_files.items():
            if not abs_path.exists() or not str(abs_path).endswith(".py"):
                continue
            try:
                source = abs_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            mod_edges: List[Tuple[str, str]] = []

            for node in ast.walk(tree):
                targets: List[Tuple[str, str]] = []  # (resolved_fqn, type)

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        resolved = _resolve(alias.name)
                        if resolved and resolved != mod_name:
                            targets.append((resolved, "import"))

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        resolved = _resolve(node.module)
                        if resolved and resolved != mod_name:
                            targets.append((resolved, "from"))

                for resolved, imp_type in targets:
                    mod_edges.append((resolved, imp_type))

            if mod_edges:
                edges_by_source[mod_name] = mod_edges

        # Persist via DaemonDB
        total_edges = 0
        for source_mod, targets in edges_by_source.items():
            self._db.set_imports(source_mod, targets)
            total_edges += len(targets)

        return total_edges

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def imports_of(self, module_name: str) -> List[str]:
        """What does *module_name* import? (outgoing edges)

        Accepts both fully-qualified names and leaf names.
        """
        results = self._db.get_imports_of(module_name)
        if not results:
            # Try suffix match via get_all_edges
            edges = self._db.get_all_edges()
            results = list({t for s, t in edges if s == module_name or s.endswith("." + module_name)})
        return results

    def imported_by(self, module_name: str) -> List[str]:
        """What modules import *module_name*? (incoming edges)

        Accepts both fully-qualified names and leaf names.
        """
        results = self._db.get_imported_by(module_name)
        if not results:
            edges = self._db.get_all_edges()
            results = list({s for s, t in edges if t == module_name or t.endswith("." + module_name)})
        return results

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Return all (source, target) edges."""
        return self._db.get_all_edges()

    def file_for_module(self, module_name: str) -> Optional[Path]:
        """Resolve a module name to a file path under project root."""
        rel = module_name.replace(".", "/") + ".py"
        candidate = self.root / rel
        if candidate.exists():
            return candidate
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
