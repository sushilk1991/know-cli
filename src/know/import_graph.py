"""Import graph: tracks dependencies between project modules.

Delegates storage to DaemonDB (the single source of truth).
Provides queries for 'what does X import?' and 'what imports X?'.

Uses fully-qualified module names to avoid false matches between
modules with the same leaf name (e.g., auth.config vs db.config).
"""

import ast
import fcntl
import posixpath
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING

from know.logger import get_logger
from know.path_filters import is_hard_excluded_path

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()

_BUILD_LOCKS: Dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()


@contextmanager
def _serialized_build(root: Path) -> Iterator[None]:
    """Serialize source snapshots and import-graph publication per project."""
    key = str(root.resolve())
    with _BUILD_LOCKS_GUARD:
        thread_lock = _BUILD_LOCKS.setdefault(key, threading.Lock())

    with thread_lock:
        lock_path = root / ".know" / "import-graph.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\s+from\s+|import\s+|export\s+.*?\s+from\s+|require\()\s*["']([^"']+)["']"""
)
_PY_IMPORT_RE = re.compile(r"^(?:from|import)\s+([\.A-Za-z_][\w\.]*)")


def _extract_import_target(import_stmt: str) -> Optional[str]:
    """Extract the imported module path from a raw import statement."""
    stmt = (import_stmt or "").strip()
    if not stmt:
        return None
    m = _JS_IMPORT_RE.search(stmt)
    if m:
        return m.group(1).strip()
    m = _PY_IMPORT_RE.match(stmt)
    if m:
        return m.group(1).strip()
    # Parser may already provide normalized bare module tokens like ".utils" or "pkg.mod".
    if re.match(r"^[\.A-Za-z_][\w\.]*$", stmt):
        return stmt
    return None


def _build_module_maps(modules: List[dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build helper maps: file->module and module->file."""
    file_to_module: Dict[str, str] = {}
    module_to_file: Dict[str, str] = {}
    for m in modules:
        file_path = m.get("path", "")
        module_name = m.get("name", "")
        if not file_path or not module_name:
            continue
        file_to_module[file_path] = module_name
        module_to_file[module_name] = file_path
    return file_to_module, module_to_file


def _resolve_relative_import(source_file: str, target: str, module_to_file: Dict[str, str]) -> Optional[str]:
    """Resolve JS/TS relative import to project file path."""
    src = Path(source_file)
    base = src.parent
    rel_target = target

    # Python relative imports use dotted level notation: .utils, ..core.types
    if target.startswith(".") and "/" not in target:
        level = len(target) - len(target.lstrip("."))
        remainder = target.lstrip(".")
        for _ in range(max(level - 1, 0)):
            base = base.parent
        rel_target = remainder.replace(".", "/") if remainder else ""

    candidates = []
    raw = base / rel_target
    candidates.extend([raw, raw.with_suffix(".ts"), raw.with_suffix(".tsx"),
                       raw.with_suffix(".js"), raw.with_suffix(".jsx"),
                       raw.with_suffix(".py")])
    candidates.extend([
        raw / "index.ts", raw / "index.tsx", raw / "index.js", raw / "index.jsx",
        raw / "__init__.py",
    ])
    module_files = {posixpath.normpath(str(v).replace("\\", "/")) for v in module_to_file.values()}
    for c in candidates:
        normalized = posixpath.normpath(str(c).replace("\\", "/"))
        if normalized in module_files:
            return normalized
    return None


def _resolve_absolute_import(target: str, module_to_file: Dict[str, str]) -> Optional[str]:
    """Resolve a module import against indexed modules."""
    if target in module_to_file:
        return module_to_file[target]
    dotted = target.replace("/", ".")
    if dotted in module_to_file:
        return module_to_file[dotted]
    if f"{dotted}.index" in module_to_file:
        return module_to_file[f"{dotted}.index"]
    for mod, fp in module_to_file.items():
        if mod.endswith("." + dotted) or mod.endswith("." + target):
            return fp
        if mod.endswith("." + dotted + ".index") or mod.endswith("." + target + ".index"):
            return fp
    return None


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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        with _serialized_build(self.root):
            return self._build_locked(modules)

    def _build_locked(self, modules: Optional[list]) -> int:
        # Build two lookup tables:
        #   fqn_set: set of all fully-qualified module names
        #   leaf_to_fqn: leaf_name -> list of FQNs (detects ambiguity)
        fqn_set: Set[str] = set()
        leaf_to_fqn: Dict[str, List[str]] = {}
        py_files: Dict[str, Path] = {}  # fqn -> abs path

        if modules is not None:
            for m in modules:
                path_str = m["path"] if isinstance(m, dict) else str(m.path)
                name = m["name"] if isinstance(m, dict) else m.name
                fqn_set.add(name)
                leaf = name.split(".")[-1]
                leaf_to_fqn.setdefault(leaf, []).append(name)
                py_files[name] = self.root / path_str

            # The scanner omits files that fail parsing, but populate_index
            # deliberately retains their last-known-good DB rows. Include that
            # authoritative indexed scope so a transient full-scan failure does
            # not make the import graph treat a still-present module as deleted.
            for path_str in self._db.list_indexed_files():
                path = Path(path_str)
                if path.suffix != ".py":
                    continue
                name = str(path.with_suffix("")).replace("\\", "/").replace("/", ".")
                if name in fqn_set:
                    continue
                fqn_set.add(name)
                leaf_to_fqn.setdefault(name.split(".")[-1], []).append(name)
                py_files[name] = self.root / path
        else:
            for py in self.root.rglob("*.py"):
                try:
                    rel = py.relative_to(self.root)
                    py.resolve().relative_to(self.root.resolve())
                except ValueError:
                    continue
                if is_hard_excluded_path(rel):
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
        failed_sources: Set[str] = set()

        for mod_name, abs_path in py_files.items():
            if not abs_path.exists() or not str(abs_path).endswith(".py"):
                continue
            try:
                source = abs_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except (OSError, SyntaxError, UnicodeDecodeError):
                failed_sources.add(mod_name)
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
                    candidates: List[str] = []
                    if node.level:
                        package = mod_name.split(".")[:-1]
                        ascend = node.level - 1
                        if ascend <= len(package):
                            base = package[:len(package) - ascend] if ascend else package
                            if node.module:
                                candidates.append(".".join(base + [node.module]))
                            else:
                                candidates.extend(
                                    ".".join(base + [alias.name])
                                    for alias in node.names
                                )
                    elif node.module:
                        candidates.append(node.module)

                    for candidate in candidates:
                        resolved = _resolve(candidate)
                        if resolved and resolved != mod_name:
                            targets.append((resolved, "from"))

                for resolved, imp_type in targets:
                    mod_edges.append((resolved, imp_type))

            # Repeated imports should not inflate counts or perform redundant
            # writes, while preserving the first import kind deterministically.
            edges_by_source[mod_name] = list(dict.fromkeys(mod_edges))

        # Successfully parsed sources are authoritative even when their import
        # set is empty. Failed sources retain their last-good outgoing edges;
        # sources absent from the authoritative module set are removed.
        existing_sources = {source for source, _target in self._db.get_all_edges()}
        total_edges = 0
        with self._db.batch():
            for stale_source in existing_sources - fqn_set:
                self._db.set_imports(stale_source, [])
            for source_mod, targets in edges_by_source.items():
                self._db.set_imports(source_mod, targets)
                total_edges += len(targets)

        if failed_sources:
            logger.debug(
                "Preserved last-good imports for %s unparseable modules",
                len(failed_sources),
            )

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

    @staticmethod
    def related_files_from_modules(file_path: str, modules: List[dict]) -> Tuple[List[str], List[str]]:
        """Language-agnostic related files based on parser-extracted imports."""
        target_path = file_path.replace("\\", "/")
        file_to_module, module_to_file = _build_module_maps(modules)
        if target_path not in file_to_module:
            # Try loose suffix match
            for fp in file_to_module:
                if fp.endswith(target_path):
                    target_path = fp
                    break

        outgoing: Set[str] = set()
        incoming: Set[str] = set()

        for m in modules:
            src_file = m.get("path", "").replace("\\", "/")
            imports = m.get("imports", []) or []
            if not src_file:
                continue
            for imp in imports:
                target = _extract_import_target(imp)
                if not target:
                    continue
                resolved_fp = None
                if target.startswith("."):
                    resolved_fp = _resolve_relative_import(src_file, target, module_to_file)
                else:
                    resolved_fp = _resolve_absolute_import(target, module_to_file)
                if not resolved_fp:
                    continue
                resolved_fp = posixpath.normpath(str(resolved_fp).replace("\\", "/"))
                if src_file == target_path:
                    outgoing.add(resolved_fp)
                if resolved_fp == target_path:
                    incoming.add(src_file)

        return sorted(outgoing), sorted(incoming)
