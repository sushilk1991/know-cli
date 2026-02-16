"""Context engine: assembles LLM-optimized context bundles.

The killer feature.  Given a natural-language query and a token budget,
produces a single Markdown (or JSON) blob that contains the most relevant
code, its dependencies, related tests, and project overview — optimally
packed to fit the budget.

Architecture (v3 — DaemonDB pivot):
  1. Query DaemonDB FTS5 with BM25F field weighting
  2. Apply file category demotion (test/vendor/generated)
  3. Apply relevance floor (return under budget, not noise)
  4. Bundle structural metadata (imports, imported_by, test_file)
  5. Token budgeting — greedily fill:
       60 % highest-relevance code chunks (with metadata)
       15 % dependency signatures
       15 % file summaries
       10 % project overview

  Fallback: direct DaemonDB if daemon socket unavailable.
  Legacy: filesystem scan only with --legacy flag.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from know.token_counter import count_tokens, truncate_to_budget, format_budget
from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def get_direct_db(config: "Config"):
    """Get a direct DaemonDB connection (no daemon socket)."""
    from know.daemon_db import DaemonDB
    return DaemonDB(config.root)


# ---------------------------------------------------------------------------
# Embedding model — uses centralized manager
# ---------------------------------------------------------------------------
from know.embeddings import get_model as _get_cached_embedding_model


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single piece of code (function, class, or module-level)."""
    file_path: str          # relative to project root
    name: str               # e.g. "create_token" or "AuthMiddleware"
    chunk_type: str         # "function" | "class" | "module" | "method"
    line_start: int
    line_end: int
    body: str               # full source text
    signature: str = ""     # first line / declaration
    docstring: str = ""
    score: float = 0.0      # semantic relevance 0-1
    recency_boost: float = 0.0
    tokens: int = 0

    @property
    def qualified_name(self) -> str:
        return f"{self.file_path}:{self.name}"

    def header(self) -> str:
        loc = f"lines {self.line_start}-{self.line_end}"
        return f"{self.file_path}:{self.name} ({loc})"


# ---------------------------------------------------------------------------
# Chunk extraction
# ---------------------------------------------------------------------------

def extract_chunks_from_file(file_path: Path, project_root: Path) -> List[CodeChunk]:
    """Extract function/class-level chunks from a Python file using AST.
    
    For non-Python files, returns a single file-level chunk.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(project_root))

    if not file_path.suffix == ".py":
        # Non-Python: whole-file chunk
        tokens = count_tokens(content)
        return [CodeChunk(
            file_path=rel_path,
            name=file_path.stem,
            chunk_type="module",
            line_start=1,
            line_end=content.count("\n") + 1,
            body=content,
            tokens=tokens,
        )]

    try:
        tree = ast.parse(content)
    except SyntaxError:
        tokens = count_tokens(content)
        return [CodeChunk(
            file_path=rel_path, name=file_path.stem, chunk_type="module",
            line_start=1, line_end=content.count("\n") + 1,
            body=content, tokens=tokens,
        )]

    lines = content.splitlines(keepends=True)
    chunks: List[CodeChunk] = []

    # Module-level summary: imports + top docstring
    module_doc = ast.get_docstring(tree) or ""
    import_lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            import_lines.extend(lines[start:end])
    
    module_body = "".join(import_lines)
    if module_doc:
        module_body = f'"""{module_doc}"""\n\n' + module_body
    if module_body.strip():
        chunks.append(CodeChunk(
            file_path=rel_path,
            name=f"{file_path.stem} (module)",
            chunk_type="module",
            line_start=1,
            line_end=max(1, len(import_lines)),
            body=module_body.strip(),
            docstring=module_doc,
            tokens=count_tokens(module_body),
        ))

    # Top-level functions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk = _ast_node_to_chunk(node, lines, rel_path)
            if chunk:
                chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):
            # Class-level chunk (signature + docstring + method signatures)
            cls_chunk = _class_to_chunk(node, lines, rel_path)
            if cls_chunk:
                chunks.append(cls_chunk)

            # Individual method chunks
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = _ast_node_to_chunk(
                        item, lines, rel_path,
                        parent_class=node.name,
                        is_method=True,
                    )
                    if chunk:
                        chunks.append(chunk)

    return chunks


def _ast_node_to_chunk(
    node: ast.AST,
    lines: list,
    rel_path: str,
    parent_class: str = "",
    is_method: bool = False,
) -> Optional[CodeChunk]:
    """Convert an AST function/async-function node to a CodeChunk."""
    start = node.lineno
    end = node.end_lineno or start
    body_text = "".join(lines[start - 1 : end]).rstrip()
    if not body_text.strip():
        return None

    name = node.name
    if parent_class:
        name = f"{parent_class}.{name}"

    sig_line = lines[start - 1].strip() if start - 1 < len(lines) else ""
    docstring = ast.get_docstring(node) or ""

    return CodeChunk(
        file_path=rel_path,
        name=name,
        chunk_type="method" if is_method else "function",
        line_start=start,
        line_end=end,
        body=body_text,
        signature=sig_line,
        docstring=docstring,
        tokens=count_tokens(body_text),
    )


def _class_to_chunk(node: ast.ClassDef, lines: list, rel_path: str) -> Optional[CodeChunk]:
    """Create a summary chunk for a class (signature + docstring + method sigs)."""
    start = node.lineno
    end = node.end_lineno or start
    full_body = "".join(lines[start - 1 : end]).rstrip()

    # Build a compact summary: class declaration + docstring + method stubs
    sig_line = lines[start - 1].strip() if start - 1 < len(lines) else ""
    docstring = ast.get_docstring(node) or ""

    summary_lines = [sig_line]
    if docstring:
        summary_lines.append(f'    """{docstring}"""')
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mline = lines[item.lineno - 1].strip() if item.lineno - 1 < len(lines) else ""
            summary_lines.append(f"    {mline}")
            mdoc = ast.get_docstring(item)
            if mdoc:
                summary_lines.append(f'        """{mdoc}"""')
            summary_lines.append("        ...")

    summary = "\n".join(summary_lines)
    return CodeChunk(
        file_path=rel_path,
        name=node.name,
        chunk_type="class",
        line_start=start,
        line_end=end,
        body=full_body,
        signature=sig_line,
        docstring=docstring,
        tokens=count_tokens(summary),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_signatures(file_path: Path) -> str:
    """Extract just the function/class signatures from a Python file."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except Exception:
        return ""

    lines = content.splitlines()
    sigs: List[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
            doc = ast.get_docstring(node) or ""
            sigs.append(sig)
            if doc:
                sigs.append(f'    """{doc}"""')
        elif isinstance(node, ast.ClassDef):
            sig = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
            doc = ast.get_docstring(node) or ""
            sigs.append(sig)
            if doc:
                sigs.append(f'    """{doc}"""')
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    msig = lines[item.lineno - 1].strip() if item.lineno - 1 < len(lines) else ""
                    sigs.append(f"    {msig}")

    return "\n".join(sigs)


# Batch git recency cache to avoid repeated subprocess calls
_GIT_RECENCY_CACHE: Dict[str, Dict[str, float]] = {}
_GIT_RECENCY_CACHE_TIME: Dict[str, float] = {}
_GIT_RECENCY_CACHE_TTL = 300  # 5 minute TTL

def _get_batch_file_recency(root: Path, rel_paths: List[str]) -> Dict[str, float]:
    """Get recency scores for multiple files in a single git command batch.
    
    Uses a file-based cache to avoid repeated git calls.
    """
    import time
    
    # Check if we have a valid cached result
    cache_key = str(root)
    now = time.time()
    
    if cache_key in _GIT_RECENCY_CACHE:
        cache_age = now - _GIT_RECENCY_CACHE_TIME.get(cache_key, 0)
        if cache_age < _GIT_RECENCY_CACHE_TTL:
            return {p: _GIT_RECENCY_CACHE[cache_key].get(p, 0.0) for p in rel_paths}
    
    # Build fresh cache
    scores: Dict[str, float] = {}
    
    try:
        # Get recent commits with file changes in one command
        result = subprocess.run(
            ["git", "log", "--name-only", "--format=%ct", "-30", "--pretty=format:%ct"],
            capture_output=True, text=True, cwd=root, timeout=10,
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                current_time = None
                for line in output.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Check if this is a timestamp
                    if line.isdigit():
                        current_time = int(line)
                    elif current_time and line in rel_paths:
                        age_days = (time.time() - current_time) / 86400
                        score = max(0.0, 1.0 - age_days / 30.0)
                        if line not in scores or score > scores[line]:
                            scores[line] = score
        
        # Store in cache
        _GIT_RECENCY_CACHE[cache_key] = scores
        _GIT_RECENCY_CACHE_TIME[cache_key] = now
        
    except Exception:
        pass
    
    return {p: scores.get(p, 0.0) for p in rel_paths}


def _git_file_recency(root: Path, rel_path: str) -> float:
    """Return a 0-1 recency score based on git log.
    
    1.0 = modified in last day, decays linearly over 30 days.
    Returns 0.0 if git is unavailable.
    
    Note: Single-file version - use batch version for multiple files.
    """
    scores = _get_batch_file_recency(root, [rel_path])
    return scores.get(rel_path, 0.0)


def _find_test_files(root: Path, source_path: str) -> List[Path]:
    """Find test files that likely correspond to a source file."""
    stem = Path(source_path).stem
    candidates = [
        f"test_{stem}.py",
        f"tests/test_{stem}.py",
        f"tests/{stem}_test.py",
    ]

    results = []
    for pattern in [f"**/test_{stem}.py", f"**/{stem}_test.py"]:
        for p in root.rglob(pattern):
            if not any(part in {"venv", ".git", "node_modules", "__pycache__"}
                       for part in p.parts):
                results.append(p)
    return results


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ContextEngine:
    """Assembles an optimally-packed context bundle for LLM consumption.

    v3 architecture: thin orchestrator that delegates to DaemonDB for
    retrieval, ranking functions for scoring, and formatters for output.
    """

    # Budget allocation percentages (v3: more code, less overhead)
    ALLOC_CODE = 0.60
    ALLOC_IMPORTS = 0.15
    ALLOC_SUMMARIES = 0.15
    ALLOC_OVERVIEW = 0.10

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_context(
        self,
        query: str,
        budget: int = 8000,
        include_tests: bool = True,
        include_imports: bool = True,
        legacy: bool = False,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build context bundle for *query* within *budget* tokens.

        Uses DaemonDB FTS5 search with BM25F field weighting.
        Falls back to direct DaemonDB if daemon socket unavailable.
        Set legacy=True to force old filesystem scan (debugging only).

        Returns a dict with keys:
            query, budget, used_tokens,
            code_chunks, dependency_chunks, test_chunks,
            summary_chunks, overview, warnings, indexing_status,
            index_stats, confidence
        """
        if legacy:
            return self._build_context_legacy(query, budget, include_tests, include_imports)

        return self._build_context_v3(
            query, budget, include_tests, include_imports,
            include_patterns, exclude_patterns, chunk_types,
        )

    def _build_context_v3(
        self,
        query: str,
        budget: int,
        include_tests: bool,
        include_imports: bool,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """v3 context building via DaemonDB."""
        # Get DaemonDB (direct connection, no daemon socket needed)
        db = get_direct_db(self.config)
        try:
            return self._build_context_v3_inner(
                db, query, budget, include_tests, include_imports,
                include_patterns, exclude_patterns, chunk_types,
            )
        finally:
            db.close()

    def _build_context_v3_inner(
        self, db, query: str, budget: int,
        include_tests: bool, include_imports: bool,
        include_patterns, exclude_patterns, chunk_types,
    ) -> Dict[str, Any]:
        """Inner v3 pipeline (db connection managed by caller)."""
        from know.file_categories import apply_category_demotion
        from know.ranking import apply_relevance_floor

        warnings: List[str] = []

        # Check indexing status
        stats = db.get_stats()
        indexing_status = "complete" if stats["files"] > 0 else "indexing"

        if stats["files"] == 0:
            # No index yet — populate DB inline (first-use indexing)
            logger.debug("DaemonDB empty, running inline index population")
            try:
                from know.daemon import populate_index
                indexed, _ = populate_index(self.config.root, self.config, db)
                logger.debug(f"Inline indexing complete: {indexed} files")
                stats = db.get_stats()
                indexing_status = "complete" if stats["files"] > 0 else "indexing"
            except Exception as e:
                logger.warning(f"Inline indexing failed, falling back to legacy: {e}")

            if stats["files"] == 0:
                # Still empty after indexing attempt — fall back to legacy
                logger.debug("DaemonDB still empty, falling back to legacy")
                result = self._build_context_legacy(
                    query, budget, include_tests, include_imports,
                )
                result["indexing_status"] = indexing_status
                result["index_stats"] = stats
                return result

        # Step 1: FTS5 search with BM25F weights
        raw_results = db.search_chunks(query, limit=100)

        if not raw_results:
            # Zero-result intelligence
            nearest = db.get_nearest_terms(query, limit=5)
            matching_files = db.get_matching_file_names(query, limit=5)
            result = self._empty_result(query, budget, warnings)
            result["indexing_status"] = indexing_status
            result["index_stats"] = stats
            result["nearest_terms"] = nearest
            result["file_names_matching"] = matching_files
            return result

        # Step 2: Apply file category demotion
        raw_results = apply_category_demotion(raw_results, query)

        # Step 3: Apply file path filtering
        if include_patterns or exclude_patterns or chunk_types:
            raw_results = self._apply_filters(
                raw_results, include_patterns, exclude_patterns, chunk_types,
            )

        # Step 4: Re-sort by demoted scores
        raw_results.sort(key=lambda c: c.get("score", 0), reverse=True)

        # Step 5: Apply relevance floor (return under budget, not noise)
        raw_results = apply_relevance_floor(raw_results)

        # Step 6: compute sub-budgets
        budget_code = int(budget * self.ALLOC_CODE)
        budget_imports = int(budget * self.ALLOC_IMPORTS) if include_imports else 0
        budget_summaries = int(budget * self.ALLOC_SUMMARIES)
        budget_overview = int(budget * self.ALLOC_OVERVIEW)
        if not include_imports:
            budget_code += int(budget * self.ALLOC_IMPORTS)

        # Step 7: greedily fill code budget using pre-computed token_count
        code_chunks: List[Dict] = []
        code_used = 0
        seen_files: set = set()
        for chunk in raw_results:
            tokens = chunk.get("token_count", 0)
            if tokens == 0:
                tokens = count_tokens(chunk.get("body", ""))
            if code_used + tokens > budget_code:
                continue
            code_chunks.append(chunk)
            code_used += tokens
            seen_files.add(chunk["file_path"])

        # Step 8: Bundle structural metadata (batch import lookups)
        if seen_files:
            self._bundle_metadata(db, code_chunks, seen_files)

        # Step 9: Group chunks by file
        code_chunks = self._group_by_file(code_chunks)

        # Step 10: Dependency signatures
        dep_chunks: List[Dict] = []
        dep_used = 0
        if include_imports and seen_files:
            dep_chunks, dep_used = self._get_dependency_sigs(
                db, seen_files, budget_imports,
            )

        # Step 11: File summaries
        summary_chunks, summary_used = self._build_summaries_from_db(
            db, seen_files, budget_summaries,
        )

        # Step 12: Overview
        overview = self._project_overview(budget_overview)
        overview_tokens = count_tokens(overview)

        total_used = code_used + dep_used + summary_used + overview_tokens

        # Convert to CodeChunk objects for backward compatibility
        code_objs = [self._dict_to_chunk(c) for c in code_chunks]
        dep_objs = [self._dict_to_chunk(c) for c in dep_chunks]
        summary_objs = [self._dict_to_chunk(c) for c in summary_chunks]

        confidence = round(total_used / budget, 2) if budget > 0 else 0

        return {
            "query": query,
            "budget": budget,
            "used_tokens": total_used,
            "budget_display": format_budget(total_used, budget),
            "code_chunks": code_objs,
            "dependency_chunks": dep_objs,
            "test_chunks": [],
            "summary_chunks": summary_objs,
            "overview": overview,
            "warnings": warnings,
            "indexing_status": indexing_status,
            "index_stats": stats,
            "confidence": confidence,
        }

    def _apply_filters(
        self,
        results: List[Dict],
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
        chunk_types: Optional[List[str]],
    ) -> List[Dict]:
        """Apply file path and chunk type filters."""
        import fnmatch
        filtered = results

        if include_patterns:
            filtered = [
                r for r in filtered
                if any(fnmatch.fnmatch(r["file_path"], p) for p in include_patterns)
            ]
        if exclude_patterns:
            filtered = [
                r for r in filtered
                if not any(fnmatch.fnmatch(r["file_path"], p) for p in exclude_patterns)
            ]
        if chunk_types:
            filtered = [
                r for r in filtered
                if r.get("chunk_type", "") in chunk_types
            ]
        return filtered

    def _bundle_metadata(self, db, chunks: List[Dict], seen_files: set):
        """Add imports, imported_by, test_file to each chunk."""
        # Convert file paths to module names for import lookup
        file_to_module = {}
        for fp in seen_files:
            mod = fp.replace("/", ".").replace("\\", ".")
            if mod.endswith(".py"):
                mod = mod[:-3]
            file_to_module[fp] = mod

        modules = list(file_to_module.values())

        # Batch lookup
        imports_map = db.get_imports_batch(modules)
        imported_by_map = db.get_imported_by_batch(modules)

        for chunk in chunks:
            fp = chunk["file_path"]
            mod = file_to_module.get(fp, "")

            chunk["imports"] = imports_map.get(mod, [])
            chunk["imported_by"] = imported_by_map.get(mod, [])

            # Test file association: check if a test file exists in imported_by
            test_file = None
            for imp_mod in imported_by_map.get(mod, []):
                leaf = imp_mod.split(".")[-1]
                if leaf.startswith("test_") or leaf.endswith("_test"):
                    test_file = imp_mod.replace(".", "/") + ".py"
                    break
            chunk["test_file"] = test_file

    def _group_by_file(self, chunks: List[Dict]) -> List[Dict]:
        """Group chunks by file path, preserving relevance order within groups."""
        if not chunks:
            return chunks
        # Preserve order of first appearance of each file
        file_order = list(dict.fromkeys(c["file_path"] for c in chunks))
        by_file: Dict[str, List[Dict]] = {}
        for c in chunks:
            by_file.setdefault(c["file_path"], []).append(c)
        result = []
        for fp in file_order:
            # Sort within file by start_line
            group = by_file[fp]
            group.sort(key=lambda c: c.get("start_line", 0))
            result.extend(group)
        return result

    def _get_dependency_sigs(
        self, db, seen_files: set, budget: int,
    ) -> Tuple[List[Dict], int]:
        """Get dependency signatures from DaemonDB."""
        seen_modules = [
            fp.replace("/", ".").replace("\\", ".").removesuffix(".py")
            for fp in seen_files
        ]
        imports_map = db.get_imports_batch(seen_modules)
        modules = set()
        for targets in imports_map.values():
            modules.update(targets)

        dep_chunks: List[Dict] = []
        used = 0
        for mod in modules:
            fp = mod.replace(".", "/") + ".py"
            if fp in seen_files:
                continue
            sigs = db.get_signatures(fp)
            if not sigs:
                continue
            sig_text = "\n".join(s.get("signature", "") for s in sigs if s.get("signature"))
            if not sig_text:
                continue
            tokens = count_tokens(sig_text)
            if used + tokens > budget:
                sig_text = truncate_to_budget(sig_text, budget - used)
                tokens = count_tokens(sig_text)
                if used + tokens > budget:
                    continue
            dep_chunks.append({
                "file_path": fp,
                "chunk_name": f"{mod.split('.')[-1]} (signatures)",
                "chunk_type": "module",
                "start_line": 1,
                "end_line": 1,
                "signature": "# signatures only",
                "body": sig_text,
                "token_count": tokens,
            })
            used += tokens
        return dep_chunks, used

    def _build_summaries_from_db(
        self, db, seen_files: set, budget: int,
    ) -> Tuple[List[Dict], int]:
        """Build file-level summaries from DaemonDB chunk data."""
        summaries: List[Dict] = []
        used = 0
        for fp in seen_files:
            chunks = db.get_chunks_for_file(fp)
            if not chunks:
                continue
            names = [f"{c['chunk_type']} {c['chunk_name']}" for c in chunks[:10]]
            summary = f"# {fp}\nContains: {', '.join(names)}\n"
            tokens = count_tokens(summary)
            if used + tokens > budget:
                continue
            summaries.append({
                "file_path": fp,
                "chunk_name": f"{Path(fp).stem} (summary)",
                "chunk_type": "module",
                "start_line": 1,
                "end_line": 1,
                "signature": "",
                "body": summary,
                "token_count": tokens,
            })
            used += tokens
        return summaries, used

    @staticmethod
    def _dict_to_chunk(d: Dict) -> CodeChunk:
        """Convert a DB dict to a CodeChunk for backward compatibility."""
        chunk = CodeChunk(
            file_path=d.get("file_path", ""),
            name=d.get("chunk_name", ""),
            chunk_type=d.get("chunk_type", "module"),
            line_start=d.get("start_line", 0),
            line_end=d.get("end_line", 0),
            body=d.get("body", ""),
            signature=d.get("signature", ""),
            score=d.get("score", 0.0),
            tokens=d.get("token_count", 0),
        )
        # Carry structural metadata from v3 pipeline
        meta = {}
        for key in ("imports", "imported_by", "test_file"):
            if key in d:
                meta[key] = d[key]
        if meta:
            chunk._metadata = meta
        return chunk

    # ------------------------------------------------------------------
    # Legacy filesystem-based context building
    # ------------------------------------------------------------------
    def _build_context_legacy(
        self, query: str, budget: int,
        include_tests: bool, include_imports: bool,
    ) -> Dict[str, Any]:
        """Old filesystem-scan based context building (for debugging)."""
        budget_code = int(budget * 0.40)
        budget_imports = int(budget * 0.30) if include_imports else 0
        budget_summaries = int(budget * 0.20)
        budget_overview = int(budget * 0.10)
        if not include_imports:
            budget_code += int(budget * 0.30)

        warnings: List[str] = []
        all_chunks = self._collect_all_chunks()
        if not all_chunks:
            warnings.append("No code chunks found. Run 'know init' first.")
            return self._empty_result(query, budget, warnings)

        scored = self._score_chunks(query, all_chunks)
        self._apply_recency_boost(scored)
        scored.sort(key=lambda c: c.score + c.recency_boost * 0.15, reverse=True)

        code_chunks: List[CodeChunk] = []
        code_used = 0
        seen_files: set = set()
        for chunk in scored:
            if code_used + chunk.tokens > budget_code:
                continue
            code_chunks.append(chunk)
            code_used += chunk.tokens
            seen_files.add(chunk.file_path)

        dep_chunks, dep_used = ([], 0)
        if include_imports and code_chunks:
            dep_chunks, dep_used = self._expand_imports(seen_files, budget_imports, scored)

        test_chunks, test_used = ([], 0)
        if include_tests:
            leftover = max(0, budget_imports - dep_used) if include_imports else 0
            test_budget = leftover + max(0, budget_summaries // 4)
            test_chunks, test_used = self._find_tests(seen_files, test_budget)

        summary_budget = budget_summaries - test_used
        summary_chunks, summary_used = self._build_summaries(seen_files, max(0, summary_budget))
        overview = self._project_overview(budget_overview)
        overview_tokens = count_tokens(overview)
        total_used = code_used + dep_used + test_used + summary_used + overview_tokens

        if total_used < budget * 0.2:
            warnings.append(
                f"Only {total_used} tokens of context found for a {budget} budget. "
                "The query may be too narrow or the project too small."
            )

        return {
            "query": query, "budget": budget,
            "used_tokens": total_used,
            "budget_display": format_budget(total_used, budget),
            "code_chunks": code_chunks, "dependency_chunks": dep_chunks,
            "test_chunks": test_chunks, "summary_chunks": summary_chunks,
            "overview": overview, "warnings": warnings,
            "indexing_status": "legacy",
            "confidence": round(total_used / budget, 2) if budget > 0 else 0,
            "index_stats": {},
        }

    # ------------------------------------------------------------------
    # Internal: chunk collection (legacy)
    # ------------------------------------------------------------------
    def _collect_all_chunks(self) -> List[CodeChunk]:
        """Collect all code chunks from the project (legacy filesystem scan)."""
        chunks: List[CodeChunk] = []
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(self.config)
        files = list(scanner._discover_files())

        for file_path, lang in files:
            if lang == "python":
                file_chunks = extract_chunks_from_file(file_path, self.root)
                chunks.extend(file_chunks)
            else:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel = str(file_path.relative_to(self.root))
                    tokens = count_tokens(content)
                    chunks.append(CodeChunk(
                        file_path=rel, name=file_path.stem,
                        chunk_type="module", line_start=1,
                        line_end=content.count("\n") + 1,
                        body=content, tokens=tokens,
                    ))
                except Exception:
                    pass
        return chunks

    # ------------------------------------------------------------------
    # Internal: scoring (legacy)
    # ------------------------------------------------------------------
    def _score_chunks(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Score chunks against the query (legacy)."""
        try:
            return self._score_semantic(query, chunks)
        except Exception as e:
            logger.debug(f"Semantic scoring unavailable ({e}), falling back to text match")
            return self._score_text(query, chunks)

    def _score_semantic(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Score via fastembed embeddings (legacy)."""
        import numpy as np
        model = _get_cached_embedding_model()
        if model is None:
            return self._score_text(query, chunks)

        query_emb = np.array(list(model.embed([query]))[0], dtype=np.float32)
        texts = [f"{c.name} {c.signature} {c.docstring} {c.body[:300]}" for c in chunks]
        embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        e_norm = embeddings / norms
        similarities = e_norm @ q_norm

        for i, chunk in enumerate(chunks):
            chunk.score = float(similarities[i])
        return chunks

    def _score_text(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Fallback: score using simple text matching (legacy)."""
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return chunks
        for chunk in chunks:
            chunk_text = f"{chunk.name} {chunk.signature} {chunk.docstring} {chunk.body[:300]}".lower()
            chunk_words = set(re.findall(r'\w+', chunk_text))
            if not chunk_words:
                chunk.score = 0.0
                continue
            overlap = len(query_words & chunk_words)
            chunk.score = overlap / len(query_words)
        return chunks

    # ------------------------------------------------------------------
    # Internal: recency (legacy)
    # ------------------------------------------------------------------
    def _apply_recency_boost(self, chunks: List[CodeChunk]):
        """Apply git recency boost to chunks using batch operations."""
        unique_files = {c.file_path for c in chunks}
        recency_scores = _get_batch_file_recency(self.root, list(unique_files))
        for chunk in chunks:
            chunk.recency_boost = recency_scores.get(chunk.file_path, 0.0)

    # ------------------------------------------------------------------
    # Internal: import expansion (legacy)
    # ------------------------------------------------------------------
    def _expand_imports(
        self, seen_files: set, budget: int, all_chunks: List[CodeChunk],
    ) -> Tuple[List[CodeChunk], int]:
        """For each relevant file, include signatures of its imports."""
        try:
            from know.import_graph import ImportGraph
            graph = ImportGraph(self.config)
        except Exception:
            return [], 0

        dep_modules: set = set()
        for fp in seen_files:
            mod_name = Path(fp).stem
            for imp in graph.imports_of(mod_name):
                dep_modules.add(imp)

        dep_chunks: List[CodeChunk] = []
        used = 0
        for mod_name in dep_modules:
            fpath = graph.file_for_module(mod_name)
            if fpath is None or not fpath.exists():
                continue
            rel = str(fpath.relative_to(self.root))
            if rel in seen_files:
                continue
            sigs = _extract_signatures(fpath)
            if not sigs:
                continue
            tokens = count_tokens(sigs)
            if used + tokens > budget:
                sigs = truncate_to_budget(sigs, budget - used)
                tokens = count_tokens(sigs)
                if used + tokens > budget:
                    continue
            dep_chunks.append(CodeChunk(
                file_path=rel, name=f"{fpath.stem} (signatures)",
                chunk_type="module", line_start=1,
                line_end=sigs.count("\n") + 1, body=sigs,
                signature="# signatures only", tokens=tokens,
            ))
            used += tokens
        return dep_chunks, used

    # ------------------------------------------------------------------
    # Internal: test discovery (legacy)
    # ------------------------------------------------------------------
    def _find_tests(
        self, seen_files: set, budget: int,
    ) -> Tuple[List[CodeChunk], int]:
        """Find and extract test functions for the relevant source files."""
        test_chunks: List[CodeChunk] = []
        used = 0
        for fp in seen_files:
            test_files = _find_test_files(self.root, fp)
            for tf in test_files:
                chunks = extract_chunks_from_file(tf, self.root)
                for chunk in chunks:
                    if chunk.chunk_type == "module":
                        continue
                    if used + chunk.tokens > budget:
                        break
                    test_chunks.append(chunk)
                    used += chunk.tokens
        return test_chunks, used

    # ------------------------------------------------------------------
    # Internal: summaries (legacy)
    # ------------------------------------------------------------------
    def _build_summaries(
        self, seen_files: set, budget: int,
    ) -> Tuple[List[CodeChunk], int]:
        """Build file-level summaries for context."""
        summaries: List[CodeChunk] = []
        used = 0
        for fp in seen_files:
            abs_path = self.root / fp
            if not abs_path.exists():
                continue
            try:
                content = abs_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content)
                doc = ast.get_docstring(tree) or ""
            except Exception:
                continue

            names = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(f"def {node.name}()")
                elif isinstance(node, ast.ClassDef):
                    names.append(f"class {node.name}")

            summary = f"# {fp}\n"
            if doc:
                summary += f"{doc}\n\n"
            if names:
                summary += "Contains: " + ", ".join(names) + "\n"

            tokens = count_tokens(summary)
            if used + tokens > budget:
                continue
            summaries.append(CodeChunk(
                file_path=fp, name=f"{Path(fp).stem} (summary)",
                chunk_type="module", line_start=1, line_end=1,
                body=summary, tokens=tokens,
            ))
            used += tokens
        return summaries, used

    # ------------------------------------------------------------------
    # Internal: overview
    # ------------------------------------------------------------------
    def _project_overview(self, budget: int) -> str:
        """Build a brief project overview."""
        overview_parts = []
        overview_parts.append(f"Project: {self.config.project.name or self.root.name}")
        if self.config.project.description:
            overview_parts.append(self.config.project.description)

        for readme_name in ["README.md", "readme.md", "README.rst"]:
            readme = self.root / readme_name
            if readme.exists():
                try:
                    content = readme.read_text(encoding="utf-8", errors="ignore")
                    paragraphs = content.split("\n\n")
                    for p in paragraphs[:3]:
                        if p.strip() and not p.strip().startswith("!["):
                            overview_parts.append(p.strip())
                            break
                except Exception:
                    pass
                break

        overview = "\n\n".join(overview_parts)
        return truncate_to_budget(overview, budget)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def format_markdown(self, result: Dict[str, Any]) -> str:
        """Format context result as Markdown."""
        lines = [
            f'# Context for: "{result["query"]}"',
            f'## Token Budget: {result["budget_display"]}',
            "",
        ]

        if result.get("warnings"):
            for w in result["warnings"]:
                lines.append(f"> {w}")
            lines.append("")

        if result["code_chunks"]:
            lines.append("### Relevant Code")
            lines.append("")
            for chunk in result["code_chunks"]:
                lines.append(f"#### {chunk.header()}")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        if result["dependency_chunks"]:
            lines.append("### Dependencies")
            lines.append("")
            for chunk in result["dependency_chunks"]:
                lines.append(f"#### {chunk.file_path} (signatures only)")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        if result["test_chunks"]:
            lines.append("### Related Tests")
            lines.append("")
            for chunk in result["test_chunks"]:
                lines.append(f"#### {chunk.header()}")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        if result["summary_chunks"]:
            lines.append("### File Summaries")
            lines.append("")
            for chunk in result["summary_chunks"]:
                lines.append(chunk.body)
                lines.append("")

        if result.get("memories_context"):
            lines.append("### Memories (Cross-Session Knowledge)")
            lines.append("")
            lines.append(result["memories_context"])
            lines.append("")

        if result["overview"]:
            lines.append("### Project Context")
            lines.append("")
            lines.append(result["overview"])
            lines.append("")

        return "\n".join(lines)

    def format_agent_json(self, result: Dict[str, Any]) -> str:
        """Format context result as JSON optimized for AI agent consumption."""
        def _chunk_to_dict(c: CodeChunk) -> dict:
            d = {
                "file": c.file_path,
                "name": c.name,
                "type": c.chunk_type,
                "signature": c.signature,
                "lines": [c.line_start, c.line_end],
                "score": round(c.score, 3),
                "tokens": c.tokens,
            }
            # Add structural metadata if available (from v3 pipeline)
            if hasattr(c, '_metadata'):
                d.update(c._metadata)
            # body LAST for truncation safety
            d["body"] = c.body
            return d

        payload = {
            "query": result["query"],
            "budget": result["budget"],
            "used_tokens": result["used_tokens"],
            "budget_utilization": result["budget_display"],
            "indexing_status": result.get("indexing_status", "unknown"),
            "confidence": result.get("confidence", 0),
            "warnings": result.get("warnings", []),
            "code": [_chunk_to_dict(c) for c in result["code_chunks"]],
            "dependencies": [_chunk_to_dict(c) for c in result["dependency_chunks"]],
            "tests": [_chunk_to_dict(c) for c in result["test_chunks"]],
            "summaries": [_chunk_to_dict(c) for c in result["summary_chunks"]],
            "overview": result["overview"],
            "memories": result.get("memories_context", ""),
            "source_files": list({c.file_path for c in result["code_chunks"]}),
        }

        # Zero-result intelligence
        if result.get("nearest_terms"):
            payload["nearest_terms"] = result["nearest_terms"]
        if result.get("file_names_matching"):
            payload["file_names_matching"] = result["file_names_matching"]
        if result.get("index_stats"):
            payload["index_stats"] = result["index_stats"]

        return json.dumps(payload, indent=2)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _empty_result(self, query: str, budget: int, warnings: List[str]) -> Dict[str, Any]:
        return {
            "query": query,
            "budget": budget,
            "used_tokens": 0,
            "budget_display": format_budget(0, budget),
            "code_chunks": [],
            "dependency_chunks": [],
            "test_chunks": [],
            "summary_chunks": [],
            "overview": "",
            "warnings": warnings,
            "indexing_status": "unknown",
            "confidence": 0,
            "index_stats": {},
        }
