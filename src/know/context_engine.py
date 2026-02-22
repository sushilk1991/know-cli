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
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

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
        session_id: Optional[str] = None,
        db=None,
    ) -> Dict[str, Any]:
        """Build context bundle for *query* within *budget* tokens.

        Uses DaemonDB FTS5 search with BM25F field weighting.
        Falls back to direct DaemonDB if daemon socket unavailable.
        Set legacy=True to force old filesystem scan (debugging only).

        Pass session_id to enable cross-query dedup: chunks returned in
        previous calls with the same session are skipped and budget is
        re-filled with new results.

        Returns a dict with keys:
            query, budget, used_tokens,
            code_chunks, dependency_chunks, test_chunks,
            summary_chunks, overview, warnings, indexing_status,
            index_stats, confidence, session_id (if provided)

        Args:
            db: Optional pre-existing DaemonDB instance (avoids creating a new one).
        """
        if legacy:
            return self._build_context_legacy(query, budget, include_tests, include_imports)

        if db is not None:
            return self._build_context_v3_inner(
                db, query, budget, include_tests, include_imports,
                include_patterns, exclude_patterns, chunk_types,
                session_id,
            )

        return self._build_context_v3(
            query, budget, include_tests, include_imports,
            include_patterns, exclude_patterns, chunk_types,
            session_id,
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
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """v3 context building via DaemonDB."""
        # Get DaemonDB (direct connection, no daemon socket needed)
        db = get_direct_db(self.config)
        try:
            return self._build_context_v3_inner(
                db, query, budget, include_tests, include_imports,
                include_patterns, exclude_patterns, chunk_types,
                session_id,
            )
        finally:
            db.close()

    def _build_context_v3_inner(
        self, db, query: str, budget: int,
        include_tests: bool, include_imports: bool,
        include_patterns, exclude_patterns, chunk_types,
        session_id: Optional[str] = None,
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
                indexed, modules = populate_index(self.config.root, self.config, db)
                logger.debug(f"Inline indexing complete: {indexed} files")

                # Build import graph so dependency budget gets filled
                try:
                    from know.import_graph import ImportGraph
                    ig = ImportGraph(self.config)
                    ig.build(modules)
                except Exception as e:
                    logger.debug(f"Inline import graph build failed: {e}")

                # Compute importance scores (in-degree from import graph)
                try:
                    db.compute_importance()
                except Exception as e:
                    logger.debug(f"Inline importance computation failed: {e}")

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

        # Step 1: Hybrid retrieval (lexical + graph + semantic lanes, RRF fused)
        raw_results = self._retrieve_hybrid_candidates(db, query, limit=120)

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

        # Step 2.5: Importance boost — boost high-in-degree modules
        try:
            file_paths = list({r["file_path"] for r in raw_results})
            modules = [
                fp.replace("/", ".").replace("\\", ".").removesuffix(".py")
                for fp in file_paths
            ]
            importance_scores = db.get_importance_batch(modules)
            fp_to_importance = {}
            for fp, mod in zip(file_paths, modules):
                fp_to_importance[fp] = importance_scores.get(mod, 0.0)
            for r in raw_results:
                imp = fp_to_importance.get(r["file_path"], 0.0)
                # Boost by up to 50% based on module importance
                r["score"] = r.get("score", 0) * (1.0 + 0.5 * imp)
        except Exception:
            pass

        # Step 2.6: Git recency boost — boost recently changed files
        try:
            file_paths = list({r["file_path"] for r in raw_results})
            recency_scores = _get_batch_file_recency(self.root, file_paths)
            for r in raw_results:
                recency = recency_scores.get(r["file_path"], 0.0)
                # Boost by up to 20% based on recency
                r["score"] = r.get("score", 0) * (1.0 + 0.2 * recency)
        except Exception:
            pass

        # Step 2.7: File-path exact match boost
        try:
            from know.query import analyze_query
            plan = analyze_query(query)
            path_terms = set(t.lower() for t in plan.identifiers + plan.terms)
            for r in raw_results:
                fp_lower = r["file_path"].lower()
                # Check if any search term appears in the file path
                for term in path_terms:
                    if term in fp_lower:
                        r["score"] = r.get("score", 0) * 2.0
                        break
        except Exception:
            pass

        # Step 2.8: Domain intent + generic UI noise demotion
        try:
            ql = query.lower()
            frontend_terms = {
                "react", "tsx", "jsx", "sidebar", "component", "page", "route",
                "frontend", "client", "ui", "css", "tailwind", "next", "redirect",
            }
            backend_terms = {
                "api", "backend", "server", "database", "db", "sql", "endpoint",
                "middleware", "auth", "worker", "queue", "python",
            }
            fscore = sum(1 for t in frontend_terms if t in ql)
            bscore = sum(1 for t in backend_terms if t in ql)
            intent = "frontend" if fscore > bscore else "backend" if bscore > fscore else "mixed"

            noisy_tokens = {"navigationmenu", "navmenu", "menuitem", "dropdownmenu"}
            for r in raw_results:
                fp = r.get("file_path", "").lower()
                ext = Path(fp).suffix
                score = float(r.get("score", 0.0))

                if intent == "frontend":
                    if ext in {".ts", ".tsx", ".js", ".jsx"}:
                        score *= 1.35
                    if any(p in fp for p in ("frontend/", "web/", "client/", "components/", "app/")):
                        score *= 1.20
                    if ext == ".py":
                        score *= 0.90
                elif intent == "backend":
                    if ext == ".py":
                        score *= 1.35
                    if any(p in fp for p in ("backend/", "server/", "api/", "services/")):
                        score *= 1.20
                    if ext in {".ts", ".tsx", ".js", ".jsx"}:
                        score *= 0.90

                body = (r.get("body", "") or "")[:240].lower()
                if any(tok in fp or tok in body for tok in noisy_tokens):
                    score *= 0.75

                r["score"] = score
        except Exception:
            pass

        # Step 3: Apply file path filtering
        if include_patterns or exclude_patterns or chunk_types:
            raw_results = self._apply_filters(
                raw_results, include_patterns, exclude_patterns, chunk_types,
            )

        # Step 4: Re-sort by boosted scores
        raw_results.sort(key=lambda c: c.get("score", 0), reverse=True)

        # Step 5: Apply relevance floor (return under budget, not noise)
        raw_results = apply_relevance_floor(raw_results)

        # Step 5.5: Session dedup — filter out already-seen chunks
        session_seen_keys: set = set()
        if session_id:
            try:
                db.create_session(session_id)
                session_seen_keys = db.get_session_seen(session_id)
                if session_seen_keys:
                    raw_results = [
                        r for r in raw_results
                        if f"{r['file_path']}:{r.get('chunk_name', '')}:{r.get('start_line', 0)}"
                        not in session_seen_keys
                    ]
            except Exception as e:
                logger.debug(f"Session dedup failed: {e}")

        # Step 6: compute sub-budgets (adaptive by query type)
        try:
            from know.query import analyze_query
            plan = analyze_query(query)
            qtype = plan.query_type
        except Exception:
            qtype = "concept"

        if qtype == "identifier":
            # Identifier queries: maximize code, minimize fluff
            alloc_code, alloc_imports, alloc_summaries, alloc_overview = 0.80, 0.15, 0.05, 0.00
        elif qtype == "error":
            # Error queries: lots of code, skip overview
            alloc_code, alloc_imports, alloc_summaries, alloc_overview = 0.70, 0.15, 0.15, 0.00
        else:
            # Concept queries: balanced
            alloc_code, alloc_imports, alloc_summaries, alloc_overview = 0.60, 0.15, 0.15, 0.10

        budget_code = int(budget * alloc_code)
        budget_imports = int(budget * alloc_imports) if include_imports else 0
        budget_summaries = int(budget * alloc_summaries)
        budget_overview = int(budget * alloc_overview)
        if not include_imports:
            budget_code += int(budget * alloc_imports)

        # Step 7: greedily fill code budget using pre-computed token_count
        code_chunks: List[Dict] = []
        code_used = 0
        seen_files: set = set()
        seen_chunk_keys: set = set()
        for chunk in raw_results:
            tokens = chunk.get("token_count", 0)
            if tokens == 0:
                tokens = count_tokens(chunk.get("body", ""))
            if code_used + tokens > budget_code:
                continue
            code_chunks.append(chunk)
            code_used += tokens
            seen_files.add(chunk["file_path"])
            seen_chunk_keys.add(
                f"{chunk['file_path']}:{chunk.get('chunk_name', '')}:{chunk.get('start_line', 0)}"
            )

        # Step 7.5: Context expansion — add neighborhoods
        code_chunks, code_used = self._expand_context(
            db, code_chunks, code_used, budget_code, seen_chunk_keys,
        )

        # Step 7.6: Chunk deduplication — remove overlapping class/method chunks
        code_chunks, code_used = self._deduplicate_chunks(code_chunks)

        # Step 8: Bundle structural metadata (batch import lookups)
        seen_files = {c["file_path"] for c in code_chunks}
        if seen_files:
            self._bundle_metadata(db, code_chunks, seen_files)

        # Step 9: Group chunks by file
        code_chunks = self._group_by_file(code_chunks)

        # Step 9.5: Prompt packing to reduce "lost in the middle"
        code_chunks = self._pack_chunks_for_prompt(code_chunks)

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

        # Step 13: Mark selected chunks as seen in session
        if session_id and code_chunks:
            try:
                new_keys = []
                new_tokens = []
                for c in code_chunks:
                    key = f"{c['file_path']}:{c.get('chunk_name', '')}:{c.get('start_line', 0)}"
                    new_keys.append(key)
                    new_tokens.append(c.get("token_count", 0))
                db.mark_session_seen(session_id, new_keys, new_tokens)
            except Exception as e:
                logger.debug(f"Session mark_seen failed: {e}")

        result = {
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

        if session_id:
            result["session_id"] = session_id

        return result

    @staticmethod
    def _chunk_key(chunk: Dict[str, Any]) -> str:
        return f"{chunk.get('file_path', '')}:{chunk.get('chunk_name', '')}:{chunk.get('start_line', 0)}"

    def _retrieve_hybrid_candidates(
        self, db, query: str, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates via lexical + graph + semantic lanes, fused with RRF."""
        lexical = db.search_chunks(query, limit=limit)
        if not lexical:
            return []

        graph = self._graph_expand_lane(db, lexical, query, limit=max(20, limit // 2))
        semantic = self._semantic_rerank_lane(query, lexical + graph, limit=max(20, limit // 2))

        fused = self._fuse_hybrid_lanes(lexical, graph, semantic, limit=limit)
        return fused or lexical[:limit]

    def _graph_expand_lane(
        self,
        db,
        seeds: List[Dict[str, Any]],
        query: str,
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Build graph-neighborhood retrieval lane from call/import neighborhoods."""
        if not seeds:
            return []

        neighbors: Dict[str, Dict[str, Any]] = {}
        top_seeds = sorted(seeds, key=lambda c: c.get("score", 0), reverse=True)[:12]

        def _add_chunk(chunk: Dict[str, Any], base_score: float) -> None:
            if not chunk:
                return
            key = self._chunk_key(chunk)
            score = float(base_score)
            current = neighbors.get(key)
            if current is None or score > float(current.get("score", 0)):
                item = dict(chunk)
                item["score"] = score
                neighbors[key] = item

        # Call-neighborhood expansion.
        for seed in top_seeds:
            seed_name = seed.get("chunk_name", "")
            if not seed_name:
                continue

            try:
                callees = db.get_callees(seed_name, limit=25)
            except Exception:
                callees = []
            for ref in callees:
                ref_name = ref.get("ref_name", "")
                if not ref_name:
                    continue
                for match in db.get_chunks_by_name(ref_name, limit=4):
                    _add_chunk(match, 1.0)
                if "." in ref_name and hasattr(db, "get_method_chunks_by_suffix"):
                    leaf = ref_name.rsplit(".", 1)[-1]
                    for match in db.get_method_chunks_by_suffix(leaf, limit=4):
                        _add_chunk(match, 0.95)

            try:
                callers = db.get_callers(seed_name, limit=25)
            except Exception:
                callers = []
            for ref in callers:
                caller_name = ref.get("containing_chunk", "")
                if not caller_name:
                    continue
                for match in db.get_chunks_by_name(caller_name, limit=4):
                    _add_chunk(match, 0.9)

        # Import-neighborhood expansion (module-level graph).
        modules = []
        for seed in top_seeds:
            fp = seed.get("file_path", "")
            if not fp:
                continue
            p = Path(fp)
            module = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
            modules.append(module)

        module_neighbors: Set[str] = set()
        if modules:
            try:
                imports_map = db.get_imports_batch(modules)
                imported_by_map = db.get_imported_by_batch(modules)
                for v in imports_map.values():
                    module_neighbors.update(v)
                for v in imported_by_map.values():
                    module_neighbors.update(v)
            except Exception:
                pass

        for mod in list(module_neighbors)[:120]:
            file_path = self._resolve_module_to_file(db, mod)
            if not file_path:
                continue
            chunks = db.get_chunks_for_file(file_path)
            if not chunks:
                continue
            rank_priority = {"function": 0, "method": 1, "class": 2, "module": 3}
            chunks = sorted(
                chunks,
                key=lambda c: (rank_priority.get(c.get("chunk_type", "module"), 9), c.get("start_line", 0)),
            )
            for chunk in chunks[:2]:
                _add_chunk(chunk, 0.7)

        # Keep graph lane deterministic and bounded.
        results = sorted(neighbors.values(), key=lambda c: c.get("score", 0), reverse=True)
        return results[:limit]

    def _resolve_module_to_file(self, db, module_name: str) -> Optional[str]:
        """Resolve a module name to an indexed file path across supported extensions."""
        module = (module_name or "").strip()
        if not module:
            return None

        stem = module.replace(".", "/")
        ext_candidates = [
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".swift",
            ".java", ".rb", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
        ]
        file_candidates = [f"{stem}{ext}" for ext in ext_candidates]
        file_candidates.append(f"{stem}/__init__.py")

        for fp in file_candidates:
            try:
                if db.get_chunks_for_file(fp):
                    return fp
            except Exception:
                continue
        return None

    def _semantic_rerank_lane(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        limit: int = 80,
    ) -> List[Dict[str, Any]]:
        """Embedding-based rerank lane over lexical+graph candidates."""
        if not candidates:
            return []

        model = _get_cached_embedding_model()
        if model is None:
            return []

        # Deduplicate before embedding.
        deduped: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            key = self._chunk_key(c)
            existing = deduped.get(key)
            if existing is None or float(c.get("score", 0)) > float(existing.get("score", 0)):
                deduped[key] = c
        unique = list(deduped.values())[:200]
        if not unique:
            return []

        texts = [
            f"{c.get('chunk_name', '')} {c.get('signature', '')}\n{(c.get('body', '') or '')[:500]}"
            for c in unique
        ]

        try:
            import numpy as np
            query_emb = np.array(list(model.embed([query]))[0], dtype=np.float32)
            chunk_emb = np.array(list(model.embed(texts)), dtype=np.float32)
            q_norm = np.linalg.norm(query_emb)
            if q_norm == 0 or chunk_emb.size == 0:
                return []
            query_emb = query_emb / q_norm
            norms = np.linalg.norm(chunk_emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            chunk_emb = chunk_emb / norms
            sims = chunk_emb @ query_emb
        except Exception as e:
            logger.debug(f"Semantic lane skipped: {e}")
            return []

        scored = []
        for i, chunk in enumerate(unique):
            item = dict(chunk)
            item["score"] = float(sims[i])
            scored.append(item)
        scored.sort(key=lambda c: c.get("score", 0), reverse=True)
        return scored[:limit]

    def _fuse_hybrid_lanes(
        self,
        lexical: List[Dict[str, Any]],
        graph: List[Dict[str, Any]],
        semantic: List[Dict[str, Any]],
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fuse ranked lanes with weighted RRF."""
        from know.ranking import fuse_rankings

        lane_defs: List[Tuple[List[Dict[str, Any]], int]] = []
        if lexical:
            lane_defs.append((lexical, 3))
        if graph:
            lane_defs.append((graph, 2))
        if semantic:
            lane_defs.append((semantic, 2))
        if not lane_defs:
            return []
        if len(lane_defs) == 1:
            return lane_defs[0][0][:limit]

        ranked_lists: List[List[Tuple[str, float]]] = []
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for lane, weight in lane_defs:
            seen = set()
            keyed: List[Tuple[str, float]] = []
            lane_sorted = sorted(lane, key=lambda c: c.get("score", 0), reverse=True)
            for chunk in lane_sorted:
                key = self._chunk_key(chunk)
                if key in seen:
                    continue
                seen.add(key)
                keyed.append((key, float(chunk.get("score", 0))))
                prev = chunk_map.get(key)
                if prev is None or float(chunk.get("score", 0)) > float(prev.get("score", 0)):
                    chunk_map[key] = dict(chunk)
            for _ in range(weight):
                ranked_lists.append(keyed)

        fused = fuse_rankings(ranked_lists)
        out: List[Dict[str, Any]] = []
        for key, fused_score in fused[:limit]:
            base = chunk_map.get(key)
            if not base:
                continue
            item = dict(base)
            item["score"] = float(fused_score)
            out.append(item)
        return out

    def _pack_chunks_for_prompt(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Place highest-utility chunks at prompt edges to reduce lost-in-middle."""
        if len(chunks) <= 2:
            return chunks

        # Keep one entry per chunk key.
        deduped: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            key = self._chunk_key(chunk)
            if key not in deduped:
                deduped[key] = chunk
        unique = list(deduped.values())

        def utility(c: Dict[str, Any]) -> float:
            score = float(c.get("score", 0.0))
            imported_by = c.get("imported_by", []) or []
            centrality = min(len(imported_by), 10) / 10.0
            token_count = max(1, int(c.get("token_count", 0) or 0))
            compactness = 1.0 / (1.0 + (token_count / 300.0))
            return score + 0.2 * centrality + 0.1 * compactness

        ranked = sorted(unique, key=utility, reverse=True)

        left: List[Dict[str, Any]] = []
        right: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(ranked):
            if idx % 2 == 0:
                left.append(chunk)
            else:
                right.append(chunk)

        return left + list(reversed(right))

    def _expand_context(
        self,
        db,
        code_chunks: List[Dict],
        code_used: int,
        budget_code: int,
        seen_chunk_keys: set,
    ) -> Tuple[List[Dict], int]:
        """Expand selected chunks with neighborhoods for better context.

        For each selected chunk:
        - Include module-level chunk (imports + docstring) from same file
        - If method, include parent class signature
        - Include adjacent chunks (within 10 lines) if budget allows
        """
        if not code_chunks:
            return code_chunks, code_used

        # Collect all file paths that need expansion
        files_to_expand = {c["file_path"] for c in code_chunks}

        # Pre-fetch all chunks for these files (batch)
        file_chunks_map: Dict[str, List[Dict]] = {}
        for fp in files_to_expand:
            file_chunks_map[fp] = db.get_chunks_for_file(fp)

        extra_chunks: List[Dict] = []

        for chunk in list(code_chunks):
            fp = chunk["file_path"]
            all_file_chunks = file_chunks_map.get(fp, [])
            chunk_start = chunk.get("start_line", 0)
            chunk_end = chunk.get("end_line", 0)
            chunk_type = chunk.get("chunk_type", "")

            # 1. Include module-level chunk (imports + docstring) from same file
            for fc in all_file_chunks:
                if fc.get("chunk_type") == "module":
                    key = f"{fc['file_path']}:{fc.get('chunk_name', '')}:{fc.get('start_line', 0)}"
                    if key not in seen_chunk_keys:
                        tokens = fc.get("token_count", 0)
                        if code_used + tokens <= budget_code:
                            extra_chunks.append(fc)
                            code_used += tokens
                            seen_chunk_keys.add(key)
                    break  # Only one module chunk per file

            # 2. If method, include parent class signature
            if chunk_type == "method":
                for fc in all_file_chunks:
                    if fc.get("chunk_type") == "class":
                        fc_start = fc.get("start_line", 0)
                        fc_end = fc.get("end_line", 0)
                        # Class encloses this method
                        if fc_start <= chunk_start and fc_end >= chunk_end:
                            key = f"{fc['file_path']}:{fc.get('chunk_name', '')}:{fc.get('start_line', 0)}"
                            if key not in seen_chunk_keys:
                                # Add just class signature (first few lines)
                                sig_body = fc.get("signature", fc.get("chunk_name", ""))
                                doc = ""
                                body = fc.get("body", "")
                                # Extract first ~5 lines as class header
                                body_lines = body.split("\n")[:5]
                                sig_text = "\n".join(body_lines)
                                tokens = count_tokens(sig_text)
                                if code_used + tokens <= budget_code:
                                    sig_chunk = dict(fc)
                                    sig_chunk["body"] = sig_text
                                    sig_chunk["token_count"] = tokens
                                    sig_chunk["chunk_name"] = f"{fc.get('chunk_name', '')} (class header)"
                                    extra_chunks.append(sig_chunk)
                                    code_used += tokens
                                    seen_chunk_keys.add(key)
                            break

            # 3. Include adjacent chunks (within 10 lines) if budget allows
            for fc in all_file_chunks:
                fc_start = fc.get("start_line", 0)
                fc_end = fc.get("end_line", 0)
                key = f"{fc['file_path']}:{fc.get('chunk_name', '')}:{fc.get('start_line', 0)}"
                if key in seen_chunk_keys:
                    continue
                # Check adjacency: within 10 lines of the selected chunk
                if (abs(fc_start - chunk_end) <= 10 or abs(chunk_start - fc_end) <= 10):
                    tokens = fc.get("token_count", 0)
                    if code_used + tokens <= budget_code:
                        extra_chunks.append(fc)
                        code_used += tokens
                        seen_chunk_keys.add(key)

        code_chunks.extend(extra_chunks)
        return code_chunks, code_used

    def _deduplicate_chunks(
        self, code_chunks: List[Dict],
    ) -> Tuple[List[Dict], int]:
        """Remove overlapping class/method chunks.

        If a class chunk AND its method chunks are both selected:
        - If 3+ methods selected from same class → drop the full class body
        - Otherwise keep the class chunk and drop methods already inside it
        """
        if not code_chunks:
            return code_chunks, 0

        # Group by file
        by_file: Dict[str, List[Dict]] = {}
        for c in code_chunks:
            by_file.setdefault(c["file_path"], []).append(c)

        keep: List[Dict] = []
        total_tokens = 0

        for fp, chunks in by_file.items():
            classes = [c for c in chunks if c.get("chunk_type") == "class"]
            methods = [c for c in chunks if c.get("chunk_type") == "method"]
            others = [c for c in chunks if c.get("chunk_type") not in ("class", "method")]

            # Always keep non-class/method chunks
            for c in others:
                keep.append(c)
                total_tokens += c.get("token_count", 0)

            for cls in classes:
                cls_start = cls.get("start_line", 0)
                cls_end = cls.get("end_line", 0)

                # Find methods that are inside this class
                enclosed = [
                    m for m in methods
                    if m.get("start_line", 0) >= cls_start
                    and m.get("end_line", 0) <= cls_end
                ]

                if len(enclosed) >= 3:
                    # Many methods selected → skip the full class body, keep methods
                    for m in enclosed:
                        keep.append(m)
                        total_tokens += m.get("token_count", 0)
                        methods = [x for x in methods if x is not m]
                else:
                    # Few methods → keep class body, skip enclosed methods
                    keep.append(cls)
                    total_tokens += cls.get("token_count", 0)
                    for m in enclosed:
                        methods = [x for x in methods if x is not m]

            # Keep methods not enclosed in any class
            for m in methods:
                keep.append(m)
                total_tokens += m.get("token_count", 0)

        return keep, total_tokens

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
    # Deep context (know deep)
    # ------------------------------------------------------------------

    def build_deep_context(
        self,
        name: str,
        budget: int = 3000,
        include_tests: bool = False,
        session_id: Optional[str] = None,
        db=None,
    ) -> Dict[str, Any]:
        """Build deep context for a function: body + callers + callees.

        Returns a dict with target, callees, callers, overflow_signatures,
        budget_used, budget, and optional session_id.

        Args:
            db: Optional pre-existing DaemonDB instance (avoids creating a new one).
        """
        needs_close = False
        if db is None:
            db = get_direct_db(self.config)
            needs_close = True

        try:
            return self._build_deep_context_inner(
                db, name, budget, include_tests, session_id,
            )
        finally:
            if needs_close:
                db.close()

    def _build_deep_context_inner(
        self,
        db,
        name: str,
        budget: int,
        include_tests: bool,
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Inner deep context pipeline."""
        from know.file_categories import categorize_file
        from know.daemon import refresh_files_if_stale

        # Step 0: Opportunistic stale-file refresh for the queried symbol.
        # This keeps deep context accurate without requiring full reindex.
        refresh_candidates = self._collect_deep_refresh_candidates(db, name)
        if refresh_candidates:
            try:
                refresh_files_if_stale(
                    self.config.root, self.config, db, refresh_candidates,
                )
            except Exception as e:
                logger.debug(f"Deep stale-file refresh failed: {e}")

        # Step 1: Resolve function name to chunk(s)
        candidates = self._resolve_function(db, name, include_tests)

        if not candidates:
            # Try fuzzy: search for chunks containing the name
            nearest = []
            try:
                search_results = db.search_chunks(name, limit=5)
                nearest = [
                    r.get("chunk_name", "") for r in search_results
                    if r.get("chunk_name")
                ]
            except Exception:
                pass
            return {"error": "not_found", "nearest": nearest}

        if len(candidates) > 1:
            return {
                "error": "ambiguous",
                "candidates": [
                    {
                        "file_path": c["file_path"],
                        "chunk_name": c["chunk_name"],
                        "chunk_type": c.get("chunk_type", "function"),
                        "start_line": c.get("start_line", 0),
                    }
                    for c in candidates
                ],
            }

        target_chunk = candidates[0]
        target_name = target_chunk["chunk_name"]
        target_body = target_chunk.get("body", "")
        target_tokens = target_chunk.get("token_count", 0) or count_tokens(target_body)

        # Cap target at 50% of budget
        max_target = int(budget * 0.50)
        if target_tokens > max_target:
            target_body = truncate_to_budget(target_body, max_target)
            target_tokens = count_tokens(target_body)

        remaining = budget - target_tokens

        # Step 2: Get callees (what the function calls)
        callees_budget = int(remaining * 0.50)
        raw_callees = db.get_callees(target_name, limit=30)

        # Deduplicate by location, not just symbol name.
        callees_data = []
        seen_callee_keys = set()
        for ref in raw_callees:
            ref_name = ref.get("ref_name", "")
            key = (
                ref_name,
                ref.get("file_path", ""),
                ref.get("line_number", 0),
            )
            if key in seen_callee_keys:
                continue
            seen_callee_keys.add(key)
            callees_data.append(ref)

        # Step 3: Get callers (what calls the function)
        callers_budget = remaining - callees_budget
        raw_callers = db.get_callers(target_name, limit=30)
        if "." in target_name:
            leaf_name = target_name.rsplit(".", 1)[-1]
            if leaf_name and leaf_name != target_name:
                raw_callers.extend(db.get_callers(leaf_name, limit=30))

        callers_data = []
        seen_caller_keys = set()
        for ref in raw_callers:
            caller_name = ref.get("containing_chunk", "")
            key = (
                caller_name,
                ref.get("file_path", ""),
                ref.get("line_number", 0),
            )
            if key in seen_caller_keys:
                continue
            seen_caller_keys.add(key)
            callers_data.append(ref)

        # Step 4: Fetch full bodies for callees, sorted by locality then size
        callees_result, callees_used, overflow = self._fill_related_chunks(
            db, callees_data, callees_budget, target_chunk["file_path"],
            key_field="ref_name", line_field="line_number",
            exclude_target=target_chunk,
            allow_method_suffix_fallback=True,
        )

        # Step 5: Fetch full bodies for callers
        callers_result, callers_used, overflow_callers = self._fill_related_chunks(
            db, callers_data, callers_budget, target_chunk["file_path"],
            key_field="containing_chunk", line_field="line_number",
            exclude_target=target_chunk,
            allow_method_suffix_fallback=False,
        )
        overflow.extend(overflow_callers)

        total_used = target_tokens + callees_used + callers_used

        # Step 6: Check call graph availability
        call_graph_available = bool(raw_callees or raw_callers)
        if not call_graph_available:
            try:
                call_graph_available = db.has_symbol_refs()
            except Exception:
                call_graph_available = False
        call_graph_reason = None if call_graph_available else "no_symbol_refs_indexed_or_resolved"

        result = {
            "target": {
                "file": target_chunk["file_path"],
                "name": target_name,
                "signature": target_chunk.get("signature", ""),
                "body": target_body,
                "line_start": target_chunk.get("start_line", 0),
                "line_end": target_chunk.get("end_line", 0),
                "tokens": target_tokens,
            },
            "callees": callees_result,
            "callers": callers_result,
            "overflow_signatures": overflow,
            "call_graph_available": call_graph_available,
            "call_graph_reason": call_graph_reason,
            "budget_used": total_used,
            "budget": budget,
        }

        # Step 7: Session integration — mark all returned chunks as seen
        if session_id:
            chunk_keys = [
                f"{target_chunk['file_path']}:{target_name}:{target_chunk.get('start_line', 0)}"
            ]
            token_list = [target_tokens]
            for c in callees_result:
                chunk_keys.append(f"{c['file']}:{c['name']}:{c.get('line_start', 0)}")
                token_list.append(c.get("tokens", 0))
            for c in callers_result:
                chunk_keys.append(f"{c['file']}:{c['name']}:{c.get('line_start', 0)}")
                token_list.append(c.get("tokens", 0))
            try:
                db.mark_session_seen(session_id, chunk_keys, token_list)
            except Exception as e:
                logger.debug(f"Session mark_seen failed in deep context: {e}")
            result["session_id"] = session_id

        return result

    def _collect_deep_refresh_candidates(self, db, name: str, limit: int = 25) -> List[str]:
        """Collect likely files to refresh for a deep query."""
        files: Set[str] = set()
        query = (name or "").strip()
        if not query:
            return []

        symbol = query
        # file:name or path:name hint
        if ":" in query:
            file_hint, symbol_hint = query.rsplit(":", 1)
            symbol = symbol_hint.strip() or symbol
            if "/" in file_hint or file_hint.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".swift")):
                files.add(file_hint.strip())

        lookup_names = {query, symbol}
        if "." in symbol:
            lookup_names.add(symbol.rsplit(".", 1)[-1])

        for lookup in lookup_names:
            if not lookup:
                continue
            try:
                for c in db.get_chunks_by_name(lookup, limit=limit):
                    fp = c.get("file_path")
                    if fp:
                        files.add(fp)
            except Exception:
                pass
            try:
                for c in db.get_callers(lookup, limit=limit):
                    fp = c.get("file_path")
                    if fp:
                        files.add(fp)
            except Exception:
                pass
            try:
                for c in db.get_callees(lookup, limit=limit):
                    fp = c.get("file_path")
                    if fp:
                        files.add(fp)
            except Exception:
                pass

        try:
            for r in db.search_chunks(symbol, limit=limit):
                fp = r.get("file_path")
                if fp:
                    files.add(fp)
        except Exception:
            pass

        # Keep refresh bounded.
        return sorted(files)[:limit]

    def _resolve_function(
        self, db, name: str, include_tests: bool,
    ) -> List[Dict]:
        """Resolve function name to chunk(s) with disambiguation."""
        from know.file_categories import categorize_file

        # 1. Try exact chunk_name match
        candidates = db.get_chunks_by_name(name)

        # 2. Try file:name format
        if not candidates and ":" in name:
            file_part, name_part = name.rsplit(":", 1)
            all_matches = db.get_chunks_by_name(name_part)
            candidates = [c for c in all_matches if file_part in c["file_path"]]

        # 3. Try Class.method format
        if not candidates and "." in name:
            parts = name.split(".")
            method_name = parts[-1]
            class_hint = parts[0]
            all_matches = db.get_chunks_by_name(method_name)
            # Match by class name in chunk_name (e.g., "ClassName.method")
            candidates = [
                c for c in all_matches
                if class_hint in c.get("chunk_name", "") or class_hint in c.get("file_path", "")
            ]

        if not candidates:
            return []

        # 4. Filter test files by default
        if not include_tests:
            source_only = [
                c for c in candidates
                if categorize_file(c["file_path"]) == "source"
            ]
            if source_only:
                candidates = source_only

        # 5. Prefer executable symbols over constants/modules when mixed.
        type_priority = {
            "function": 0,
            "method": 0,
            "class": 1,
            "constant": 2,
            "module": 2,
        }
        best = min(type_priority.get(c.get("chunk_type", "function"), 3) for c in candidates)
        narrowed = [
            c for c in candidates
            if type_priority.get(c.get("chunk_type", "function"), 3) == best
        ]
        if narrowed:
            candidates = narrowed

        # 6. If still ambiguous (>1), return all for disambiguation
        return candidates

    def _fill_related_chunks(
        self,
        db,
        refs: List[Dict],
        budget: int,
        target_file: str,
        key_field: str,
        line_field: str,
        exclude_target: Optional[Dict] = None,
        allow_method_suffix_fallback: bool = True,
    ) -> Tuple[List[Dict], int, List[str]]:
        """Fill budget with related chunk bodies, sorted by locality then size.

        Returns (filled_chunks, tokens_used, overflow_signatures).
        """
        if not refs:
            return [], 0, []

        # Fetch chunk bodies for all refs
        enriched = []
        for ref in refs:
            name = ref.get(key_field, "")
            if not name:
                continue

            def _exclude_target(chunks_list: List[Dict]) -> List[Dict]:
                if not exclude_target:
                    return chunks_list
                target_key = (
                    exclude_target.get("file_path"),
                    exclude_target.get("chunk_name"),
                    exclude_target.get("start_line"),
                )
                return [
                    c for c in chunks_list
                    if (
                        c.get("file_path"),
                        c.get("chunk_name"),
                        c.get("start_line"),
                    ) != target_key
                ]

            chunks = db.get_chunks_by_name(name)
            chunks = _exclude_target(chunks)
            if (
                allow_method_suffix_fallback
                and not chunks
                and "." not in name
                and hasattr(db, "get_method_chunks_by_suffix")
            ):
                # Attribute calls like `service.create_agent()` usually reference methods
                # stored as `ClassName.create_agent` chunk names.
                chunks = db.get_method_chunks_by_suffix(name)
                chunks = _exclude_target(chunks)
            if chunks:
                chunk = chunks[0]
                # Prefer same-file chunk
                same_file = [c for c in chunks if c["file_path"] == target_file]
                if same_file:
                    chunk = same_file[0]
                enriched.append({
                    "chunk": chunk,
                    "ref": ref,
                    "same_file": chunk["file_path"] == target_file,
                    "tokens": chunk.get("token_count", 0) or count_tokens(chunk.get("body", "")),
                })
            else:
                # External/unresolved reference
                enriched.append({
                    "chunk": None,
                    "ref": ref,
                    "same_file": False,
                    "tokens": 0,
                })

        # Sort: same-file first, then by smallest token count
        enriched.sort(key=lambda e: (not e["same_file"], e["tokens"]))

        filled = []
        used = 0
        overflow = []

        for entry in enriched:
            chunk = entry["chunk"]
            ref = entry["ref"]
            name = ref.get(key_field, "")

            if chunk is None:
                # External reference — add to overflow
                overflow.append(f"external: {name}")
                continue

            tokens = entry["tokens"]
            if used + tokens <= budget:
                filled.append({
                    "file": chunk["file_path"],
                    "name": chunk["chunk_name"],
                    "body": chunk.get("body", ""),
                    "tokens": tokens,
                    "line_start": chunk.get("start_line", 0),
                    "line_end": chunk.get("end_line", 0),
                    "call_site_line": ref.get(line_field, 0),
                })
                used += tokens
            else:
                # Budget overflow — add as signature only
                sig = chunk.get("signature", chunk["chunk_name"])
                overflow.append(
                    f"{sig} — {chunk['file_path']}:{chunk.get('start_line', 0)}"
                )

        return filled, used, overflow

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
        if result.get("session_id"):
            payload["session_id"] = result["session_id"]

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
