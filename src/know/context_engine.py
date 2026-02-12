"""Context engine: assembles LLM-optimized context bundles.

The killer feature.  Given a natural-language query and a token budget,
produces a single Markdown (or JSON) blob that contains the most relevant
code, its dependencies, related tests, and project overview — optimally
packed to fit the budget.

Algorithm:
  1. Semantic search → top-N function/class chunks
  2. Import expansion → pull in dependencies of relevant files
  3. Test matching → find test_*.py files for source files
  4. Git recency boost → recently-changed chunks score higher
  5. Token budgeting → greedily fill:
       40 % highest-relevance code chunks
       30 % import context (signatures only)
       20 % file-level summaries
       10 % project overview
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from know.token_counter import count_tokens, truncate_to_budget, format_budget
from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


# ---------------------------------------------------------------------------
# Embedding model caching - avoid reloading on every call
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_CACHE = None
_EMBEDDING_MODEL_LOCK = threading.Lock()

def _get_cached_embedding_model():
    """Get or create cached embedding model for semantic search (thread-safe)."""
    global _EMBEDDING_MODEL_CACHE
    if _EMBEDDING_MODEL_CACHE is None:
        with _EMBEDDING_MODEL_LOCK:
            if _EMBEDDING_MODEL_CACHE is None:
                try:
                    from fastembed import TextEmbedding
                    import time
                    start = time.time()
                    _EMBEDDING_MODEL_CACHE = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                    logger.debug(f"Loaded embedding model in {time.time()-start:.2f}s")
                except ImportError:
                    logger.debug("fastembed not available for semantic scoring")
                    return None
    return _EMBEDDING_MODEL_CACHE


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
    """Assembles an optimally-packed context bundle for LLM consumption."""

    # Budget allocation percentages
    ALLOC_CODE = 0.40
    ALLOC_IMPORTS = 0.30
    ALLOC_SUMMARIES = 0.20
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
    ) -> Dict[str, Any]:
        """Build context bundle for *query* within *budget* tokens.
        
        Returns a dict with keys:
            query, budget, used_tokens,
            code_chunks, dependency_chunks, test_chunks,
            summary_chunks, overview, warnings
        """
        # Step 0: compute sub-budgets
        budget_code = int(budget * self.ALLOC_CODE)
        budget_imports = int(budget * self.ALLOC_IMPORTS) if include_imports else 0
        budget_summaries = int(budget * self.ALLOC_SUMMARIES)
        budget_overview = int(budget * self.ALLOC_OVERVIEW)

        # Redistribute unused import budget to code
        if not include_imports:
            budget_code += int(budget * self.ALLOC_IMPORTS)

        warnings: List[str] = []

        # Step 1: collect all chunks from project
        all_chunks = self._collect_all_chunks()
        if not all_chunks:
            warnings.append("No code chunks found. Run 'know init' first.")
            return self._empty_result(query, budget, warnings)

        # Step 2: score chunks against query
        scored = self._score_chunks(query, all_chunks)

        # Step 3: git recency boost
        self._apply_recency_boost(scored)

        # Sort by combined score (descending)
        scored.sort(key=lambda c: c.score + c.recency_boost * 0.15, reverse=True)

        # Step 4: greedily fill code budget
        code_chunks: List[CodeChunk] = []
        code_used = 0
        seen_files: set = set()
        for chunk in scored:
            if code_used + chunk.tokens > budget_code:
                continue
            code_chunks.append(chunk)
            code_used += chunk.tokens
            seen_files.add(chunk.file_path)

        # Step 5: import expansion
        dep_chunks: List[CodeChunk] = []
        dep_used = 0
        if include_imports and code_chunks:
            dep_chunks, dep_used = self._expand_imports(
                seen_files, budget_imports, scored,
            )

        # Step 6: test matching
        test_chunks: List[CodeChunk] = []
        test_used = 0
        if include_tests:
            # Use leftover budget from imports
            leftover = max(0, budget_imports - dep_used) if include_imports else 0
            test_budget = leftover + max(0, budget_summaries // 4)
            test_chunks, test_used = self._find_tests(seen_files, test_budget)

        # Step 7: file-level summaries (use remaining summary budget)
        summary_budget = budget_summaries - test_used
        summary_chunks, summary_used = self._build_summaries(seen_files, max(0, summary_budget))

        # Step 8: project overview
        overview = self._project_overview(budget_overview)
        overview_tokens = count_tokens(overview)

        total_used = code_used + dep_used + test_used + summary_used + overview_tokens

        if total_used < budget * 0.2:
            warnings.append(
                f"Only {total_used} tokens of context found for a {budget} budget. "
                "The query may be too narrow or the project too small."
            )

        return {
            "query": query,
            "budget": budget,
            "used_tokens": total_used,
            "budget_display": format_budget(total_used, budget),
            "code_chunks": code_chunks,
            "dependency_chunks": dep_chunks,
            "test_chunks": test_chunks,
            "summary_chunks": summary_chunks,
            "overview": overview,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Internal: chunk collection
    # ------------------------------------------------------------------
    def _collect_all_chunks(self) -> List[CodeChunk]:
        """Collect all code chunks from the project."""
        chunks: List[CodeChunk] = []

        # Discover Python files (respecting .know config)
        from know.scanner import CodebaseScanner, LANGUAGE_EXTENSIONS
        scanner = CodebaseScanner(self.config)
        files = list(scanner._discover_files())

        for file_path, lang in files:
            if lang == "python":
                file_chunks = extract_chunks_from_file(file_path, self.root)
                chunks.extend(file_chunks)
            else:
                # Non-Python: single file-level chunk
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    rel = str(file_path.relative_to(self.root))
                    tokens = count_tokens(content)
                    chunks.append(CodeChunk(
                        file_path=rel,
                        name=file_path.stem,
                        chunk_type="module",
                        line_start=1,
                        line_end=content.count("\n") + 1,
                        body=content,
                        tokens=tokens,
                    ))
                except Exception:
                    pass

        return chunks

    # ------------------------------------------------------------------
    # Internal: scoring
    # ------------------------------------------------------------------
    def _score_chunks(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Score chunks against the query.
        
        Tries fastembed semantic search first; falls back to text matching.
        """
        try:
            return self._score_semantic(query, chunks)
        except Exception as e:
            logger.debug(f"Semantic scoring unavailable ({e}), falling back to text match")
            return self._score_text(query, chunks)

    def _score_semantic(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Score via fastembed embeddings.
        
        Uses compact text (name + signature + docstring + body[:300])
        to keep embedding fast while retaining semantic meaning.
        """
        import numpy as np

        # Use cached model to avoid reloading on every call
        model = _get_cached_embedding_model()
        if model is None:
            from fastembed import TextEmbedding
            model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Embed query
        query_emb = np.array(list(model.embed([query]))[0], dtype=np.float32)

        # Build compact search texts for efficiency
        texts = []
        for c in chunks:
            search_text = f"{c.name} {c.signature} {c.docstring} {c.body[:300]}"
            texts.append(search_text)

        # Batch embed all chunks at once
        embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

        # Cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        e_norm = embeddings / norms
        similarities = e_norm @ q_norm

        for i, chunk in enumerate(chunks):
            chunk.score = float(similarities[i])

        return chunks

    def _score_text(self, query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Fallback: score using simple text matching (word overlap)."""
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
    # Internal: recency
    # ------------------------------------------------------------------
    def _apply_recency_boost(self, chunks: List[CodeChunk]):
        """Apply git recency boost to chunks using batch operations."""
        # Collect unique files
        unique_files: set = {c.file_path for c in chunks}
        
        # Batch fetch recency scores
        recency_scores = _get_batch_file_recency(self.root, list(unique_files))
        
        # Apply to all chunks
        for chunk in chunks:
            chunk.recency_boost = recency_scores.get(chunk.file_path, 0.0)

    # ------------------------------------------------------------------
    # Internal: import expansion
    # ------------------------------------------------------------------
    def _expand_imports(
        self,
        seen_files: set,
        budget: int,
        all_chunks: List[CodeChunk],
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
                file_path=rel,
                name=f"{fpath.stem} (signatures)",
                chunk_type="module",
                line_start=1,
                line_end=sigs.count("\n") + 1,
                body=sigs,
                signature="# signatures only",
                tokens=tokens,
            ))
            used += tokens

        return dep_chunks, used

    # ------------------------------------------------------------------
    # Internal: test discovery
    # ------------------------------------------------------------------
    def _find_tests(
        self,
        seen_files: set,
        budget: int,
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
                        continue  # skip module-level, only want test functions
                    if used + chunk.tokens > budget:
                        break
                    test_chunks.append(chunk)
                    used += chunk.tokens

        return test_chunks, used

    # ------------------------------------------------------------------
    # Internal: summaries
    # ------------------------------------------------------------------
    def _build_summaries(
        self,
        seen_files: set,
        budget: int,
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

            # Build a short summary: docstring + function/class names
            lines_list = content.splitlines()
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

        # Project name from config
        overview_parts.append(f"Project: {self.config.project.name or self.root.name}")
        if self.config.project.description:
            overview_parts.append(self.config.project.description)

        # README snippet if available
        for readme_name in ["README.md", "readme.md", "README.rst"]:
            readme = self.root / readme_name
            if readme.exists():
                try:
                    content = readme.read_text(encoding="utf-8", errors="ignore")
                    # First paragraph
                    paragraphs = content.split("\n\n")
                    for p in paragraphs[:3]:
                        if p.strip() and not p.strip().startswith("!["):
                            overview_parts.append(p.strip())
                            break
                except Exception:
                    pass
                break

        overview = "\n\n".join(overview_parts)
        return truncate_to_budget(overview, budget, mode="text")

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
                lines.append(f"> ⚠️ {w}")
            lines.append("")

        # Relevant code
        if result["code_chunks"]:
            lines.append("### Relevant Code")
            lines.append("")
            for chunk in result["code_chunks"]:
                lines.append(f"#### {chunk.header()}")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        # Dependencies
        if result["dependency_chunks"]:
            lines.append("### Dependencies")
            lines.append("")
            for chunk in result["dependency_chunks"]:
                lines.append(f"#### {chunk.file_path} (signatures only)")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        # Tests
        if result["test_chunks"]:
            lines.append("### Related Tests")
            lines.append("")
            for chunk in result["test_chunks"]:
                lines.append(f"#### {chunk.header()}")
                lines.append(f"```python\n{chunk.body}\n```")
                lines.append("")

        # Summaries
        if result["summary_chunks"]:
            lines.append("### File Summaries")
            lines.append("")
            for chunk in result["summary_chunks"]:
                lines.append(chunk.body)
                lines.append("")

        # Memories
        if result.get("memories_context"):
            lines.append("### Memories (Cross-Session Knowledge)")
            lines.append("")
            lines.append(result["memories_context"])
            lines.append("")

        # Overview
        if result["overview"]:
            lines.append("### Project Context")
            lines.append("")
            lines.append(result["overview"])
            lines.append("")

        return "\n".join(lines)

    def format_agent_json(self, result: Dict[str, Any]) -> str:
        """Format context result as JSON optimized for AI agent consumption."""
        def _chunk_to_dict(c: CodeChunk) -> dict:
            return {
                "file": c.file_path,
                "name": c.name,
                "type": c.chunk_type,
                "lines": [c.line_start, c.line_end],
                "tokens": c.tokens,
                "score": round(c.score, 3),
                "body": c.body,
            }

        payload = {
            "query": result["query"],
            "budget": result["budget"],
            "used_tokens": result["used_tokens"],
            "budget_utilization": result["budget_display"],
            "warnings": result.get("warnings", []),
            "code": [_chunk_to_dict(c) for c in result["code_chunks"]],
            "dependencies": [_chunk_to_dict(c) for c in result["dependency_chunks"]],
            "tests": [_chunk_to_dict(c) for c in result["test_chunks"]],
            "summaries": [_chunk_to_dict(c) for c in result["summary_chunks"]],
            "overview": result["overview"],
            "memories": result.get("memories_context", ""),
            "source_files": list({c.file_path for c in result["code_chunks"]}),
        }
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
        }
