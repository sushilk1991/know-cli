---
title: "feat: 10x Agent Search — Hybrid Ranking, RRF Fusion, and Structural Intelligence"
type: feat
date: 2026-02-16
deepened: 2026-02-16
---

## Enhancement Summary

**Deepened on:** 2026-02-16
**Research agents used:** Python Pro, Performance Oracle, Architecture Strategist, Code Simplicity Reviewer, Agent-Native Reviewer, Data Migration Expert, Pattern Recognition Specialist, SQL/FTS5 Optimizer, Context Engineering Specialist

### Key Improvements from Deepening
1. **Critical FTS5 migration bugs found** — `PRAGMA table_info` fails on FTS5 virtual tables; triggers must be dropped/recreated to avoid column swap corruption
2. **Simplification opportunity** — in-degree count may suffice over full PageRank; RRF can be a simple function not a class; multi-query and scope features cut as premature
3. **Agent-native redesign** — remove query reformulation (agents do this better); replace scope enum with `--include`/`--exclude` primitives; add missing `signature` field to JSON output
4. **Performance validated** — full pipeline achievable in <50ms (not 500ms) after ContextEngine pivot; batch import lookups critical for metadata bundling
5. **Context engineering insights** — implement relevance floor (return under budget rather than filling with marginal results); group output by file not flat relevance; add `indexing_status` to all responses

### Simplification Decision

The **Code Simplicity Reviewer** recommended collapsing 10 steps to 3 (BM25F + file demotion, in-degree scoring, zero-result intelligence) arguing RRF and PageRank are over-engineering. The **Architecture Strategist** and **Performance Oracle** endorsed the full approach. **Decision: implement the full approach but keep abstractions minimal** — no new classes where functions suffice, no multi-query or scope enum, keep ranking.py as stateless functions.

---

# feat: 10x Agent Search — Hybrid Ranking, RRF Fusion, and Structural Intelligence

## Overview

Make `know context` the single most useful tool for AI coding agents by replacing naive BM25-only search with a multi-signal hybrid ranking system. The goal: agents call `know --json context "query" --budget 8000` once and get results so good they skip 3-10 Grep/Read cycles. Zero optional dependencies required — BM25F + import-graph signals alone should be a massive improvement.

**Why this matters:** Research shows agents waste 75K+ tokens on iterative search-then-read loops. Better ranking + structural metadata bundling eliminates most follow-up searches. This is the highest-leverage improvement for know-cli adoption.

## Problem Statement

### Current Limitations

1. **BM25 with equal field weights** — symbol names, signatures, and code bodies are all weighted equally in FTS5. A query for `verify_session` scores the same whether it matches a function definition or a passing mention in a comment.

2. **No signal fusion** — BM25, semantic search, and import-graph proximity are used independently, never combined. Each path misses what the others catch.

3. **No structural intelligence** — search results don't include imports, importers, or test files. Agents must make follow-up calls to understand context.

4. **ContextEngine bypasses the index** — `ContextEngine._collect_all_chunks()` rescans the entire filesystem on every call instead of using the daemon's pre-indexed FTS5 database. This is slow and prevents BM25F from working.

5. **No zero-result diagnostics** — when a query returns sparse or zero results, the agent gets silence instead of actionable data about what terms exist in the index.

6. **No file-category awareness** — test files, vendor code, and generated files rank equally with core source code.

### Research Insights: What Agents Actually Need

From analysis of Claude Code, Cursor, Aider, SWE-grep, and context engineering research:

- **Precision > recall** — 3-5 highly relevant chunks beat 20 somewhat-relevant ones. Chroma research shows focused prompts (~300 tokens) outperform full prompts (~113K tokens). Context poisoning is the single biggest performance killer.
- **Structural metadata bundled per-result** — imports, imported-by, test files eliminate 2-3 follow-up searches per function. This is the most agent-useful feature possible.
- **Relevance floor** — return under budget rather than filling with marginal results. Every irrelevant token actively degrades agent reasoning.
- **Group by file, order by dependency** — Microsoft/Salesforce study found 39% performance drop from interleaved presentation vs. coherent blocks. Chunks from the same file should be clustered.
- **Token budgets enforced server-side** — agents can't self-limit reliably.
- **Agents reformulate better than tools** — do NOT return query suggestions. Instead, return raw vocabulary data (`nearest_terms`, `file_names_matching`) so the agent can reason.
- **`signature` field is missing from current JSON output** — agents need this to understand functions without reading full bodies. This is a bug.
- **`indexing_status` needed in all responses** — agents cannot distinguish "nothing matches" from "still indexing."

## Proposed Solution

### Architecture: ContextEngine → DaemonDB Pivot

The foundational change: **ContextEngine stops rescanning the filesystem and instead queries DaemonDB's pre-indexed FTS5 tables.** This enables BM25F field weighting, makes search instant, and eliminates redundant file parsing.

```
BEFORE:
  know context "query" → ContextEngine → scan all files → AST parse → word-overlap score → budget fill
  (2-5 seconds for 500 files, 10+ seconds for 2000 files)

AFTER:
  know context "query" → ContextEngine → DaemonDB.ranked_search() → score fusion → budget fill + metadata
  (<50ms for any indexed project)
```

ContextEngine becomes a thin orchestrator (Mediator pattern): query → rank → budget → format. The heavy lifting (indexing, FTS5, import graph) lives in the daemon.

### Research Insights: Architecture

- **ContextEngine is already a borderline God Class** with 8 responsibilities (chunk collection, scoring, recency boosting, import expansion, test discovery, summary building, overview generation, output formatting). The pivot should extract scoring into `ranking.py` AND formatting into its own concern. ContextEngine becomes an orchestrator that delegates to: (a) DaemonDB for retrieval, (b) ranking functions for scoring, (c) formatter for output.
- **Inject DaemonDB, don't construct it.** Currently `ImportGraph.__init__`, `KnowledgeBase.__init__`, and `ContextEngine._expand_imports` all create their own `DaemonDB` instances (multiple connections). Pass a shared instance instead.
- **Fallback should be DaemonDB used directly (no daemon socket), NOT filesystem scanning.** The daemon process is a performance optimization; the SQLite database is always available. If the DB file itself is missing (fresh project), trigger synchronous index into DaemonDB.
- **The existing daemon-first/fallback pattern** from `cli/agent.py` (`_get_daemon_client` / `_get_db_fallback`) should be extracted into a shared utility used by ContextEngine too.

### Signal Stack (All Zero-Dependency)

| Signal | Source | What It Catches | Implementation |
|--------|--------|----------------|----------------|
| **BM25F** | FTS5 with field weights | Exact identifiers, API names | `bm25(chunks_fts, 5.0, 5.0, 3.0, 1.0)` |
| **In-degree / PageRank** | Import graph | Structurally important files | In-degree count or power iteration |
| **File category** | Path heuristics | Demote test/vendor/generated | Score multiplier per category |
| **Git recency** | `git log` (existing) | Recently changed code | Existing 15% boost |
| **Semantic** | fastembed (optional) | Conceptual similarity | Existing cosine similarity |

### Research Insights: Simplification Decision

The **Simplicity Reviewer** argued that in-degree count gives 90% of PageRank's signal with zero iteration. This is valid for most codebases. **Decision: implement in-degree first as the fast path. If quality testing shows meaningful improvement from full PageRank, add it behind a config flag.** The in-degree approach is 3 lines of SQL, zero new code.

### Score Fusion

Instead of a full RRF class, use a stateless function that accepts generic ranked lists:

```python
def fuse_rankings(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked lists. Stateless, pure function."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Research Insights: Ranking

- **ranking.py should be stateless and pure** — accept ranked lists as input, produce fused rankings as output. No DB connections, no filesystem access. This makes it trivially testable.
- **Do NOT hardcode signal names in the fusion function signature.** Accept generic `list[list[tuple]]` so new signals can be added without modifying fusion code.
- **Consolidate the 4 existing scoring implementations** (ContextEngine semantic, ContextEngine text, SemanticSearcher cosine, KnowledgeBase word-overlap) as part of this work. The plan MUST replace these with the new pipeline.
- **Keep ranking invisible to agents.** Do not expose RRF weights or algorithm choice as parameters. The agent should not need to know or care which ranking algorithm runs.

## Technical Approach

### Phase 1: Foundation — BM25F + ContextEngine Pivot

**Goal:** Make the zero-dependency search path dramatically better. This phase alone delivers ~80% of the improvement.

#### 1a. FTS5 Schema Migration

**File:** `src/know/daemon_db.py`

Add `file_path` to the FTS5 virtual table and add `prefix` indexes for autocompletion:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_name,
    file_path,
    signature,
    body,
    content='chunks',
    content_rowid='id',
    prefix='2,3'
);
```

BM25F weights via `bm25()` auxiliary function (weights are query-time, not schema-time):

```sql
SELECT c.*, -bm25(chunks_fts, 5.0, 5.0, 3.0, 1.0) AS score
FROM chunks_fts
JOIN chunks c ON chunks_fts.rowid = c.id
WHERE chunks_fts MATCH ?
ORDER BY score DESC
LIMIT ?
```

Weights: `chunk_name=5.0, file_path=5.0, signature=3.0, body=1.0`

**Performance:** BM25F with explicit weights is negligible overhead (<0.1ms) — it's just extra floating-point multiplications per result row. The `bm25()` function is O(matches), not O(table_size).

##### Research Insights: Critical Migration Issues

**CRITICAL BUG #1: `PRAGMA table_info` FAILS on FTS5 virtual tables.** It returns an empty result set. The plan originally said "detect old schema via PRAGMA table_info on chunks_fts" — this will silently fail, and the migration will never run.

**Fix:** Use `sqlite_master` to inspect the CREATE TABLE statement:

```python
def _needs_fts_migration(self, conn: sqlite3.Connection) -> bool:
    """Check if FTS5 needs migration. PRAGMA table_info fails on virtual tables."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    if row is None:
        return False  # Table doesn't exist; schema script will create it
    return 'file_path' not in row[0]
```

**CRITICAL BUG #2: Triggers MUST be dropped and recreated.** The content-sync triggers (`chunks_ai`, `chunks_ad`, `chunks_au`) reference the old 3-column schema. After recreating `chunks_fts` with 4 columns, old triggers insert into wrong column positions — `file_path` values go into `signature`, `signature` values go into `body`. Search results silently degrade with swapped data.

**CRITICAL BUG #3: BM25 weight vector silently degrades on old schema.** If migration fails and the old 3-column table persists, `bm25(chunks_fts, 5.0, 5.0, 3.0, 1.0)` passes 4 weights to a 3-column table. SQLite FTS5 silently ignores extra weights. Result: `chunk_name=5.0, signature=5.0, body=3.0` — `file_path` is not indexed at all. Add runtime detection.

**Correct migration sequence:**

```python
def _migrate_fts_schema(self, conn: sqlite3.Connection):
    """Migrate FTS5 from 3 columns to 4 columns (add file_path)."""
    if not self._needs_fts_migration(conn):
        return

    # MUST drop triggers FIRST (they reference old column positions)
    conn.execute("DROP TRIGGER IF EXISTS chunks_ai")
    conn.execute("DROP TRIGGER IF EXISTS chunks_ad")
    conn.execute("DROP TRIGGER IF EXISTS chunks_au")

    # Drop FTS5 virtual table (cascades to shadow tables)
    conn.execute("DROP TABLE IF EXISTS chunks_fts")

    # NOTE: Do NOT use executescript() here — it implicitly COMMITs,
    # breaking transaction control. Use individual execute() calls.

    # Schema script's CREATE IF NOT EXISTS will now create the new table.
    # After schema creation, rebuild index from content table:
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
```

**Use `rebuild` command, not manual INSERT...SELECT.** For content-sync tables, `INSERT INTO fts_table(fts_table) VALUES('rebuild')` is the only correct way to repopulate. It reads from the content table using column names (not positions), so it handles the new schema correctly.

**Schema version tracking:** Add a `schema_version` table to avoid ad-hoc detection for future migrations:

```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at REAL NOT NULL
);
```

##### Research Insights: FTS5 Optimization

- **Column-specific MATCH queries** for targeted search: `chunk_name:daemon` searches only symbol names. Useful for symbol-focused queries.
- **`fts5vocab` virtual tables** provide free diagnostics — create them alongside the FTS table for zero-result intelligence:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab(chunks_fts, row);
```

- **Default `unicode61` tokenizer is good for code** — it splits `snake_case` and `dotted.paths` naturally on `_` and `.`. No need for `tokenchars` override. However, **CamelCase is NOT split** — `parseJsonResponse` becomes one token. Pre-process in Python if needed:

```python
import re
def expand_camel(name: str) -> str:
    """'parseJsonResponse' → 'parseJsonResponse parse Json Response'"""
    parts = re.sub(r'([A-Z])', r' \\1', name).split()
    return name + ' ' + ' '.join(p.lower() for p in parts)
```

- **`prefix='2,3'`** pre-builds prefix indexes for 2- and 3-character prefixes. Makes prefix autocompletion O(1) instead of scanning the full token list. Small index size increase (~2-3x), worthwhile for a single-project CLI tool.

#### 1b. ContextEngine → DaemonDB Pivot

**File:** `src/know/context_engine.py`

Replace `_collect_all_chunks()` + `_score_chunks()` with DaemonDB query:

```python
def _ranked_search(self, query: str, limit: int = 50) -> list[dict]:
    """Get ranked chunks from daemon DB instead of rescanning filesystem."""
    db = self._get_daemon_db()
    if db:
        return db.search_chunks_ranked(query, limit=limit)
    # Fallback: direct DaemonDB if daemon socket unavailable
    return self._direct_db_search(query, limit=limit)
```

**Fallback is direct DaemonDB, NOT filesystem scan.** The filesystem scan is kept only as a `--legacy` CLI flag for debugging.

##### Research Insights: Performance Impact

- **This is the single highest-value change.** Current `_collect_all_chunks()` is O(n) file reads + O(n) AST parses. For 500 files: 2-5 seconds. For 2000 files: 10+ seconds.
- **After pivot: <15ms for 500 files, <25ms for 2000 files, <50ms for 10,000 files.** All queries become O(log n) FTS5 lookups against pre-indexed data.
- **Use pre-computed `token_count` from the chunks table** instead of calling `count_tokens()` (tiktoken) during budget fill. The counts are already stored — no need to re-encode.
- **Empty results during indexing:** The fallback should detect `get_stats()["files"] == 0` and either wait briefly for indexing or return an `indexing_status` field so the agent knows to retry.

##### Research Insights: Architecture

- **Extract formatting into a separate concern.** The current `format_agent_json` at `context_engine.py:832` and `format_markdown` at `context_engine.py:768` should move out of ContextEngine.
- **Follow the daemon-first/fallback pattern from `cli/agent.py`** — extract `_get_daemon_client` / `_get_db_fallback` into a shared utility.

#### 1c. File Category Demotion

**File:** `src/know/file_categories.py` (new utility module, NOT in `daemon_db.py`)

```python
FILE_CATEGORY_PATTERNS = {
    "test": ["test_*", "*_test.*", "tests/", "spec/", "__tests__/", "*_spec.*"],
    "vendor": ["vendor/", "third_party/", "node_modules/", ".venv/"],
    "generated": ["*_generated.*", "*.pb.*", "*_pb2.py", "generated/"],
}

DEMOTION_MULTIPLIERS = {
    "source": 1.0,
    "test": 0.3,
    "vendor": 0.1,
    "generated": 0.1,
}

def categorize_file(file_path: str) -> str:
    """Return 'source', 'test', 'vendor', or 'generated'."""

def apply_category_demotion(chunks: list[dict], query: str) -> list[dict]:
    """Demote non-source chunks. If query contains 'test', skip test demotion."""
```

##### Research Insights: Separation of Concerns

- **File category detection should NOT live in `daemon_db.py`** — that is a pure data access layer. Categories are a classification concern.
- **Apply at indexing time** and store as a column in the `chunks` table. Then the demotion is just a score multiplier in the ranking pipeline, not a post-query filter.
- **The `scanner.py` module already has file classification heuristics** (line 391) — reuse those patterns.

### Phase 2: Graph Intelligence — Structural Scoring + Metadata Bundling

#### 2a. Import Graph Structural Scoring

**File:** `src/know/import_graph.py` (new method)

**Start with in-degree count** (number of files that import this module). This is 90% of PageRank's signal with zero iteration:

```sql
SELECT target_module, COUNT(*) as in_degree
FROM imports
GROUP BY target_module
```

Store as a column in a `module_importance` table or inline in the import graph:

```python
def compute_importance(self) -> dict[str, float]:
    """In-degree count as structural importance signal."""
    conn = self._db._get_conn()
    rows = conn.execute(
        "SELECT target_module, COUNT(*) FROM imports GROUP BY target_module"
    ).fetchall()
    max_deg = max((r[1] for r in rows), default=1)
    return {r[0]: r[1] / max_deg for r in rows}  # Normalize to 0-1
```

**If quality testing shows meaningful improvement from full PageRank**, add power iteration:

```python
def compute_pagerank(self, damping: float = 0.85, iterations: int = 20) -> dict[str, float]:
    """Pure-Python PageRank using power iteration on sparse adjacency dict.
    Memory: O(N) for ranks + O(E) for graph. <5ms for 10K modules."""
    # Precompute reverse graph for efficient iteration
    in_edges: dict[str, list[str]] = {node: [] for node in nodes}
    out_degree: dict[str, int] = {}
    for node, targets in graph.items():
        out_degree[node] = len(targets)
        for target in targets:
            if target in in_edges:
                in_edges[target].append(node)
    # ... power iteration with dangling node handling ...
```

##### Research Insights: Performance

- **Power iteration on 10K modules × 20 iterations < 5ms** in pure Python with dict-based adjacency lists. Convergence is fast because import graphs are sparse and nearly acyclic.
- **20 iterations is more than sufficient** — achieves L1 residual below 1e-6 for graphs under 50K nodes.
- **Compute once per index cycle, cache in memory.** Do NOT recompute per query. Store in DaemonDB for persistence across daemon restarts.
- **`module_pagerank` table schema:**

```sql
CREATE TABLE IF NOT EXISTS module_pagerank (
    module_name TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.0,
    computed_at REAL NOT NULL
);
```

- **File path to module name mapping needed:** `src/know/daemon_db.py` → `src.know.daemon_db`

#### 2b. Score Fusion

**File:** `src/know/ranking.py` (new module — stateless functions only, no classes)

```python
"""Score fusion for combining multiple ranking signals.

This module contains ONLY pure functions. No DB connections,
no filesystem access, no side effects. Trivially testable.
"""

def fuse_rankings(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def apply_relevance_floor(
    chunks: list[dict],
    top_score_ratio: float = 0.3,
) -> list[dict]:
    """Drop chunks below relevance floor. Return under budget rather than filling with noise."""
    if not chunks:
        return chunks
    top_score = chunks[0].get("score", 1.0)
    floor = top_score * top_score_ratio
    return [c for c in chunks if c.get("score", 0) >= floor]
```

##### Research Insights: Relevance Floor

- **This is the single highest-impact quality improvement** according to context engineering research. Chroma's study shows that irrelevant context linearly degrades performance from the first irrelevant result.
- **Do NOT fill the budget with marginal results.** If only 1200 tokens of genuinely relevant code exist, return 1200 tokens, not 8000 with filler.
- **Add a `confidence` field to JSON output** — ratio of budget used to budget available. Low confidence means the query had sparse matches.

#### 2c. Structural Metadata Bundling

**File:** `src/know/context_engine.py`

Each code chunk in the JSON output gains metadata fields:

```json
{
  "file": "src/auth/session.py",
  "name": "verify_session",
  "type": "function",
  "signature": "def verify_session(token: str) -> User:",
  "lines": [42, 68],
  "score": 0.92,
  "tokens": 245,
  "imports": ["src/auth/tokens.py", "src/models/user.py"],
  "imported_by": ["src/api/middleware.py", "src/api/routes.py"],
  "test_file": "tests/test_auth.py",
  "truncated": false,
  "body": "def verify_session(token: str) -> User: ..."
}
```

**Key changes from original plan:**
- **`signature` field added** — currently missing from `_chunk_to_dict`. Agents need this to understand functions without reading full bodies.
- **`truncated` field added** — if a chunk body was truncated to fit budget, the agent needs to know so it can request the full file.
- **`body` is LAST** — agents that hit context limits truncate from the end. Put parseable fields first.
- **`test_file` singular** (most files have one test file) instead of array, to save tokens.

**Budget allocation updated:**

```
60% Code chunks (with inline metadata)
15% Dependency signatures (reduced — metadata covers key deps)
15% File summaries
10% Project overview (or less — Aider proves 1K tokens suffices)
```

##### Research Insights: Metadata Implementation

- **Batch import lookups are critical.** The current `get_imports_of()` and `get_imported_by()` do single-module queries. For 20 result chunks across 8 files, that's 16 DB queries. Worse: the fallback path calls `get_all_edges()` which fetches the entire edge table — up to 40 times.

**Add batch methods to DaemonDB:**

```python
def get_imports_batch(self, modules: list[str]) -> dict[str, list[str]]:
    """Get imports for multiple modules in one query."""
    placeholders = ','.join('?' * len(modules))
    rows = conn.execute(
        f"SELECT source_module, target_module FROM imports WHERE source_module IN ({placeholders})",
        modules,
    ).fetchall()
    result: dict[str, list[str]] = {}
    for s, t in rows:
        result.setdefault(s, []).append(t)
    return result
```

- **Cache `get_all_edges()` result** — it doesn't change between queries. Add a 60-second TTL cache.
- **Build a suffix index** at import graph build time so fallback path doesn't scan all edges.
- **Test file association via import graph** — test files import the modules they test, so the import graph already captures this in reverse. No filesystem glob needed.

##### Research Insights: Context Structure

- **Group chunks by file, not flat relevance.** An agent seeing `DaemonDB._get_conn` needs the class definition and `_SCHEMA` in the same cluster, not scattered across a flat list. Within each file group: definition before usage. Between files: dependency order (if A imports B, show B first).
- **Compress metadata format for within-budget efficiency:**

```
# In markdown output (for humans), use compact annotations:
# deps: sqlite3, pathlib, typing
# used-by: knowledge_base.py, daemon.py
# tests: test_daemon_db.py
```

### Phase 3: Agent Intelligence — Zero-Result Diagnostics + Filtering

**Revised scope:** Removed query reformulation hints (agents reformulate better themselves) and multi-query batch (not how agents work). Replaced scope enum with primitives.

#### 3a. Zero-Result Intelligence

**File:** `src/know/daemon_db.py` (new methods using `fts5vocab`)

When zero results or sparse results are returned, provide raw data for agent reasoning:

```json
{
  "code": [],
  "index_stats": {
    "total_chunks": 847,
    "total_files": 42,
    "indexing_status": "complete"
  },
  "nearest_terms": ["authenticate", "auth_handler", "AuthService"],
  "file_names_matching": ["src/auth/handler.py", "src/auth/service.py"]
}
```

**Implementation using `fts5vocab`:**

```python
def get_nearest_terms(self, query: str, limit: int = 5) -> list[str]:
    """Find terms in the FTS index closest to query terms."""
    conn = self._get_conn()
    terms = self._tokenize(query)
    results = []
    for term in terms:
        # Use fts5vocab for prefix matching in the actual index
        rows = conn.execute(
            "SELECT term, doc FROM chunks_fts_vocab "
            "WHERE term >= ? AND term < ? ORDER BY doc DESC LIMIT ?",
            (term[:3], term[:3] + '\uffff', limit),
        ).fetchall()
        results.extend(r[0] for r in rows)
    return list(dict.fromkeys(results))[:limit]  # Deduplicate, preserve order
```

**As a fallback, use `difflib.get_close_matches`** (stdlib, zero dependencies):

```python
from difflib import get_close_matches

def suggest_symbols(self, query: str, limit: int = 5) -> list[str]:
    """Fuzzy match query against known symbol names."""
    vocabulary = self._get_symbol_vocabulary()  # Cached SELECT DISTINCT chunk_name
    return get_close_matches(query, vocabulary, n=limit, cutoff=0.6)
```

##### Research Insights: Zero-Result Design

- **Return data, not explanations.** "No results found for 'auth middleware'. The codebase uses 'auth_handler' instead." is a Workflow anti-pattern — you're encoding diagnostic reasoning. Instead, return `nearest_terms` and `file_names_matching` so the agent can reason.
- **`indexing_status` in every response** — agents cannot distinguish "nothing matches" from "still indexing". This is Context Starvation.
- **FTS5 prefix queries are O(log n)** — use them for nearest-term lookup. Do NOT run Levenshtein over 10K symbols (50-200ms in pure Python). If fuzzy matching is needed, use `difflib.get_close_matches` which has early-rejection optimization, or `rapidfuzz` as optional dependency.

#### 3b. File Path Filtering (replaces scope parameter)

**File:** `src/know/cli/search.py`

Instead of a `--scope` enum (which forces know-cli to define what "definitions" means across languages), provide primitives:

```bash
# Include only source files
know --json context "auth" --include "src/**" --exclude "tests/**"

# Include only test files
know --json context "auth" --include "tests/**"

# Filter by chunk type
know --json context "auth" --chunk-types function,class
```

This is more flexible than `--scope definitions` and lets the agent control filtering directly. Document convenience patterns in MCP `instructions` field:

```
To search only definitions, use: --chunk-types function,class
To search only tests, use: --include "tests/**"
To search only types, use: --chunk-types class
```

## Acceptance Criteria

### Phase 1 (Foundation)

- [x] FTS5 schema includes `file_path` column with auto-migration using `sqlite_master` detection
- [x] Migration drops and recreates triggers (not just FTS table)
- [x] `schema_version` table tracks migrations
- [x] `search_chunks()` uses `bm25(chunks_fts, 5.0, 5.0, 3.0, 1.0)` weights with runtime detection for old schemas
- [x] `know --json context "query"` uses DaemonDB instead of filesystem scan
- [x] Direct DaemonDB fallback when daemon socket unavailable (NOT filesystem scan)
- [x] `--legacy` flag triggers filesystem scan for debugging
- [x] Test/vendor/generated files demoted via score multiplier (not rank shifting)
- [x] File category detection in separate `file_categories.py` module
- [x] Pre-computed `token_count` used (not tiktoken re-encoding)
- [x] All 169 existing tests pass (195 total now)
- [x] Manual validation: `know context "daemon socket"` returns 13 relevant chunks from daemon.py

### Phase 2 (Graph Intelligence)

- [x] In-degree importance computed during indexing, stored in DB
- [x] Score fusion via stateless `fuse_rankings()` function in `ranking.py`
- [x] Relevance floor: drops chunks below 30% of top score
- [x] Each code chunk in JSON includes `signature`, `imports`, `imported_by`, `test_file`, `truncated`
- [x] `body` field is last in JSON object for truncation safety
- [x] Batch import lookups (not N+1 queries)
- [x] Batch lookups replace per-chunk queries (better than edge caching)
- [x] Chunks grouped by file in output, dependency-ordered between files
- [x] Budget allocation: 60/15/15/10
- [x] Graceful degradation: works with any subset of signals
- [x] `indexing_status` field in every JSON response

### Phase 3 (Agent Intelligence)

- [x] Zero/sparse results return `nearest_terms` and `file_names_matching` from fts5vocab
- [x] `--include`/`--exclude` glob filtering on file paths
- [x] `--chunk-types` filtering (function, class, method, module)
- [x] All new features work without fastembed installed
- [x] 4 existing scoring implementations consolidated into ranking pipeline

## Success Metrics

- **Search quality:** Manual evaluation on 10 representative queries shows majority of results are directly relevant (vs. current ~50% relevance)
- **Agent token savings:** A typical "find and fix" workflow needs 1-2 `know context` calls instead of 5-10 Grep/Read cycles
- **Zero-dependency improvement:** BM25F + in-degree + category demotion (no fastembed) produces meaningfully better results than current BM25-only
- **Latency:** `know context` returns in <50ms for indexed projects (validated by Performance Oracle analysis)

| Repo Size | Expected Latency |
|-----------|-----------------|
| 500 files | <15ms |
| 2,000 files | <25ms |
| 10,000 files | <50ms |

## Dependencies & Prerequisites

- Phase 1 blocks Phase 2 (ContextEngine must use DaemonDB before fusion can be added)
- Phase 3a (zero-result) can be implemented independently of Phase 2
- Phase 3b (filtering) can be implemented independently

## Risk Analysis & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| FTS5 schema migration corrupts index (trigger column swap) | Silent wrong results | Drop triggers BEFORE FTS table; use `rebuild` command; add runtime schema detection |
| `PRAGMA table_info` fails silently on FTS5 | Migration never runs | Use `sqlite_master` SQL inspection instead |
| Migration interrupted mid-transaction | Broken FTS table | Use individual `execute()` calls (not `executescript()` which auto-commits); if FTS missing on startup, delete and re-create DB |
| ContextEngine pivot introduces regressions | Wrong results | Keep filesystem scan as `--legacy` fallback; extensive testing |
| Metadata bundling N+1 queries | Slow metadata lookup | Batch import lookups; cache `get_all_edges()`; suffix index for fallback path |
| Non-Python projects get no graph signals | Degraded ranking | Fusion gracefully degrades; BM25F alone is still better than BM25 |
| JSON schema changes break agents | Agent failures | Additive changes only; new fields, no removed fields |

## Implementation Order

```
Phase 1a: FTS5 schema migration + BM25F weights + schema_version table
Phase 1b: ContextEngine → DaemonDB pivot + shared fallback utility
Phase 1c: File category detection + demotion multipliers
Phase 2a: In-degree importance computation + storage
Phase 2b: Score fusion function (ranking.py) + relevance floor
Phase 2c: Structural metadata bundling + batch lookups + output restructuring
Phase 3a: Zero-result intelligence (fts5vocab + nearest_terms)
Phase 3b: File path filtering (--include/--exclude/--chunk-types)
```

Phase 1 is the most impactful and should ship first. Phase 2 builds on it. Phase 3 is polish.

## What Was Cut (and Why)

| Feature | Reason for Cutting |
|---------|-------------------|
| Query reformulation hints | Agents reformulate better themselves. Return raw vocabulary data instead. |
| Multi-query batch support | Not how agents work. They think-act-observe in single queries. Agents can make concurrent CLI calls at the transport level. |
| `--scope` enum (definitions/tests/types) | Wrong abstraction. `--include`/`--exclude`/`--chunk-types` are composable primitives that agents control directly. |
| `RankFusion` class | Over-abstraction. A stateless `fuse_rankings()` function is sufficient and more testable. |
| `ScoredChunk` type | Existing dicts work fine. No need for a new type. |

## Future Considerations

- **Call-graph analysis** — tree-sitter-based function call extraction for `--chunk-types call_site`
- **Non-Python import graphs** — extend import graph to TypeScript, Go, Rust via tree-sitter
- **Learned ranking** — track implicit feedback: correlate returned results with subsequent file edits via git. `stats.db` already exists for this.
- **Map mode** (`--map`) — condensed signatures-only view (~30% of budget) for progressive disclosure. Agent sees the map, decides what to expand. Matches Aider's architecture.
- **Diff-aware context** (`--diff` / `--staged`) — focus search on modified files and their dependents for PR review workflows
- **`read_file` MCP tool** — close the action parity gap for MCP-only environments (Claude Desktop) where agents have no filesystem access
- **Remove `explain_component` MCP tool** — it makes an LLM call inside a tool called by an LLM (anti-pattern). The agent IS an AI; it doesn't need another AI call inside its tool.

## References

### Internal

- Previous plan: `docs/plans/2026-02-15-refactor-know-cli-v2-daemon-architecture-plan.md`
- Fix plan: `docs/plans/2026-02-16-fix-testing-issues-search-graph-performance-plan.md`
- Architectural learnings: `docs/solutions/architecture/p2-p3-architectural-improvements.md`
- Key files:
  - `src/know/context_engine.py` — ContextEngine (main refactor target)
  - `src/know/daemon_db.py` — DaemonDB, FTS5 schema, search_chunks
  - `src/know/import_graph.py` — import graph building
  - `src/know/daemon.py` — daemon indexing lifecycle
  - `src/know/semantic_search.py` — fastembed integration
  - `src/know/ranking.py` — new score fusion module (to be created)
  - `src/know/file_categories.py` — new file categorization module (to be created)
  - `src/know/cli/agent.py` — next-file, signatures, related commands
  - `src/know/cli/search.py` — search, context commands

### External Research

- [Sourcegraph BM25F](https://sourcegraph.com/blog/keeping-it-boring-and-relevant-with-bm25f) — field-weighted BM25 in production code search
- [Aider repo map](https://aider.chat/docs/repomap.html) — tree-sitter + PageRank for token-efficient code context
- [Cognition SWE-grep](https://cognition.ai/blog/swe-grep) — parallel multi-turn search, precision > recall
- [OpenSearch RRF](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/) — RRF in production hybrid search
- [LongCodeZip](https://arxiv.org/html/2510.00446v1) — code-specific context compression (5.6x ratio)
- [Agent READMEs study](https://arxiv.org/html/2511.12884v1) — what context agents actually need
- [Milvus: Why grep burns tokens](https://milvus.io/blog/why-im-against-claude-codes-grep-only-retrieval-it-just-burns-too-many-tokens.md) — agent token waste analysis
- [Anthropic: Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — context structure and relevance floor
- [Chroma: Context Rot](https://research.trychroma.com/context-rot) — irrelevant context degrades performance linearly
- [Dbreunig: How Long Contexts Fail](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) — every extra tool/document hurts
- [Factory.ai: Context Window Problem](https://factory.ai/news/context-window-problem) — structural blindness in flat retrieval
