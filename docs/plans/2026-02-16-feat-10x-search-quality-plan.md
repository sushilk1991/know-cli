---
title: "feat: 10x Search Quality — Query Understanding, Dual-Lane Search, Context Expansion"
type: feat
date: 2026-02-16
---

# feat: 10x Search Quality

## Problem

In an honest A/B test, an agent using know-cli performed **worse** than one using Grep+Read:

| Metric | Agent with know | Agent with Grep+Read |
|--------|----------------|---------------------|
| API tokens | 70,542 | 75,602 |
| Wall time | 114s | **89s** |
| Tool calls | 13 | 12 |
| Quality | Accurate | **More precise** |

The existing plan (Phase 1-3 BM25F, category demotion, metadata bundling) is complete but insufficient. The fundamental issues are deeper.

## Root Causes

### 1. No query understanding
`_build_fts_query("fix the auth bug")` → `"fix" OR "the" OR "auth" OR "bug"`. Stop words like "the" match everything. An agent using grep is smarter — it ignores "fix the" and greps for `auth`.

### 2. OR-only matching = low precision
Every chunk containing "the" anywhere scores > 0. BM25 can't save you when the query is 75% noise. Need AND-boosted search + exact substring matching.

### 3. Isolated chunks lack narrative
know returns a naked function. Read returns the full file — imports, class, adjacent functions, constants. The agent gets the story vs fragments.

### 4. Importance scores computed but never used
`module_importance` table exists, `compute_importance()` runs, but `_build_context_v3_inner()` never reads it. Core modules rank the same as leaf files.

### 5. Git recency only in legacy path
`_get_batch_file_recency()` exists but v3 pipeline never calls it.

### 6. Static budget allocation wastes tokens
Fixed 60/15/15/10 split. When agent asks "find verify_session", 10% goes to project overview (useless), 15% to summaries (useless).

## Solution: 7 Phases, Priority Order

### Phase 1: Query Understanding (P0 — fixes the worst failures)

**New file: `src/know/query.py`**

- [x] Stop-word removal: strip "fix", "the", "help", "me", "how", "does", "with", "this", "add", "create", "update", "change" etc (50-100 common agent words)
- [x] Identifier detection: tokens with `_` or camelCase are code identifiers → exact-match boost
- [x] CamelCase/snake_case splitting: `AuthMiddleware` → also search "auth", "middleware"
- [x] Agent prefix stripping: remove "help me", "how do I", "find the", "show me"
- [x] Query type classification: "identifier" vs "concept" vs "error" (drives budget allocation)
- [x] If after stop-word removal < 2 terms remain, fall back to original terms

**Modify: `src/know/daemon_db.py` `_build_fts_query()`**
- [x] Replace raw whitespace split with `analyze_query()` from `query.py`
- [x] Use column-specific matching for identifiers: `chunk_name:"auth" OR signature:"auth"` (not body noise)

### Phase 2: Dual-Lane Search with AND Boosting (P0 — precision parity with grep)

**Modify: `src/know/daemon_db.py` `search_chunks()`**

- [x] Lane 1: FTS5 BM25F with smart query (from Phase 1)
- [x] Lane 2: AND query — all meaningful terms must appear. If > 3 results, these rank highest
- [x] Lane 3: Exact substring LIKE match on `chunk_name` for identified identifiers
- [x] Fuse lanes via existing `fuse_rankings()` (already in `ranking.py`, currently unused in v3!)
- [x] AND results get 2x weight in RRF, exact matches get 3x

### Phase 3: Wire Unused Signals into v3 Pipeline (P1 — low effort, already computed)

**Modify: `src/know/context_engine.py` `_build_context_v3_inner()`**

Between step 2 (category demotion) and step 5 (relevance floor):

- [x] Step 2.5: Importance boost — read `module_importance` table, boost high-in-degree modules by up to 50%
- [x] Step 2.6: Git recency boost — call existing `_get_batch_file_recency()`, boost recent files by up to 20%
- [x] Step 2.7: File-path exact match boost — if query terms match file path components, 2x boost

### Phase 4: Context Expansion — Return Neighborhoods, Not Fragments (P1 — quality parity with Read)

**Modify: `src/know/context_engine.py`**

After greedy budget fill (step 7), for each selected chunk:

- [ ] Include module-level chunk (imports + module docstring) from same file, if budget allows
- [ ] If method, include parent class signature
- [ ] Include adjacent chunks (within 10 lines) if budget allows
- [ ] Group same-file chunks into a single block with line gaps preserved

This gives agents the "story" instead of fragments.

### Phase 5: Adaptive Budget Allocation (P1 — stop wasting tokens)

**Modify: `src/know/context_engine.py`**

- [x] Replace static 60/15/15/10 with query-type-driven allocation:
  - Identifier query: 80% code / 15% imports / 5% summaries / 0% overview
  - Concept query: 55% code / 15% imports / 20% summaries / 10% overview
  - Error query: 60% code / 10% imports / 10% summaries / 0% overview / 20% tests
- [x] Use SearchPlan.query_type from Phase 1

### Phase 6: Chunk Deduplication (P1 — eliminate waste)

**Modify: `src/know/context_engine.py`**

- [ ] If a class chunk and its method chunks are both selected, keep only the more specific match
- [ ] Class body contains method bodies → skip method chunks already inside a selected class
- [ ] Or vice versa: if 3 methods selected from same class, skip the class body chunk

### Phase 7: Call Graph from Tree-sitter (P2 — the 10x moat grep cannot replicate)

**New table in `src/know/daemon_db.py`:**

```sql
CREATE TABLE IF NOT EXISTS symbol_refs (
    file_path TEXT NOT NULL,
    ref_name TEXT NOT NULL,
    ref_type TEXT NOT NULL,  -- 'call', 'attribute', 'import'
    line_number INTEGER NOT NULL,
    containing_chunk TEXT NOT NULL
);
```

- [ ] During `populate_index()`, walk Tree-sitter AST to extract function call references
- [ ] Store in `symbol_refs` table with indexes on `ref_name` and `containing_chunk`
- [ ] Add `get_callers(chunk_name)` and `get_callees(chunk_name)` to DaemonDB
- [ ] In ranking: boost chunks that call or are called by top matches (dependency proximity)
- [ ] New CLI command: `know callers <function_name>` — what grep fundamentally cannot do

## Acceptance Criteria

### Phase 1+2 (Query + Search)
- [x] Query "fix the auth bug" returns auth-related code, not random chunks containing "the"
- [x] Query "verify_session" finds exact function definition in top 3 results
- [x] `fuse_rankings()` is actually called in v3 pipeline (currently dead code)
- [x] Stop-word list tested against 20 real agent queries

### Phase 3 (Wire signals)
- [x] `module_importance` scores affect chunk ordering
- [x] Recently git-modified files rank higher
- [x] File path matches boost results

### Phase 4+5+6 (Context + Budget)
- [ ] Selected method chunks include parent class signature
- [ ] Same-file chunks grouped with imports visible
- [x] Budget allocation varies by query type
- [ ] No duplicate code in output (class + method overlap eliminated)

### Phase 7 (Call graph)
- [ ] `symbol_refs` table populated during indexing
- [ ] `know callers verify_session` returns list of calling functions
- [ ] Call graph is cross-file (not just within-file)

## Verification: Re-run A/B Test

After implementation, re-run the exact same A/B test:
- Same task: "explain how the search ranking pipeline works and write a README section"
- Same two agent configurations
- **Target: Agent with know uses 50% fewer API tokens and completes in equal or less time**

## Implementation Order

```
Phase 1: Query understanding (query.py + _build_fts_query)     — 1 day
Phase 2: Dual-lane search + AND + RRF fusion                  — 1 day
Phase 3: Wire importance + recency + path boost into v3        — 0.5 day
Phase 4: Context expansion (neighborhoods)                     — 1 day
Phase 5: Adaptive budget allocation                            — 0.5 day
Phase 6: Chunk deduplication                                   — 0.5 day
Phase 7: Call graph from Tree-sitter                           — 2 days
```

Phases 1+2 alone should flip the A/B test. Phase 7 creates the moat.

## Key Files

- `src/know/query.py` — NEW: query understanding, stop words, intent classification
- `src/know/daemon_db.py` — `_build_fts_query()`, `search_chunks()`, schema for symbol_refs
- `src/know/context_engine.py` — `_build_context_v3_inner()`, budget allocation, context expansion
- `src/know/ranking.py` — `fuse_rankings()` (exists, wire it in), importance/recency boosts
- `src/know/daemon.py` — `populate_index()` extend to extract symbol refs
- `src/know/parsers.py` — Tree-sitter AST already available, extend to extract calls
