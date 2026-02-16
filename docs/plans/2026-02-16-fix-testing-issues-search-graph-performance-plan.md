---
title: "fix: Search fallback, empty import graph, and slow daemon startup"
type: fix
date: 2026-02-16
---

# fix: Search fallback, empty import graph, and slow daemon startup

## Overview

Four issues discovered when testing know-cli v0.4.1 in a real project plus two cross-cutting concerns identified by 4-reviewer analysis.

## Problem Statement

1. **`know search` crashes without numpy/fastembed** — crashes with `No module named 'numpy'` instead of falling back to BM25.
2. **`know next-file` returns "No more relevant files found"** — BM25 exact-phrase quoting (`daemon_db.py:212`) can't match semantic queries.
3. **`know related src/know/daemon.py` returns empty** — import graph only stores modules with outgoing edges (`import_graph.py:137-138`).
4. **~21 second first-run latency** — daemon runs `_full_index()` before accepting connections (`daemon.py:118`).
5. **`recall_memories` has same exact-phrase bug** as `search_chunks` (`daemon_db.py:419`).
6. **Thread-unsafe shared SQLite connection** — background indexing thread + event loop queries share one `sqlite3.Connection`.

## Proposed Solution

### Fix 1: BM25 fallback for `know search`

**File:** `src/know/cli/search.py`

- [x] Wrap `from know.semantic_search import SemanticSearcher` in `try/except ImportError`
- [x] Fallback: use daemon-first pattern (`_get_daemon_client` → direct `DaemonDB`)
- [x] Mirror all three output branches: `--json`, `--quiet`, and rich
- [x] Always print hint: `[dim]Tip: pip install know-cli[search] for semantic search[/dim]`
- [x] Track stats in fallback path via `StatsTracker.record_search()`

### Fix 2: OR-based BM25 query parsing + shared helper

**File:** `src/know/daemon_db.py`

- [x] Add `_build_fts_query()` static method
- [x] Cap at 12 terms to bound FTS5 OR query complexity
- [x] Update `search_chunks()` to use `_build_fts_query()`
- [x] Update `recall_memories()` to use `_build_fts_query()` (same exact-phrase bug)

### Fix 3: Store all modules in import graph

**File:** `src/know/import_graph.py:137-138`

- [x] Remove the `if mod_edges:` guard — store even empty edge lists

### Fix 4: Non-blocking daemon startup

**File:** `src/know/daemon.py`

- [x] Move `start_unix_server()` before `_full_index()`
- [x] Change `await self._full_index()` to `self._index_task = asyncio.create_task(self._full_index())`
- [x] Store task reference to prevent GC and enable status checks
- [x] No `_indexing` flag or `_background_index` wrapper needed (YAGNI)

### Fix 5: Thread-safe SQLite connections

**File:** `src/know/daemon_db.py`

- [x] Replace shared `self._conn` with `threading.local()` per-thread connections
- [x] Add `PRAGMA busy_timeout=5000` to prevent `SQLITE_BUSY` errors
- [x] Each thread gets its own connection; WAL mode handles concurrent reads + single writer
- [x] Update `close()` to close thread-local connections

## Acceptance Criteria

- [x] `know search "daemon socket"` works without fastembed (uses BM25)
- [x] `know search "daemon socket"` prints fastembed hint
- [x] `know search "daemon socket" --json` returns valid JSON in fallback
- [x] `know next-file "message framing protocol"` returns `src/know/daemon.py` (verified via direct DB)
- [x] `know next-file "knowledge base"` returns relevant file (OR-based BM25 matches partial terms)
- [x] `know related src/know/daemon.py` shows imports and imported-by
- [x] `know related src/know/daemon_db.py` shows both directions
- [x] `know recall "project architecture"` matches even without exact phrase
- [x] Daemon accepts first query within 2 seconds of startup
- [x] Full index completes in background
- [x] All 169 existing tests pass (excluding pre-existing test_week4 failures)
- [x] All 27 v2 tests pass

## Execution Order

1. Fix 5 (thread-safe connections) — foundational for Fix 4
2. Fix 2 (BM25 query helper) — foundational for Fix 1
3. Fix 1 (search fallback) — depends on Fix 2
4. Fix 3 (import graph) — independent
5. Fix 4 (non-blocking startup) — depends on Fix 5

## References

- Previous plan: `docs/plans/2026-02-15-refactor-know-cli-v2-daemon-architecture-plan.md`
- Solution doc: `docs/solutions/architecture/p2-p3-architectural-improvements.md`
- Key files:
  - `src/know/cli/search.py:34` — search command
  - `src/know/daemon_db.py:208-225` — `search_chunks()` BM25 method
  - `src/know/daemon_db.py:415-432` — `recall_memories()` BM25 method
  - `src/know/import_graph.py:137-138` — edge storage logic
  - `src/know/daemon.py:106-135` — daemon serve/startup
  - `src/know/cli/agent.py:32-71` — next-file command
