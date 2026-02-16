---
title: "feat: Progressive Disclosure, Session Dedup, Deep Context"
type: feat
date: 2026-02-16
---

# Progressive Disclosure, Session Dedup, Deep Context

## Overview

Three features that together deliver **2-3x token efficiency** for AI coding agents using know-cli. The core insight from benchmarking: know-cli v0.6.0 uses 2.5x fewer tool calls than grep/read, but only 7% fewer tokens — because 40% of the token budget goes to metadata agents ignore. These features fix that.

**The 3-tier model:**

| Tier | Command | Tokens/result | Use case |
|------|---------|---------------|----------|
| **Map** | `know map "query"` | ~30-50 | Orient: what exists? |
| **Context** | `know context "query" --session S` | ~300-500 | Investigate: relevant code bodies (deduped) |
| **Deep** | `know deep "function_name"` | ~1500-2000 | Surgical: function + callers + callees |

Plus: README rewrite and agent skill update to teach the new workflow.

## Problem Statement

### Benchmark Evidence (v0.6.0 on farfield — 762 files)

| Metric | Agent A (know-cli) | Agent B (grep/read) |
|--------|-------------------|---------------------|
| Tool calls | 14 | 36 |
| Total tokens | 105,950 | 113,471 |
| **Token delta** | — | **Only 7% more** |

**Why the token savings are so small:**

1. **Metadata overhead**: 40% of budget allocated to imports/summaries/overview that agents typically ignore
2. **No progressive disclosure**: Agent gets full bodies even when it just wants to know what exists
3. **No session memory**: Repeated queries return the same chunks, wasting budget on already-seen code
4. **No dependency bundling**: Agent must make follow-up calls to read callees/callers

### Market Context

- Aider's repo-map: sends signatures only (~50 tokens/function), agent requests full bodies on demand
- Augment Code ($2.5B valuation): primary pitch is "smart context for agents"
- Claude Code costs ~$6/dev/day. A 2-3x token reduction = $3-4/dev/day savings at scale
- jxnl/SWE-bench research: grep beats embeddings, but smarter agentic tools beat grep

## Proposed Solution

### Phase 1: `know map` — Lightweight Signature Search

A new command that returns **signatures + first-line docstrings** matching a query. No bodies. Agents use this to orient before deciding what to read.

**CLI:**
```bash
know map "billing subscription"              # Rich table output
know --json map "billing" --limit 30         # JSON for agents
know map "auth" --type function              # Filter by chunk type
```

**Output (JSON):**
```json
{
  "query": "billing subscription",
  "results": [
    {
      "file": "src/billing/service.py",
      "name": "check_cloud_access",
      "type": "function",
      "signature": "async def check_cloud_access(self, workspace: Workspace) -> None",
      "docstring": "Verify workspace has active subscription and available sandbox slots.",
      "line": 142,
      "score": 0.89
    }
  ],
  "count": 12,
  "truncated": false
}
```

**Implementation in `src/know/cli/agent.py`:**

```python
@click.command("map")
@click.argument("query")
@click.option("--limit", "-k", default=20, help="Max results")
@click.option("--type", "chunk_type", type=click.Choice(["function", "class", "module"]), help="Filter chunk type")
@click.pass_context
def map_cmd(ctx, query, limit, chunk_type):
    """Lightweight signature search — orient before reading."""
```

**Key decisions:**
- [ ] Reuses existing `search_chunks()` pipeline but projects only signature fields
- [ ] Includes first line of docstring (capped at 120 chars) for decision-making context
- [ ] Default limit=20, max=100
- [ ] Does NOT add to session seen-set (no bodies returned)
- [ ] Uses `click.echo` for JSON (not `console.print`) to avoid Rich ANSI contamination

**Files to modify:**
- `src/know/cli/agent.py` — add `map_cmd` command
- `src/know/cli/__init__.py` — register command
- `src/know/daemon.py` — add `_handle_map` handler
- `src/know/daemon_db.py` — add `search_signatures()` method (thin wrapper over `search_chunks` returning only sig fields)

### Phase 2: Session-Aware Deduplication

Track which chunks an agent has already received. Subsequent queries skip seen chunks and fill budget with new results.

**Architecture decision: Server-side sessions in daemon.db.**

Rationale:
- Works with daemon protocol (no client-side file I/O)
- Works with direct DB fallback (same tables)
- Auto-expires via TTL (no cleanup needed)
- Session ID auto-generated, passed back to agent in response

**CLI:**
```bash
# First call — creates session automatically
know --json context "billing" --budget 4000 --session auto
# Response includes "session_id": "a1b2c3d4"

# Second call — reuses session, skips already-seen chunks
know --json context "subscription limits" --budget 4000 --session a1b2c3d4

# Explicit new session
know --json context "auth" --budget 4000 --session new

# No session (current behavior, backward compatible)
know --json context "auth" --budget 4000
```

**New DB tables in `src/know/daemon_db.py`:**

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_seen (
    session_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL,  -- "file_path:chunk_name:start_line"
    provided_at REAL NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, chunk_key),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
```

**Session lifecycle:**
- `--session auto`: Creates new session on first call, returns `session_id` in response. Agent reuses on subsequent calls.
- `--session <id>`: Reuses existing session. If expired/missing, creates new one silently.
- `--session new`: Forces fresh session even if one existed.
- No `--session`: Current behavior, no dedup (backward compatible).
- **TTL**: 4 hours. Expired sessions cleaned up lazily on next daemon start or when table exceeds 100 sessions.
- Sessions survive within daemon lifetime. On daemon restart, old sessions in DB are expired by TTL check.

**Context engine changes (`src/know/context_engine.py`):**

```python
def build_context(self, query, budget=8000, session_id=None, ...):
    # After search + scoring + relevance floor:
    if session_id:
        seen_keys = self.db.get_session_seen(session_id)
        raw_results = [r for r in raw_results if self._chunk_key(r) not in seen_keys]
        # Continue filling budget with remaining results

    # After building final result:
    if session_id:
        new_keys = [self._chunk_key(c) for c in result["code_chunks"]]
        self.db.mark_session_seen(session_id, new_keys)

    result["session_id"] = session_id  # Include in response
```

**Key decisions:**
- [ ] Budget is re-filled after dedup (lower-ranked new results promoted to fill freed space)
- [ ] Chunk key format: `"file_path:chunk_name:start_line"` (matches existing `seen_chunk_keys` pattern)
- [ ] `know deep` also participates: returned function + callers/callees are marked seen
- [ ] `know map` does NOT mark anything seen (no bodies returned)
- [ ] `--session` is a CLI flag on `context` and `deep` commands

**DaemonDB methods to add:**
- `create_session(session_id) -> str`
- `get_session_seen(session_id) -> Set[str]`
- `mark_session_seen(session_id, chunk_keys: List[str], tokens: List[int])`
- `cleanup_expired_sessions(ttl_seconds=14400)`
- `get_session_stats(session_id) -> Dict` (total chunks seen, total tokens)

**Files to modify:**
- `src/know/daemon_db.py` — session tables + methods
- `src/know/context_engine.py` — dedup logic in `build_context()` / `_build_context_v3_inner()`
- `src/know/daemon.py` — session params in `_handle_context`
- `src/know/cli/search.py` — `--session` flag on `context` command
- `src/know/cli/agent.py` — `--session` flag on `deep` command

### Phase 3: `know deep` — Dependency-Aware Context

Given a function name, return the function body + callees (what it calls) + callers (what calls it), all within a token budget.

**CLI:**
```bash
know deep "check_cloud_access" --budget 3000
know --json deep "BillingService.check_cloud_access" --budget 3000
know --json deep "src/billing/service.py:check_cloud_access" --budget 3000
know deep "check_cloud_access" --session a1b2c3 --budget 3000
```

**Name resolution strategy (handles ambiguity):**

1. Try exact match on `chunk_name` in `chunks` table
2. If multiple matches, try `file_path:chunk_name` format
3. If still ambiguous, try `Class.method` format (split on `.`)
4. If still ambiguous: return error with candidates list
5. Filter: exclude test files by default (add `--include-tests` to override)

```python
def resolve_function(self, name: str) -> List[Dict]:
    """Resolve function name to chunk(s). Returns candidates."""
    # 1. Try exact chunk_name match
    candidates = self.db.get_chunks_by_name(name)
    if len(candidates) == 1:
        return candidates

    # 2. Try file:name format
    if ":" in name:
        file_part, name_part = name.rsplit(":", 1)
        candidates = [c for c in self.db.get_chunks_by_name(name_part)
                      if file_part in c["file_path"]]
        if candidates:
            return candidates[:1]

    # 3. Try Class.method format
    if "." in name:
        parts = name.split(".")
        candidates = [c for c in self.db.get_chunks_by_name(parts[-1])
                      if any(p in c.get("file_path", "") or p in c.get("chunk_name", "")
                             for p in parts[:-1])]
        if candidates:
            return candidates[:1]

    # 4. Filter test files by default
    source_candidates = [c for c in candidates
                         if categorize_file(c["file_path"]) == "source"]
    return source_candidates or candidates
```

**Budget allocation:**

```
Total budget: B tokens
├── Target function body: min(body_tokens, B * 0.50)  — always included
├── Callees budget: (B - body_tokens) * 0.50           — what it calls
└── Callers budget: (B - body_tokens) * 0.50           — what calls it
```

Within callee/caller budgets, prioritize by:
1. Same-file functions first (locality)
2. Smallest token count first (maximize coverage)
3. Overflow items listed as signature-only (falls back to map-style)

**Output (JSON):**
```json
{
  "target": {
    "file": "src/billing/service.py",
    "name": "check_cloud_access",
    "signature": "async def check_cloud_access(self, workspace) -> None",
    "body": "...",
    "line_start": 142,
    "line_end": 178,
    "tokens": 487
  },
  "callees": [
    {
      "file": "src/billing/repository.py",
      "name": "count_active_cloud_sessions",
      "body": "...",
      "tokens": 312,
      "call_site_line": 156
    }
  ],
  "callers": [
    {
      "file": "src/cloud_sessions/service.py",
      "name": "create_cloud_session",
      "body": "...",
      "tokens": 445,
      "call_site_line": 89
    }
  ],
  "overflow_signatures": [
    "send_notification(workspace, event) — src/billing/service.py:201"
  ],
  "call_graph_available": true,
  "budget_used": 2744,
  "budget": 3000,
  "session_id": "a1b2c3d4"
}
```

**Edge cases:**
- [ ] Function not found → exit code 2, JSON `{"error": "not_found", "nearest": [...]}`
- [ ] Ambiguous name → exit code 1, JSON `{"error": "ambiguous", "candidates": [...]}`
- [ ] No call graph (regex-parsed files) → return body only, `"call_graph_available": false`, warning in stderr
- [ ] 50+ callees → budget overflow: include as many full bodies as fit, rest as signatures in `overflow_signatures`
- [ ] Self-referential/mutual recursion → deduplicate by chunk_key, never include target in its own callers/callees
- [ ] External calls (stdlib/third-party) → listed in `overflow_signatures` as "external: json.dumps" (no body)
- [ ] Body exceeds 50% budget → truncate body at budget * 0.50, add "# ... truncated" marker

**Files to modify:**
- `src/know/context_engine.py` — add `build_deep_context()` method
- `src/know/daemon_db.py` — add `get_chunks_by_name()` method
- `src/know/daemon.py` — add `_handle_deep` handler
- `src/know/cli/agent.py` — add `deep` command
- `src/know/cli/__init__.py` — register command

### Phase 4: README Rewrite

Rewrite README.md to position know-cli as a **3-tier context engine for AI agents** with clear benchmark evidence.

**New structure:**

```
# know — 3x Fewer Tokens for AI Coding Agents

## The Problem (3 lines)
## The Solution — Map, Context, Deep (diagram)
## Benchmarks
  - Farfield benchmark table (762 files, 14 vs 36 tool calls)
  - Token efficiency per-query table
## Quick Start (5 lines of bash)
## The 3-Tier Workflow
  ### Tier 1: Map — Orient (know map)
  ### Tier 2: Context — Investigate (know context --session)
  ### Tier 3: Deep — Surgical (know deep)
## Works With (Claude Code, Cursor, Aider, any agent)
## All Commands (reference table)
## Installation
## How It Works (architecture)
## Configuration
```

**Key messaging changes:**
- Lead with the 3-tier model, not feature lists
- Benchmark table is above the fold
- "8-18x fewer tokens" → update to actual measured numbers from farfield benchmark
- Add session dedup to the workflow
- Remove "agent-native commands" framing → replace with "3-tier context engine"

**File:** `README.md`

### Phase 5: Agent Skill Update

Rewrite `~/.claude/skills/know-cli/SKILL.md` to teach the Map → Context → Deep workflow.

**New structure:**

```markdown
---
name: know-cli
description: 3-tier context engine for AI agents. Use when exploring codebases,
  finding relevant code, or understanding function dependencies. Provides map
  (signatures), context (ranked bodies), and deep (function + callers/callees)
  with session dedup to avoid re-reading code.
---

# know-cli — 3-Tier Context for AI Agents

## Quick Start
know map "query"                           # What exists? (~50 tokens/result)
know --json context "query" --budget 4000  # Relevant code bodies
know --json deep "function_name"           # Function + dependencies

## The Workflow
### 1. Orient (always start here)
know map "billing"
### 2. Investigate (with session tracking)
know --json context "billing" --budget 6000 --session auto
### 3. Go Deep (when you need one function fully)
know --json deep "check_cloud_access" --session <id> --budget 3000
### 4. Remember insights
know remember "key finding" --tags "billing"

## When to Use What
- know map: You don't know what functions exist for a topic
- know context: You need actual code for 3-8 relevant functions
- know deep: You need one specific function + what calls it / what it calls
- Grep: You need exact string matching (know doesn't replace grep)

## Guidelines
- Use --json for structured output (global flag, before subcommand)
- Use --session auto to enable cross-query dedup (saves ~40% tokens on follow-ups)
- Budget 4000-8000 tokens for context, 2000-4000 for deep
- First daemon call may return partial results — retry after 2-3s if empty
```

**File:** `~/.claude/skills/know-cli/SKILL.md`

## Implementation Phases

### Phase 1: `know map` command
- [ ] Add `search_signatures()` to `daemon_db.py` — search_chunks but returns only sig fields + first-line docstring
- [ ] Add `_handle_map` to `daemon.py` handler dispatch
- [ ] Add `map_cmd` to `cli/agent.py` with `--limit`, `--type` options
- [ ] Register in `cli/__init__.py`
- [ ] Add tests: zero results, limit, type filter, JSON output, Rich output
- [ ] **Verify**: `know --json map "billing" --limit 10` returns signatures only, ~50 tokens/result

### Phase 2: Session-aware dedup
- [ ] Add `sessions` and `session_seen` tables to `daemon_db.py` schema
- [ ] Bump `SCHEMA_VERSION` to 5 (force reindex not needed — additive tables only)
- [ ] Add session CRUD methods to `DaemonDB`
- [ ] Add `--session` flag to `context` command in `cli/search.py`
- [ ] Modify `build_context()` / `_build_context_v3_inner()` in `context_engine.py` to filter seen chunks and re-fill budget
- [ ] Add `session_id` to JSON response
- [ ] Add session param to `_handle_context` in `daemon.py`
- [ ] Add lazy cleanup in daemon startup: `cleanup_expired_sessions()`
- [ ] Add tests: first call creates session, second call deduplicates, expired session creates new, budget re-fills after dedup, no session = backward compatible
- [ ] **Verify**: Two sequential `know context` calls with same session return zero overlap

### Phase 3: `know deep` command
- [ ] Add `get_chunks_by_name()` to `daemon_db.py`
- [ ] Add `resolve_function()` to `context_engine.py` — name resolution with disambiguation
- [ ] Add `build_deep_context()` to `context_engine.py` — budget allocation + call graph assembly
- [ ] Add `_handle_deep` to `daemon.py`
- [ ] Add `deep` command to `cli/agent.py` with `--budget`, `--session`, `--include-tests` options
- [ ] Register in `cli/__init__.py`
- [ ] Handle edge cases: not found, ambiguous, no call graph, budget overflow, self-referential
- [ ] Add `--session` support (mark returned chunks as seen)
- [ ] Add tests: single match, ambiguous name, file:name resolution, callees included, callers included, budget overflow → signatures, no symbol_refs → body only with warning, session integration
- [ ] **Verify**: `know --json deep "check_cloud_access" --budget 3000` returns target + callees + callers

### Phase 4: README rewrite
- [ ] Rewrite `README.md` with new 3-tier structure
- [ ] Update benchmark table with farfield results (762 files, 14 vs 36 tool calls)
- [ ] Add `know map` and `know deep` examples
- [ ] Add session workflow example
- [ ] Update command reference table
- [ ] Keep installation section, update feature list

### Phase 5: Agent skill update
- [ ] Rewrite `~/.claude/skills/know-cli/SKILL.md` with Map → Context → Deep workflow
- [ ] Update description in frontmatter for better skill discovery
- [ ] Add "When to Use What" decision guide
- [ ] Add session usage guidelines
- [ ] Keep backward-compatible advice (remember/recall, signatures, related still work)

### Phase 6: Bump version and publish
- [ ] Bump version to 0.7.0 in `pyproject.toml`
- [ ] Run full test suite: `pytest tests/ -q --ignore=tests/test_week4.py`
- [ ] Build and publish to PyPI
- [ ] Install via pipx: `pipx install know-cli==0.7.0 --force`
- [ ] Test on farfield project: `know map "billing"`, `know context "billing" --session auto`, `know deep "check_cloud_access"`

## Acceptance Criteria

### Functional Requirements

- [ ] `know map "query"` returns signatures + first-line docstrings, no bodies
- [ ] `know map` JSON output averages <60 tokens per result
- [ ] `know context --session auto` returns a session_id and deduplicates on follow-up calls
- [ ] Second `context` call with same session returns zero overlapping chunks
- [ ] Session dedup re-fills budget with lower-ranked new results (not short-changed budget)
- [ ] `know deep "fn"` returns target body + callers + callees within budget
- [ ] `know deep` handles ambiguous names with candidate list error
- [ ] `know deep` with no call graph returns body only + `call_graph_available: false`
- [ ] All new commands support `--json` flag correctly (no Rich ANSI contamination)
- [ ] All new commands work via daemon AND direct DB fallback
- [ ] Backward compatibility: existing commands unchanged when `--session` is not used

### Quality Gates

- [ ] All existing tests pass (169 core + 27 v2)
- [ ] 20+ new tests covering map, session dedup, deep, edge cases
- [ ] README includes real benchmark numbers
- [ ] Skill file teaches 3-tier workflow with clear decision criteria

## Success Metrics

After implementation, re-run the farfield benchmark (same 3 questions):

| Metric | v0.6.0 (current) | v0.7.0 (target) |
|--------|-------------------|------------------|
| Tool calls | 14 | **8-10** |
| Total tokens | 106K | **50-70K** |
| Quality | Equivalent | Equivalent or better |

The target is **40-50% fewer tokens** with the 3-tier workflow:
1. `know map` for orientation (~500 tokens total)
2. `know context --session` for investigation (~4000 tokens, zero overlap on follow-ups)
3. `know deep` for surgical reads (~2000 tokens, callees included)

## Technical Considerations

### Schema Migration
- Phase 2 adds `sessions` and `session_seen` tables — additive only, no reindex needed
- Bump SCHEMA_VERSION to 5 with migration that creates tables without touching existing data

### Backward Compatibility
- All existing commands unchanged
- `--session` is opt-in (default: no session)
- `know map` is a new command, no conflicts
- `know deep` is a new command, no conflicts

### Performance
- `search_signatures()` should be faster than `search_chunks()` (no body field in projection)
- Session lookup is O(1) via PRIMARY KEY on `(session_id, chunk_key)`
- Session cleanup is lazy (on daemon start), not on every request
- `know deep` does 3 DB queries: chunk lookup + get_callees + get_callers. All indexed.

### Known Limitations
- Call graph requires tree-sitter for non-Python languages. Regex-parsed files have empty symbol_refs.
- Session dedup is within daemon lifetime. If daemon restarts, sessions in DB may be stale (TTL handles this).
- `know deep` for heavily-called utility functions (e.g., `log()`) may return too many callers — limit + budget handles this.

## References

### Internal
- `src/know/context_engine.py:451` — `_build_context_v3_inner()` main pipeline
- `src/know/daemon_db.py:405` — `search_chunks()` with RRF fusion
- `src/know/daemon.py:308` — handler dispatch table
- `src/know/cli/agent.py:162` — `callers` command (pattern for `deep`)
- `src/know/ranking.py:27` — `apply_relevance_floor()`
- `src/know/query.py:89` — `analyze_query()` pipeline

### External Research
- [Aider repo-map](https://aider.chat/2023/10/22/repomap.html) — signatures + PageRank for orientation
- [Augment Context Engine MCP](https://www.augmentcode.com/blog/context-engine-mcp-now-live) — 70% agent performance improvement
- [Why grep beats embeddings (jxnl)](https://jxnl.co/writing/2025/09/11/why-grep-beat-embeddings-in-our-swe-bench-agent-lessons-from-augment/) — agentic tools > RAG
- [cAST: AST-aware chunking (CMU)](https://arxiv.org/html/2506.15655v1) — structure-preserving chunks
- [Progressive disclosure for AI agents](https://www.honra.ai/articles/progressive-disclosure-for-ai-agents) — just-in-time context
- Farfield benchmark (this project): Agent A 14 calls/106K tokens vs Agent B 36 calls/113K tokens
