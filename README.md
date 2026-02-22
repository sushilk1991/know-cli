# know — 3x Fewer Tokens for AI Coding Agents

> Your AI agent wastes tokens reading code it doesn't need. **know** gives it exactly what it needs, in 3 tiers.

[![PyPI](https://img.shields.io/pypi/v/know-cli)](https://pypi.org/project/know-cli/)
[![Python](https://img.shields.io/pypi/pyversions/know-cli)](https://pypi.org/project/know-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Problem

AI agents dump entire files into context. Every `grep` match pulls in thousands of tokens of irrelevant code. Repeated queries re-read the same functions.

**Result:** Slow. Expensive. Your agent burns through token budget reading imports and boilerplate it doesn't need.

## The Solution — Map, Context, Deep

**know** is a 3-tier context engine. Agents start broad and zoom in, paying only for what they need.

| Tier | Command | Tokens/result | Use case |
|------|---------|---------------|----------|
| **Map** | `know map "query"` | ~10-25 per signature | Orient: what exists? |
| **Context** | `know context "query" --session S` | ~150-350 per chunk | Investigate: relevant code bodies |
| **Deep** | `know deep "function_name"` | ~300-1500 per call | Surgical: function + callers + callees |

Session dedup means the second query never re-sends code from the first.

---

## Benchmarks

### Dual-Repo Parallel Benchmark (v0.8.0, February 22, 2026)

Method:
- Ran in parallel per query: `grep+read` baseline vs `know workflow` (single daemon RPC).
- 4 shared architecture questions, on both repos.
- File coverage in baseline search: `py, ts, tsx, js, jsx, go, rs, swift`.
- Includes one warm-up workflow call per repo before measured queries (steady-state).

| Repo | Grep Tokens (4 queries) | know Tokens (4 queries) | Token Reduction | Grep Time | know Time | Latency (know/grep) | Tool Calls (grep -> know) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `know-cli` | 235,329 | 15,295 | **93.5%** | 0.265s | 0.479s | 1.81x | 64 -> 4 |
| `farfield` | 284,293 | 13,995 | **95.1%** | 0.458s | 0.788s | 1.72x | 64 -> 4 |

Deep call-graph quality snapshot from the same run:
- `know-cli`: call-graph available in 100% of deep queries, non-empty edges in 100%.
- `farfield`: call-graph available in 100%, non-empty edges in 25% (major remaining gap).

Artifacts:
- `benchmark/results/dual_repo_parallel.json`
- `benchmark/results/DUAL_REPO_BENCHMARK.md`
- Runner: `benchmark/bench_dual_repo_parallel.py`

### What this means

- **Token and tool-call efficiency is strong now.**
- **Single-daemon workflow significantly cuts orchestration overhead** (tool calls dropped to 1/query).
- **Latency is still behind raw grep**, but the gap is much smaller than before.
- The biggest product gap remains **deep call-graph completeness on large TSX/Python codebases**.

---

## Quick Start

```bash
pip install know-cli
cd your-project
know init
```

### What's New in 0.8.0

- New `know workflow "query"` command: single-call daemon workflow (`map -> context -> deep`).
- `know context` now uses daemon-first full v3 context assembly (same ranking/budget pipeline as local fallback).
- `know deep` now performs opportunistic stale-file refresh for likely candidate files before resolving symbols.
- `know related` refreshes the target file on demand, reducing stale import/dependency outputs.
- Incremental refresh path re-indexes chunks + symbol refs for changed files without full reindex.
- Fixed search lane weighting bug when OR lane is empty (AND/exact weights now remain stable).

### Single-Call Workflow (Recommended for agents)

```bash
know --json workflow "billing subscription limits" \
  --map-limit 20 \
  --context-budget 4000 \
  --deep-budget 3000 \
  --session auto
```

### The 3-Tier Workflow

```bash
# 1. Orient — what functions exist for "billing"?
know map "billing"

# 2. Investigate — get ranked code bodies (with session tracking)
know --json context "billing subscription" --budget 4000 --session auto
# Returns session_id: "a1b2c3d4"

# 3. Go deep — one function + its callers/callees
know --json deep "check_cloud_access" --budget 3000

# Follow-up queries skip already-seen code
know --json context "payment processing" --budget 4000 --session a1b2c3d4
```

---

## Commands

### Workflow — Single Daemon Call

```bash
know workflow "billing subscription limits"
know --json workflow "auth token validation" --context-budget 5000 --deep-budget 2500
```

Runs `map -> context -> deep` in a single daemon request to cut tool-call overhead for coding agents.

### Docs Update Policy (Local-First)

- Recommended default: run docs updates locally via `know watch` or git hooks (`know hooks install`).
- GitHub workflow examples in this repo are `workflow_dispatch` and **non-mutating by default**.
- If a downstream repo enables CI docs updates, keep CI read-only (review/comment artifacts) unless explicit auto-commit is required.

### Map — Orient Before Reading

```bash
know map "billing subscription"              # What exists?
know --json map "auth" --limit 30            # JSON for agents
know map "config" --type function            # Filter by type
```

Returns signatures + first-line docstrings. No bodies. Typically ~10-25 tokens per result.

### Context — Ranked Code Bodies

```bash
know context "fix the auth bug" --budget 8000
know --json context "query" --budget 4000 --session auto
echo "refactor config" | know context --budget 6000
```

Finds relevant functions across the codebase. Token-budgeted. Optionally deduplicates across queries with `--session`.

### Deep — Function + Dependencies

```bash
know deep "check_cloud_access" --budget 3000
know --json deep "BillingService.process_payment"
know --json deep "service.py:check_cloud_access"
```

Returns the function body + what it calls (callees) + what calls it (callers), all within budget. Handles ambiguous names, budget overflow, and missing call graphs.

### Memory — Cross-Session Knowledge

```bash
know remember "Auth uses JWT with Redis session store"
know recall "how does auth work?"
```

Memories are automatically included in `know context` results.

### All Commands

| Command | Description |
|---------|-------------|
| `know workflow "query"` | Single-call daemon workflow (map + context + deep) |
| `know map "query"` | Lightweight signature search |
| `know context "query"` | Smart, budgeted code context |
| `know deep "name"` | Function + callers + callees |
| `know search "query"` | Semantic code search |
| `know remember "text"` | Store a memory |
| `know recall "query"` | Recall memories |
| `know signatures [file]` | Function/class signatures |
| `know related <file>` | Import deps and dependents |
| `know callers <function>` | What calls this function |
| `know callees <chunk>` | What this function calls |
| `know next-file "query"` | Best file for a query |
| `know graph <file>` | Import graph visualization |
| `know status` | Project health check |
| `know stats` | Usage statistics |
| `know diff --since "1w"` | Architectural changes over time |
| `know mcp serve` | Start MCP server |
| `know init` | Initialize know in project |

### Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--quiet` | Minimal output |
| `--verbose` | Detailed output |
| `--time` | Show execution time |

---

## Works With

| Tool | Integration |
|------|-------------|
| **Claude Code** | Agent skill — Claude uses know automatically |
| **Claude Desktop** | MCP server: `know mcp serve` |
| **Cursor** | MCP server or CLI |
| **Any CLI agent** | Pipe-friendly: `know --json context "query"` |

---

## How It Works

```
Your Query → know context
  ├─ FTS5 Search (BM25F field weighting)
  │    └─ Finds relevant functions/classes
  ├─ Ranking Pipeline
  │    ├─ File category demotion (test/vendor/generated)
  │    ├─ Import graph importance boost
  │    ├─ Git recency boost
  │    └─ File-path match boost
  ├─ Context Expansion
  │    └─ Module imports, parent classes, adjacent chunks
  ├─ Session Dedup (optional)
  │    └─ Skips chunks already returned in this session
  ├─ Knowledge Base
  │    └─ Injects cross-session memories
  └─ Token Budget Allocator
       └─ 60% code | 15% imports | 15% summaries | 10% overview
```

All processing is **local**. No data leaves your machine.

---

## Installation

```bash
# Core (CLI + context engine + memory)
pip install know-cli

# With semantic search
pip install know-cli[search]

# With MCP server
pip install know-cli[mcp]

# Everything
pip install know-cli[search,mcp]
```

**Requirements:** Python 3.10+

---

## Configuration

```bash
know init  # Creates .know/config.yaml
```

```yaml
project:
  name: my-project
  description: "A web application"
languages:
  - python
include_paths:
  - src/
exclude_paths:
  - tests/fixtures/
```

## Architecture

```
.know/
  config.yaml     # Project configuration
  daemon.db       # SQLite database (chunks, memories, imports, sessions)
```

Single SQLite database with FTS5 and WAL mode. Background daemon for sub-100ms latency. Falls back to direct DB access when daemon is unavailable.

## Contributing

```bash
git clone https://github.com/sushilk1991/know-cli
cd know-cli
pip install -e ".[dev,search,mcp]"
pytest tests/ -v
```

## License

MIT
