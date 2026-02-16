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
| **Map** | `know map "query"` | ~50 | Orient: what exists? |
| **Context** | `know context "query" --session S` | ~300-500 | Investigate: relevant code bodies |
| **Deep** | `know deep "function_name"` | ~1500 | Surgical: function + callers + callees |

Session dedup means the second query never re-sends code from the first.

---

## Benchmarks

### Token efficiency: know context vs Grep+Read

**farfield** (762 files, production TypeScript+Python):

| Scenario | Grep+Read | know context | Reduction |
|---|---|---|---|
| WebSocket handling | 12,936 tokens | 1,714 tokens | **7.5x** |
| Auth and API keys | 3,383 tokens | 1,623 tokens | **2.1x** |
| Model routing | 27,556 tokens | 1,772 tokens | **15.6x** |
| Error handling | 25,160 tokens | 1,773 tokens | **14.2x** |
| Database + storage | 5,357 tokens | 1,773 tokens | **3.0x** |
| **Total** | **74,392** | **8,655** | **8.6x** |

### Head-to-head agent benchmark (farfield, 762 files)

| Metric | Agent with `know` | Agent with Grep+Read |
|---|---|---|
| Tool calls | 14 | 36 |
| Total tokens | 105,950 | 113,471 |
| Quality | Equivalent | Equivalent |

---

## Quick Start

```bash
pip install know-cli
cd your-project
know init
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

### Map — Orient Before Reading

```bash
know map "billing subscription"              # What exists?
know --json map "auth" --limit 30            # JSON for agents
know map "config" --type function            # Filter by type
```

Returns signatures + first-line docstrings. No bodies. ~50 tokens per result.

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
