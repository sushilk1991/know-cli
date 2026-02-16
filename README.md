# know — Context Intelligence for AI Coding Agents

> Your AI agent wastes tokens. **know** gives it exactly what it needs.

[![PyPI](https://img.shields.io/pypi/v/know-cli)](https://pypi.org/project/know-cli/)
[![Python](https://img.shields.io/pypi/pyversions/know-cli)](https://pypi.org/project/know-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The Problem

AI coding agents dump entire files into context. Every `@file` reference, every `cat` command — full files, whether you need 3 lines or 3000.

**Result:** Slow. Expensive. Often irrelevant. Your agent burns through your token budget reading imports it doesn't need.

## The Solution

**know** understands your codebase and serves the *minimum* context needed.

- 🎯 Semantic search finds the *relevant* functions, not entire files
- 📊 Import graph knows what depends on what
- 🧠 Cross-session memory means agents never re-discover the same things
- 💰 Token budgeting keeps costs predictable
- ⚡ Background daemon for sub-100ms query latency
- 🤖 Agent-native commands (`next-file`, `signatures`, `related`) for autonomous workflows

**10-18x fewer tokens. Same (or better) results.**

---

## Benchmarks

We compared `know context` against the traditional agent workflow (Grep to find files → Read full files) on two real projects.

### know-cli (35 files, 369 functions)

| Scenario | Grep+Read | know context | Reduction |
|---|---|---|---|
| Daemon indexing logic | 18,791 tokens | 1,064 tokens | **17.7x** |
| FTS5 search implementation | 21,937 tokens | 1,046 tokens | **21.0x** |
| Python parser functions | 11,342 tokens | 975 tokens | **11.6x** |
| Context engine budget | 24,075 tokens | 1,191 tokens | **20.2x** |
| Import graph logic | 20,916 tokens | 1,149 tokens | **18.2x** |
| **Total** | **97,061 tokens** | **5,425 tokens** | **17.9x** |

### farfield (762 files, 2,457 functions — production app)

| Scenario | Grep+Read | know context | Reduction |
|---|---|---|---|
| WebSocket connection handling | 12,936 tokens | 1,475 tokens | **8.8x** |
| Authentication and API keys | 3,383 tokens | 1,456 tokens | **2.3x** |
| Model routing and inference | 27,556 tokens | 1,472 tokens | **18.7x** |
| Error handling and retries | 25,160 tokens | 1,474 tokens | **17.1x** |
| Database and storage | 5,357 tokens | 1,451 tokens | **3.7x** |
| **Total** | **74,392 tokens** | **7,328 tokens** | **10.2x** |

### Summary

| Metric | Grep+Read | know context |
|---|---|---|
| Avg token reduction | — | **10-18x fewer** |
| Tool calls per query | 7-9 (grep + read) | **1** |
| Signal-to-noise ratio | ~5-10% signal | **~100% signal** |
| Returns actual source code | Full files (mostly irrelevant) | Ranked functions/classes |
| Warm query latency | N/A (multi-step) | **<100ms** |

---

## Quick Start (30 seconds)

```bash
pip install know-cli
cd your-project
know init
know context "help me fix the auth bug" --budget 4000
```

That's it. You just got the most relevant code for your task, packed into exactly 4000 tokens.

The background daemon starts automatically on first use — subsequent queries return in under 100ms.

---

## Works With

| Tool | Integration |
|------|-------------|
| **Claude Code** | Drop in `KNOW_SKILL.md` — Claude uses know automatically |
| **Claude Desktop** | MCP server: `know mcp serve` |
| **Cursor** | MCP server or CLI |
| **Any CLI agent** | Pipe-friendly: `know context "query" --json` |
| **Any MCP client** | Standard MCP protocol |

---

## Features

### 🎯 Smart Context — `know context`

The killer feature. Ask for what you need, get exactly that — within budget.

```bash
know context "fix the authentication middleware" --budget 8000
know context "add pagination to the users API" --budget 4000 --json
echo "refactor the config system" | know context --budget 6000
```

**How it works:**
1. Semantic search finds relevant functions/classes (not whole files)
2. Import graph pulls in dependencies (signatures only)
3. Test matcher finds related tests
4. Git recency boosts recently-changed code
5. Token budgeting packs it all optimally

### 🧠 Cross-Session Memory — `know remember` / `know recall`

Agents forget everything between sessions. know doesn't.

```bash
know remember "The auth system uses JWT with Redis session store"
know remember "Never modify the migration files directly" --tags "warning,db"
know recall "how does auth work?"
```

Memories are automatically included in `know context` results. Your agent gets smarter over time.

### 🔍 Semantic Search — `know search`

Real embeddings, not grep. Understands meaning, not just keywords.

```bash
know search "error handling"
know search "database connection pooling" --top-k 10 --json
know search "authentication" --chunk  # Search at function level
```

Uses [fastembed](https://github.com/qdrant/fastembed) (BAAI/bge-small-en-v1.5) — runs locally, no API calls.

### 📊 Import Graph — `know graph`

Real dependency resolution, not guessing.

```bash
know graph src/auth/middleware.py
```

```
📊 Import Graph: src/auth/middleware.py

## Imports (dependencies)
  → src.auth.tokens
  → src.db.session

## Imported by (dependents)
  ← src.api.routes
  ← src.api.admin
```

### 🔌 MCP Server — `know mcp serve`

Standard [Model Context Protocol](https://modelcontextprotocol.io/) server. Works with Claude Desktop, Cursor, and any MCP client.

```bash
know mcp serve                    # stdio transport (Claude Desktop)
know mcp serve --sse --port 3000  # SSE transport (web clients)
know mcp config                   # Print Claude Desktop config
```

**MCP Tools:** `get_context`, `search_code`, `remember`, `recall`, `explain_component`, `show_graph`

**MCP Resources:** `codebase://digest`, `codebase://structure`, `codebase://memories`

### 🤖 Agent Commands

Purpose-built for autonomous AI agent workflows.

```bash
know next-file "authentication" --exclude src/auth/old.py
know signatures src/auth/middleware.py
know related src/auth/middleware.py
know generate-context --budget 8000
know diff --since "3 days ago"
```

| Command | Description |
|---------|-------------|
| `know next-file "query"` | Return the single most relevant file for a query |
| `know signatures [file]` | Get function/class signatures for a file or project |
| `know related <file>` | Show import dependencies and dependents |
| `know generate-context` | Generate `.know/CONTEXT.md` for agents to read on session start |
| `know diff --since "1 week ago"` | Show architectural changes over time |

### ⚡ Background Daemon

A Unix socket daemon keeps indexes in memory for instant responses.

```bash
know status   # Shows daemon status, index age, cache size
```

- Starts automatically on first CLI call
- Serves search, signatures, related, and memory queries over IPC
- Falls back to direct SQLite access if daemon is unavailable
- Disable with `KNOW_NO_DAEMON=1` for CI/CD environments

### 📈 Usage Stats — `know stats`

Track your ROI. See how much context you're serving and how efficiently.

```bash
know stats
```

```
📊 know-cli Statistics
─────────────────────
  Project: my-app (42 files, 380 functions)

  Knowledge Base:
    12 memories (5 manual, 7 auto)

  Context Engine:
    Queries served: 47
    Avg budget utilization: 82%
    Avg response time: 340ms

  Search:
    Queries: 23
    Avg response time: 85ms
```

---

## For AI Agents

### Claude Code Skill

Drop `KNOW_SKILL.md` into your project root (see below) and Claude Code will automatically use know-cli:

```markdown
# know-cli Integration

Before starting a task:
  Run: know context "<task description>" --budget 8000 --quiet

When you learn something about the codebase:
  Run: know remember "<insight>"

To search for specific code:
  Run: know search "<query>" --json

To understand dependencies:
  Run: know graph <file_path>
```

### MCP Setup (Claude Desktop)

```bash
know mcp config
```

Outputs the JSON config to add to your Claude Desktop settings:

```json
{
  "mcpServers": {
    "know-cli": {
      "command": "know",
      "args": ["mcp", "serve"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

### Pipe-Friendly Output

Every command supports `--json` and `--quiet` for machine consumption:

```bash
know context "query" --json | jq '.code[0].body'
know search "auth" --json
know recall "patterns" --json
know status --json
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `know init` | Initialize know in your project |
| `know context "query"` | Build smart, budgeted context |
| `know search "query"` | Semantic code search |
| `know remember "text"` | Store a memory |
| `know recall "query"` | Recall memories |
| `know forget <id>` | Delete a memory |
| `know memories list` | List all memories |
| `know graph <file>` | Show import dependencies |
| `know explain -c <name>` | AI-explain a component |
| `know stats` | Usage statistics |
| `know status` | Project health check |
| `know reindex` | Rebuild search index |
| `know mcp serve` | Start MCP server |
| `know mcp config` | Print MCP client config |
| `know digest` | Generate codebase summary |
| `know watch` | Auto-update on file changes |
| `know next-file "query"` | Best file for a query (agent use) |
| `know signatures [file]` | Function/class signatures |
| `know related <file>` | Import deps and dependents |
| `know generate-context` | Generate `.know/CONTEXT.md` |
| `know diff --since "1w"` | Architectural changes over time |
| `know hooks install` | Install git hooks for auto-update |
| `know hooks uninstall` | Remove git hooks |

### Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--quiet` | Minimal output |
| `--verbose` | Detailed output |
| `--time` | Show execution time |
| `--config <path>` | Custom config file |

---

## Installation

```bash
# Core (CLI + context engine + memory)
pip install know-cli

# With semantic search (recommended)
pip install know-cli[search]

# With MCP server
pip install know-cli[mcp]

# Everything
pip install know-cli[search,mcp]
```

**Requirements:** Python 3.10+

---

## How It Works

```
Your Query → know context
  ├─ Semantic Search (fastembed embeddings)
  │    └─ Finds relevant functions/classes
  ├─ Import Graph (AST-based)
  │    └─ Pulls in dependency signatures
  ├─ Test Matcher
  │    └─ Finds related test files
  ├─ Git Recency
  │    └─ Boosts recently-changed code
  ├─ Knowledge Base
  │    └─ Injects cross-session memories
  └─ Token Budget Allocator
       └─ 40% code | 30% imports | 20% summaries | 10% overview
```

All processing is **local**. Embeddings run on your machine. No data leaves your laptop (except `know explain` which calls Claude API).

Semantic search via [fastembed](https://github.com/qdrant/fastembed) is optional — install with `pip install know-cli[search]`. Without it, know uses fast BM25 full-text search as the default.

---

## Configuration

```bash
know init  # Creates .know/config.yaml
```

Configuration lives in `.know/config.yaml`:

```yaml
project:
  name: my-project
  description: "A web application"
languages:
  - python
include_paths:
  - src/
  - lib/
exclude_paths:
  - tests/fixtures/
  - scripts/
```

---

## Pricing

**Free forever:**
- Full CLI (`know context`, `know search`, `know remember`, etc.)
- MCP server
- Local embeddings
- Unlimited usage

**Pro (coming soon):**
- Cloud sync across machines
- Team knowledge sharing
- Advanced analytics
- Priority support

---

## Architecture

```
.know/
  config.yaml     # Project configuration
  daemon.db       # Unified SQLite database (chunks, memories, imports)
  cache/
    index.db      # Scanner cache
```

**Single database:** All data (code chunks, memories, import graph) lives in one SQLite database with WAL mode for concurrent access. The background daemon keeps it in memory; CLI commands fall back to direct access when the daemon is unavailable.

## Contributing

```bash
git clone https://github.com/sushilk1991/know-cli
cd know-cli
pip install -e ".[dev,search,mcp]"
python -m pytest tests/ -v
```

---

## License

MIT
