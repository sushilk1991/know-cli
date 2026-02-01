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

**60-80% fewer tokens. Same (or better) results.**

---

## Quick Start (30 seconds)

```bash
pip install know-cli
cd your-project
know init
know context "help me fix the auth bug" --budget 4000
```

That's it. You just got the most relevant code for your task, packed into exactly 4000 tokens.

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
| `know graph <file>` | Show import dependencies |
| `know explain -c <name>` | AI-explain a component |
| `know stats` | Usage statistics |
| `know status` | Project health check |
| `know reindex` | Rebuild search index |
| `know mcp serve` | Start MCP server |
| `know mcp config` | Print MCP client config |
| `know digest` | Generate codebase summary |
| `know watch` | Auto-update on file changes |

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

## Contributing

```bash
git clone https://github.com/vic/know-cli
cd know-cli
pip install -e ".[dev,search,mcp]"
python -m pytest tests/ -v
```

---

## License

MIT
