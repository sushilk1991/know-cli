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

### Workflow Benchmark (current `main`, July 3, 2026)

Method:
- Repo: `know-cli`, 10 architecture/code-navigation queries.
- Baseline: `grep+read` with common code extensions.
- Candidate: `know workflow --json-compact --read-only`, measured across workflow modes.
- Reported token reduction uses workflow payload tokens, with a separate quality-adjusted reduction that excludes degraded/fallback rows.

| Profile | Grep Tokens | know Tokens | Quality-Adjusted Reduction | Warm p50/p95 | Quality Gate | Deep Graph |
|---|---:|---:|---:|---:|---:|---:|
| `explore` | 715,514 | 12,993 | n/a | 0.419s / 0.430s | fail (deep skipped) | 0% available |
| `implement --max-latency-ms 2000` | 715,514 | 24,501 | **96.6%** | 1.181s / 1.350s | pass | 100% available, 90% non-empty |
| `implement` | 715,514 | 24,595 | **96.6%** | 1.369s / 1.528s | pass | 100% available, 90% non-empty |
| `thorough` | 715,514 | 27,293 | **96.2%** | 1.423s / 1.573s | pass | 100% available, 100% non-empty |

Relevance check from `benchmark/bench_workflow_relevance.py`:
- `implement --max-latency-ms 2000`: expected file surfaced in 10/10 cases; selected top target matched in 6/10.
- `thorough --max-latency-ms 15000`: expected file surfaced in 10/10 cases; selected top target matched in 6/10.
- No latency degradation or deep errors in either relevance run.

Reproduce:

```bash
python benchmark/bench_dual_repo_parallel.py --repo "$PWD" --workflow-mode implement --max-latency-ms 2000 --results-dir /tmp/know-bench
python benchmark/bench_workflow_relevance.py --repo "$PWD" --workflow-mode implement --max-latency-ms 2000 --refresh-index --output /tmp/know-relevance.json
```

### What this means

- **Best default for implementation work:** `know --json workflow "<query>" --mode implement --max-latency-ms 2000 --json-compact`.
- **Single-daemon workflow significantly cuts orchestration overhead** (tool calls dropped to 1/query).
- **Explore is fast but not implementation-safe** because it skips deep context.
- **Thorough improves deep graph completeness** but did not improve expected-file relevance in this 10-case set.
- The biggest product gap is now **top-target precision**: expected files are present, but the selected first target was correct in only 60% of the relevance cases.

---

## Quick Start

```bash
pip install know-cli
cd your-project
know init
know warm
```

Embedding runtime note:
- `fastembed` + `onnxruntime` are included in the default `know-cli` install.
- `know-cli[search]` remains accepted for backward compatibility, but is no longer required.
- If an environment still reports missing fastembed, repair with:

```bash
python -m pip install -U know-cli
know doctor --repair --reindex
```

### Agent Skill Bootstrap (Codex, Claude, Gemini)

`know-cli` includes a built-in agent skill file and installs it automatically on first CLI run.

```bash
# Trigger bootstrap once (after install/upgrade)
know commands

# Verify where skill files were installed
know --json doctor
```

Expected install targets:

- `~/.codex/skills/know-cli/SKILL.md` (or `$CODEX_HOME/skills/know-cli/SKILL.md`)
- `~/.claude/skills/know-cli/SKILL.md`
- `~/.agents/skills/know-cli/SKILL.md`

If a target is missing:

```bash
rm -f ~/.cache/know-cli/skill_bootstrap.json
know commands
know --json doctor
```

Disable auto-install if you manage skills manually:

```bash
KNOW_AUTO_INSTALL_SKILL=0 know --version
```

### What's New in 0.8.7

- New `know workflow "query"` command: single-call daemon workflow (`map -> context -> deep`).
- `know context` now uses daemon-first full v3 context assembly (same ranking/budget pipeline as local fallback).
- `know context` retrieval is now hybrid: lexical BM25 + graph neighborhood + semantic rerank fused with RRF.
- Added graph-first neighborhood expansion (call/import neighbors) before final rerank.
- Added prompt packing policy to reduce "lost in the middle" by placing highest-utility chunks at context edges.
- `know deep` now performs opportunistic stale-file refresh for likely candidate files before resolving symbols.
- `know related` refreshes the target file on demand, reducing stale import/dependency outputs.
- Incremental refresh path re-indexes chunks + symbol refs for changed files without full reindex.
- Fixed search lane weighting bug when OR lane is empty (AND/exact weights now remain stable).

### Single-Call Workflow (Recommended for agents)

```bash
know --json workflow "billing subscription limits" \
  --json-compact \
  --mode implement \
  --max-latency-ms 6000 \
  --map-limit 20 \
  --context-budget 4000 \
  --deep-budget 3000 \
  --session auto
```

JSON profile behavior:
- `--json` on an interactive terminal returns compact JSON.
- `--json` in non-interactive/piped mode returns full JSON (backward-compatible).
- Override explicitly with `--json-compact` or `--json-full`.

Workflow mode behavior:
- `--mode explore`: fastest path (`map + context`), deep step skipped.
- `--mode implement`: balanced quality/speed for normal coding tasks.
- `--mode thorough`: larger budgets + deeper analysis.
- `--max-latency-ms`: hard latency guard; workflow degrades gracefully instead of stalling.

### The 3-Tier Workflow

```bash
# 1. Orient — what functions exist for "billing"?
know map "billing" --session auto

# 2. Investigate — get ranked code bodies (with session tracking)
know --json context "billing subscription" --budget 4000 --session auto
# Returns session_id: "a1b2c3d4"

# 3. Go deep — one function + its callers/callees
know --json deep "check_cloud_access" --budget 3000

# Follow-up queries skip already-seen code
know --json context "payment processing" --budget 4000 --session a1b2c3d4
```

---

## Commands Reference

### Simplified Layout (Backward-Compatible)

Use this minimal command set for day-to-day flow:

```bash
know ask "billing subscription limits"
know recall "what did we decide about auth?"
know decide "Use daemon workflow" --why "fewer tool calls"
know done 12
know hooks suggest --agent claude
know docs
know status
```

Legacy top-level commands are unchanged and still supported:
`know workflow`, `know digest`, `know diagram`, `know memories ...`, etc.
Use `know commands --all` to list everything.

### Workflow — Single Daemon Call

```bash
know workflow "billing subscription limits"
know --json workflow "auth token validation" --json-compact --context-budget 5000 --deep-budget 2500
know --json workflow "auth token validation" --mode explore --max-latency-ms 2500 --json-compact
```

Runs `map -> context -> deep` in a single daemon request to cut tool-call overhead for coding agents.
The response now includes `workflow_mode`, `latency_budget_ms`, stage timings (`latency_ms`) and whether it degraded due to latency.
Human-readable output and JSON now also include a `usage` block (`tokens_used` + `elapsed_ms`) to make cost/latency visible after each run.

Recommended by task type:
- Exploration / architecture: `know --json workflow "<query>" --mode explore --json-compact`
- Implementation task: `know --json workflow "<query>" --mode implement --json-compact`
- Complex refactor: `know --json workflow "<query>" --mode thorough --max-latency-ms 15000 --json-compact`

### Hook-First Adoption (Suggest Mode)

```bash
know hooks suggest --agent claude
know --json hooks suggest --agent claude
know hooks suggest --agent codex
```

- `hooks suggest` is non-mutating guidance only.
- Claude output provides PreToolUse suggestion snippets.
- Codex output provides skill + workflow command guidance.
- Suggest output intentionally never emits rewrite payloads like `updatedInput`.

### Docs Update Policy (Local-First)

- Recommended default: run docs updates locally via `know watch` or `know update`.
- For Git-based freshness across commits/checkouts/merges, use:
  - `know hooks install --index-hooks` (post-commit + post-merge + post-checkout index refresh)
  - `know hooks uninstall --index-hooks` (remove all know-managed index refresh hooks)
  - `know hooks status` (verify hook state)
- `know hooks install` is non-mutating: it never runs `git add`, `git commit`, or `git commit --amend`.
- Upgrades rewrite stale know-managed hooks in place; run `know hooks install` once after upgrading.
- GitHub workflow examples in this repo are `workflow_dispatch` and **non-mutating by default**.
- If a downstream repo enables CI docs updates, keep CI read-only (review/comment artifacts) unless explicit auto-commit is required.

Targeted docs refresh:

```bash
know update --only system    # updates docs/arc.md only
know update --only diagrams  # updates docs/architecture.md (mermaid) or docs/architecture-c4.md (plantuml)
```

`docs/arc.md` now uses deterministic scan evidence (file/module/language stats + key paths) instead of free-form project-name inference.

### Background Auto-Fill (No Extra Flags)

- Daemon now performs incremental background refresh of changed/deleted files (no manual full reindex loop needed for normal edits).
- `know warm` starts daemon and reports index readiness (`warming` vs `complete`) without request-thread full indexing.
- Full indexing now purges out-of-scope artifacts from old indexes (for example `.venv*`, `site-packages`, `build`, `dist`, cache trees).
- Workflow sessions are persisted to `.know/current_session`.
- `know remember` and `know decide` auto-fill `session_id` from the active session when not provided.
- `--session auto` / `--session new` now resolve to concrete IDs consistently (workflow/context/map/deep) and persist active session.
- Reliability fallback: `know doctor --repair --reindex` repairs embedding cache issues and rebuilds chunk index.

Daemon auto-refresh controls:

```bash
KNOW_DAEMON_AUTO_REFRESH=1        # default on
KNOW_DAEMON_REFRESH_INTERVAL=60   # seconds, min 15
KNOW_DAEMON_AUTO_REFRESH_MAX_FILES=2500  # auto-suspend over this size unless explicitly forced
```

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
know decide "Use single daemon workflow" --why "lower tool-call overhead" --evidence src/know/daemon.py:572
know recall "workflow decisions" --type decision --status active
know memories resolve 12 --status resolved
know memories export > memories.json
```

Memories are automatically included in `know context` results.
Structured memory fields now include `memory_type`, `decision_status`, `confidence`, `evidence`, `session_id`, `agent`, and `trust_level`.
If `session_id` is omitted, `remember/decide` auto-bind to `.know/current_session` when available.

### All Commands

| Command | Description |
|---------|-------------|
| `know ask "query"` | Simple one-command retrieval (workflow wrapper) |
| `know docs` | One-shot docs refresh (system + digest + api + architecture) |
| `know done <id>` | Shortcut for `know memories resolve <id> --status resolved` |
| `know commands --all` | Show full command list |
| `know workflow "query"` | Single-call daemon workflow (map + context + deep) |
| `know warm` | Start daemon + check warmup/index readiness |
| `know hooks install` | Install non-mutating post-commit index hook (use `--index-hooks` for merge/checkout hooks) |
| `know hooks uninstall` | Remove know-managed hooks (use `--index-hooks` for merge/checkout hooks) |
| `know hooks status` | Show hook status for post-commit/merge/checkout |
| `know hooks suggest --agent <claude\|codex>` | Suggest hook-first retrieval adoption (non-mutating) |
| `know watch` | Watch local file edits and refresh docs continuously |
| `know map "query"` | Lightweight signature search |
| `know context "query"` | Smart, budgeted code context |
| `know deep "name"` | Function + callers + callees |
| `know search "query"` | Semantic code search |
| `know grep "query"` | Grep+read baseline with token/time telemetry |
| `know remember "text"` | Store a memory |
| `know decide "decision"` | Store structured decision memory |
| `know recall "query"` | Recall memories |
| `know memories resolve <id>` | Resolve/supersede/reject a memory |
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

### MCP Memory Interop (Cross-Agent)

When using `know mcp serve`, agents can share structured memory through MCP tools:

- `remember(...)` with structured fields (`memory_type`, `decision_status`, `confidence`, `evidence`, `session_id`, `agent`, `trust_level`)
- `recall(...)` with filters (`memory_type`, `decision_status`, blocked/expiry controls)
- `resolve_memory(memory_id, status)`
- `export_memories()`

This is the canonical path for Codex/Claude/Gemini memory portability.

### Global Flags

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--json-compact` | Compact workflow JSON profile (`workflow` command) |
| `--json-full` | Full workflow JSON profile (`workflow` command, strict schema compatibility) |
| `--quiet` | Minimal output |
| `--verbose` | Detailed output |
| `--time` | Show execution time |

### Workflow Flags

| Flag | Description |
|------|-------------|
| `--mode explore|implement|thorough` | Choose speed-vs-depth workflow profile |
| `--max-latency-ms N` | End-to-end budget; workflow skips expensive steps when needed |

### Retrieval Analytics

`know stats` now includes retrieval KPIs for `map`, `context`, `deep`, and `workflow`:
- per-command call count, p50/p95 latency, average tokens
- workflow quality rates:
  - degraded-by-latency percentage
  - call-graph available percentage
  - non-empty caller/callee edge percentage

JSON output includes a `retrieval` block while preserving existing stats keys.

---

## Works With

| Tool | Integration |
|------|-------------|
| **Claude Code** | Agent skill — Claude uses know automatically |
| **Claude Desktop** | MCP server: `know mcp serve` |
| **Cursor** | MCP server or CLI |
| **Any CLI agent** | Pipe-friendly: `know --json context "query"` |

## Agent Skill File (Auto-Installed)

`know-cli` ships with an agent skill file (`KNOW_SKILL.md`) and auto-installs it on first CLI run (per version) into common agent homes:

- `~/.codex/skills/know-cli/SKILL.md` (or `$CODEX_HOME/skills/know-cli/SKILL.md`)
- `~/.claude/skills/know-cli/SKILL.md`
- `~/.agents/skills/know-cli/SKILL.md`

Verify installation:

```bash
know --json doctor
```

In JSON output, check `checks.agent_skill.targets.*.exists`.

Disable auto-install if needed:

```bash
KNOW_AUTO_INSTALL_SKILL=0 know --version
```

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

## Pricing

`know-cli` is free and open-source (MIT). It runs locally and does not add usage-based fees.
If you enable optional model APIs in your own workflows, those provider costs are separate.

## Troubleshooting

If you see `Error: No such command 'workflow'`:

```bash
which know
know --version
know commands --all | grep workflow
python -c "import know,sys; print(know.__version__, know.__file__, sys.executable)"
```

If the command path/version is wrong, reinstall in the active environment:

```bash
python -m pip uninstall -y know know-cli
python -m pip install -U know-cli
```

`know` is the only supported command name.

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
