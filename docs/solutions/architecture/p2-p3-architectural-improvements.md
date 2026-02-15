---
title: "P2/P3 Architectural Improvements: DB Consolidation, Message Framing, CLI Split"
category: architecture
tags:
  - sqlite
  - daemon
  - cli-refactor
  - message-framing
  - database-consolidation
  - ipc
module: know-cli
symptoms:
  - silent exception swallowing hiding bugs
  - fragmented data across 3 SQLite databases
  - monolithic 1575-line cli.py
  - raw socket reads vulnerable to partial message delivery
  - CLI commands bypassing daemon IPC
date: 2026-02-15
severity: P2
---

# P2/P3 Architectural Improvements

## Problem

After the initial v2 daemon architecture was implemented and critical P1 issues were addressed, five P2/P3 architectural issues remained — identified by a 6-agent code review:

1. **Silent exception handlers** (~12 `except Exception: pass` blocks) hiding bugs in non-critical paths
2. **No message framing** on the Unix socket protocol — raw `readline()` vulnerable to partial delivery and large message truncation
3. **Three separate SQLite databases** (`daemon.db`, `knowledge.db`, import graph in `index.db`) causing data fragmentation, multiple connections, and schema duplication
4. **CLI agent commands** directly accessing DaemonDB instead of going through daemon IPC
5. **Monolithic `cli.py`** at 1575 lines with 20+ commands in a single file

## Root Cause

These were deliberate shortcuts taken during v1/v2 rapid development:
- Exception silencing was used to keep optional features from crashing the CLI
- `readline()` was the simplest socket read approach
- Each subsystem (knowledge, imports, chunks) created its own database independently
- Agent commands were added before the daemon IPC layer existed
- The CLI grew organically without a module split point

## Solution

### Task 5: Replace Silent Exception Handlers with Logging

Added `logger = get_logger()` at module level and replaced ~12 instances:

```python
# Before
except Exception:
    pass

# After
except Exception as e:
    logger.debug(f"Description of what failed: {e}")
```

All replacements were in non-critical paths (stats tracking, auto-memory, memory injection) where failure should be logged but not crash the CLI.

### Task 3: Add Message Framing to Socket Protocol

Added `struct`-based 4-byte big-endian length prefix to both send and receive:

```python
async def write_framed_message(writer, data):
    writer.write(struct.pack(">I", len(data)))
    writer.write(data)
    await writer.drain()

async def read_framed_message(reader):
    header = await reader.readexactly(4)
    length = struct.unpack(">I", header)[0]
    return await reader.readexactly(length)
```

Key details:
- 10MB max message size validation
- `asyncio.IncompleteReadError` handling for clean disconnect detection
- Updated both `KnowDaemon._handle_connection()` and `DaemonClient.call()`

### Task 1: Consolidate 3 Databases into 1

Made `DaemonDB` (`.know/daemon.db`) the single source of truth:

**DaemonDB extensions:**
- Added `embedding BLOB DEFAULT NULL` column to memories table
- Schema migration via `PRAGMA table_info` to detect missing columns
- 6 new methods: `get_memory_by_id()`, `delete_memory()`, `list_memories()`, `count_memories()`, `recall_memories_semantic()`, `get_all_edges()`

**KnowledgeBase → thin DaemonDB wrapper:**
- Replaced direct SQLite connection with `self._db = DaemonDB(config.root)`
- Integer-to-text UUID ID mapping for CLI backward compatibility (users say `know forget 3`, DaemonDB stores UUIDs)
- Recall chain: semantic (cosine similarity) → FTS5 (BM25) → text (LIKE fallback)

**ImportGraph → DaemonDB delegate:**
- Replaced `self._conn` / `_ensure_table()` with `self._db = DaemonDB(config.root)`
- `build()` writes via `self._db.set_imports()`, reads via `get_imports_of()` / `get_imported_by()`

### Task 2: Wire CLI Commands Through Daemon IPC

Added daemon-first pattern with direct DB fallback:

```python
def _get_daemon_client(config):
    if os.environ.get("KNOW_NO_DAEMON"):
        return None
    try:
        from know.daemon import ensure_daemon
        return ensure_daemon(config.root, config)
    except Exception as e:
        logger.debug(f"Daemon unavailable, falling back: {e}")
        return None
```

Updated 4 agent commands (`next-file`, `signatures`, `related`, `generate-context`) to use `client.call_sync()` with graceful fallback.

### Task 4: Split cli.py into Sub-modules

Split 1575-line monolith into 9-module package:

```
src/know/cli/
  __init__.py    — main group, shared utilities, command registration
  core.py        — init, explain, diagram, api, onboard, digest, update, watch
  search.py      — search, context, graph, reindex
  knowledge.py   — remember, recall, forget, memories group
  stats.py       — stats, status
  hooks.py       — hooks group (install/uninstall)
  mcp.py         — mcp group (serve/config)
  agent.py       — next-file, signatures, related, generate-context
  diff.py        — diff command
```

`pyproject.toml` entry point `know.cli:main` works unchanged since `know.cli` is now a package with `main` in `__init__.py`.

## Prevention

- **Exception handling**: Always log exceptions, even in non-critical paths. Use `logger.debug()` at minimum.
- **Protocol design**: Always use length-prefixed framing for IPC. Never rely on `readline()` for structured data.
- **Database architecture**: Decide on single vs. multiple databases early. Consolidate before the schema diverges.
- **IPC wiring**: New commands should go through the daemon IPC layer from day one, with direct DB fallback for CI/CD.
- **Module size**: Split files proactively when they exceed ~500 lines.

## Verification

All changes verified with:
- 169 existing tests + 27 v2 tests passing (196 total)
- `know --help` and all 25 commands functional
- Memory round-trip: `know remember` / `know recall`
- Agent commands: `know next-file --help`
- `from know.cli import cli` backward compatibility maintained

## Related

- Plan: `docs/plans/2026-02-15-refactor-know-cli-v2-daemon-architecture-plan.md`
- Branch: `feat/v2-daemon-architecture`
