---
title: "Refactor know-cli v2: Daemon-Backed Context Engine for AI Agents"
type: refactor
date: 2026-02-15
---

# Refactor know-cli v2: Daemon-Backed Context Engine for AI Agents

## Overview

Transform know-cli from a slow, Python-only, runtime-scanning tool (current score: 5.5/10) into a daemon-backed, multi-language, sub-100ms context engine that CLI coding agents (Claude Code, Codex, Aider) genuinely want to use. Target: 10/10 technical quality.

**Core insight from audit:** The tool currently makes agents *slower*, not faster. Every `know context` call re-scans the entire filesystem, re-parses every file, and re-embeds all chunks (5-30 seconds). Claude Code's native `Grep` + `Read` takes <200ms. We must invert this: pre-compute everything, serve instantly.

**The two things agents can't do natively:**
1. **Persistent cross-session memory** — agents forget everything between sessions
2. **Pre-computed project understanding** — agents re-discover the same things every session

Everything else (search, context assembly) must be faster than the agent's built-in tools or it's worthless.

## Problem Statement

### Current Technical Debt (from audit)

| Problem | Severity | Impact |
|---------|----------|--------|
| Full filesystem re-scan on every query | Critical | 5-30s latency per call |
| Python-only AST chunking | Critical | Non-Python files = whole-file dumps |
| Token counting without tiktoken | High | 20-40% budget inaccuracy |
| Triple-redundant embedding model loading | High | 300MB memory waste, 3x load time |
| Import graph uses leaf-name matching | High | False dependency edges |
| model_router.py with fictional data | Medium | Dead code, misleading |
| Fabricated baseline in stats (50K tokens) | Medium | Misleading ROI claims |
| SQL interpolation for table names | Low | Security anti-pattern |
| 30+ `except Exception: pass` silently swallowing errors | Medium | Impossible to debug |
| MCP tools are async-decorated but synchronous | Medium | Event loop blocking |

### Competitive Gap

| Capability | Claude Code Native | know-cli v0.3 | know-cli v2 Target |
|-----------|-------------------|---------------|-------------------|
| Code search | Grep (<100ms) | Embedding search (3-10s) | Hybrid search (<100ms) |
| File read | Read (<50ms) | Full re-scan (5-30s) | Pre-indexed (<50ms) |
| Cross-session memory | None | Basic (works) | Best-in-class |
| Project understanding | Manual CLAUDE.md | Auto-generated (slow) | Auto-maintained (<1s) |
| Multi-language | All | Python only | 8 languages |
| Setup friction | Zero | `know init` + indexing | Zero-config auto-start |

## Proposed Solution

### Architecture: Daemon + Thin Client

```
┌─────────────────────────────────────────────────────┐
│                 Background Daemon                    │
│              (one per project root)                  │
│                                                      │
│  ┌───────────┐  ┌──────────────────────────────┐    │
│  │   File    │  │        SQLite Store           │    │
│  │  Watcher  │  │                                │    │
│  │ (watchdog │  │  chunks     : AST chunks+meta │    │
│  │  200ms    │  │  embeddings : sqlite-vec HNSW │    │
│  │  debounce)│  │  memory_fts : FTS5 keyword    │    │
│  └─────┬─────┘  │  imports    : qualified edges │    │
│        │        │  memories   : cross-session KB │    │
│        ▼        └──────────────┬─────────────────┘    │
│  ┌───────────┐               │                      │
│  │ Tree-sitter│               │                      │
│  │ Parser    │───────────────┘                      │
│  │ (8 langs) │                                      │
│  └───────────┘                                      │
│        │                                             │
│  ┌─────┴──────────────────────┐                     │
│  │    EmbeddingManager        │                     │
│  │ (single model, optional)   │                     │
│  └────────────────────────────┘                     │
│                                                      │
│  ┌────────────────────────────┐                     │
│  │   Unix Socket IPC          │                     │
│  │   JSON-RPC protocol        │                     │
│  └────────────┬───────────────┘                     │
└───────────────┼──────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───┴────┐           ┌─────┴─────┐
│  CLI   │           │ MCP Server│
│ (thin  │           │  (thin    │
│ client)│           │  proxy)   │
└────────┘           └───────────┘
```

### Design Decisions

1. **Daemon auto-starts on first CLI call.** No explicit `know daemon start` required. Socket at `~/.know/sockets/<project-hash>.sock`. Auto-shutdown after 30 min idle.
2. **Synchronous fallback when daemon unavailable.** CI/CD auto-detected via `CI` env var or `--sync` flag. Does a fast cold scan without embedding.
3. **FTS5 (BM25) as default search.** Embeddings are opt-in via `know-cli[search]`. Text search handles 80% of agent queries with zero heavyweight dependencies.
4. **One daemon per project root.** Multiple projects = multiple daemons. Managed via PID files.
5. **Tree-sitter for all chunking.** No more `ast.parse()`. Uniform chunking for 8 languages.
6. **Anthropic token counting via local heuristic calibrated against API.** No runtime API calls for counting. Ship a calibration table.

---

## Technical Approach

### Phase 1: Foundation — Fix the Broken Core (Week 1-2)

The goal is to fix correctness bugs and eliminate dead code without changing architecture. Every change is independently testable and shippable.

#### 1.1 Delete Dead Code

**Files to delete:**
- `src/know/model_router.py` — fictional model data, no callers except two CLI commands
- Remove `know route` and `know burnrate` CLI commands from `cli.py`

**Files to simplify:**
- `src/know/quality.py` — replace string-counting heuristics with Tree-sitter node counting (Phase 2 dependency) or remove entirely. The current `content.count("if ")` matching inside strings/comments is worse than nothing.
- `src/know/stats.py` — remove `AVG_TOKENS_NAIVE = 50000` fabricated baseline. Report actual numbers only.

#### 1.2 Fix Token Counting

**File:** `src/know/token_counter.py`

- Add `tiktoken` as a core dependency in `pyproject.toml`
- Use `cl100k_base` as the default encoding (best available approximation for Claude)
- Ship a calibration offset: measure `cl100k_base` vs Anthropic `count_tokens` API for code samples, store the ratio (~1.05-1.15x)
- Apply calibration when `provider=anthropic` is configured
- Add `provider` parameter to `count_tokens()` and `truncate_to_budget()`
- Remove the crude `CODE_TOKENS_PER_WORD = 1.3` heuristic — tiktoken is fast and accurate enough

```python
# src/know/token_counter.py — new interface
def count_tokens(text: str, provider: str = "anthropic") -> int:
    """Count tokens using provider-appropriate tokenizer."""
    ...

def truncate_to_budget(text: str, budget: int, provider: str = "anthropic") -> str:
    """Truncate text to fit within token budget."""
    ...
```

#### 1.3 Fix Import Graph

**File:** `src/know/import_graph.py`

- Store fully-qualified module names instead of leaf names
- Replace `short = name.split(".")[-1]` with full dotted path matching
- Remove `LIKE` queries — use exact `WHERE source = ?` matches
- Add a module registry that maps file paths to their fully-qualified module names
- Current: `import os.path` matches project's `myapp.path` (WRONG)
- Fixed: `import os.path` resolves to `os.path`, only matches if `os.path` is a project module

```python
# Before (broken):
short = alias.name.split(".")[-1]
if short in known_modules: ...

# After (correct):
full_name = alias.name  # e.g., "know.scanner"
if full_name in known_modules: ...
# Also check prefix matches for submodule imports
```

#### 1.4 Centralize Embedding Management

**New file:** `src/know/embeddings.py`

- Single `EmbeddingManager` class with process-wide singleton
- Lazy loading with proper thread-safe caching
- Used by `context_engine.py`, `semantic_search.py`, `knowledge_base.py`
- Delete the three duplicate caches

```python
# src/know/embeddings.py
class EmbeddingManager:
    """Singleton embedding model manager. Loads model once, shared everywhere."""
    _instance: Optional["EmbeddingManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "EmbeddingManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts. Returns (N, 384) array."""
        ...

    def is_available(self) -> bool:
        """Check if fastembed is installed."""
        ...
```

#### 1.5 Fix SQL Interpolation

**Files:** `src/know/semantic_search.py`, `src/know/index.py`

- Replace f-string table name interpolation with parameterized queries where possible
- For table names (which can't be parameterized in SQLite), validate against `^[a-zA-Z0-9_]+$` regex before interpolation

---

### Phase 2: Multi-Language Chunking via Tree-sitter (Week 2-3)

#### 2.1 Replace AST Parsing with Tree-sitter

**File:** `src/know/parsers.py` (major rewrite)

Add Tree-sitter grammars for 8 languages. Each language gets a query file that extracts functions, classes, and methods:

| Language | Grammar Package | Key Node Types |
|----------|----------------|----------------|
| Python | `tree-sitter-python` | `function_definition`, `class_definition` |
| TypeScript/JS | `tree-sitter-typescript` | `function_declaration`, `class_declaration`, `method_definition` |
| Go | `tree-sitter-go` | `function_declaration`, `method_declaration` |
| Rust | `tree-sitter-rust` | `function_item`, `impl_item` |
| Java | `tree-sitter-java` | `method_declaration`, `class_declaration` |
| Ruby | `tree-sitter-ruby` | `method`, `class` |
| C/C++ | `tree-sitter-c`, `tree-sitter-cpp` | `function_definition`, `struct_specifier` |

**New architecture:**

```python
# src/know/parsers.py — Tree-sitter unified parser
class TreeSitterChunker:
    """Extract function/class chunks from any supported language."""

    def __init__(self):
        self._parsers: dict[str, Parser] = {}
        self._queries: dict[str, Query] = {}

    def chunk_file(self, path: Path, content: bytes) -> list[CodeChunk]:
        """Parse file with Tree-sitter, extract function/class chunks."""
        lang = self._detect_language(path)
        parser = self._get_parser(lang)
        tree = parser.parse(content)
        return self._extract_chunks(tree, lang, path, content)
```

#### 2.2 Update Context Engine Chunking

**File:** `src/know/context_engine.py`

- Replace `extract_chunks_from_file()` (Python-only via `ast.parse()`) with `TreeSitterChunker.chunk_file()`
- All languages now produce function/class-level chunks instead of whole-file dumps
- Remove the `ast` import entirely from context_engine.py

#### 2.3 Multi-Language Import Graph

**File:** `src/know/import_graph.py`

- Add Tree-sitter queries for import extraction per language:
  - Python: `import_statement`, `import_from_statement`
  - TypeScript: `import_statement` (ES6), `require` calls
  - Go: `import_declaration`
  - Rust: `use_declaration`
  - Java: `import_declaration`
  - Ruby: `require`, `require_relative`
  - C/C++: `preproc_include`
- Normalize all imports to fully-qualified module paths for the graph

#### 2.4 Update pyproject.toml Dependencies

Move Tree-sitter from optional `[parser]` group to core dependencies:

```toml
dependencies = [
    "click>=8.1.0",
    "rich>=13.0.0",
    "pyyaml>=6.0",
    "watchdog>=3.0.0",
    "pathspec>=0.11.0",
    "xxhash>=3.0.0",
    "tiktoken>=0.5.0",
    "tree-sitter>=0.24.0",
    "tree-sitter-python>=0.23.0",
    "tree-sitter-javascript>=0.23.0",
    "tree-sitter-typescript>=0.23.0",
    "tree-sitter-go>=0.23.0",
    "tree-sitter-rust>=0.23.0",
    "tree-sitter-java>=0.23.0",
    "tree-sitter-ruby>=0.23.0",
    "tree-sitter-c>=0.23.0",
    "tree-sitter-cpp>=0.23.0",
]
```

Remove `anthropic` from core dependencies (only needed for `know explain`, move to optional `[ai]` group).

---

### Phase 3: Background Daemon Architecture (Week 3-5)

This is the most impactful change. It converts all hot-path operations from O(files) to O(1).

#### 3.1 Daemon Process

**New file:** `src/know/daemon.py`

```python
class KnowDaemon:
    """Background daemon that maintains hot indexes for a project.

    Lifecycle:
    - Auto-started on first CLI call (no explicit start needed)
    - Listens on Unix socket at ~/.know/sockets/<project-hash>.sock
    - PID file at ~/.know/pids/<project-hash>.pid
    - Auto-shutdown after 30 min idle
    - Crash recovery: stale PID detection, SQLite WAL mode for consistency

    Protocol: JSON-RPC 2.0 over Unix domain socket
    """

    def __init__(self, project_root: Path, config: Config):
        self.root = project_root
        self.config = config
        self.db = DaemonDatabase(project_root)
        self.watcher = FileWatcher(project_root, self._on_file_change)
        self.chunker = TreeSitterChunker()
        self.embedder = EmbeddingManager.get() if EmbeddingManager.is_available() else None
        self._idle_timer = IdleTimer(timeout=1800)  # 30 min

    async def serve(self):
        """Main event loop: listen on socket, handle requests."""
        ...

    async def _handle_request(self, method: str, params: dict) -> dict:
        """Route JSON-RPC methods to handlers."""
        handlers = {
            "context": self._handle_context,
            "search": self._handle_search,
            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "next_file": self._handle_next_file,
            "signatures": self._handle_signatures,
            "related": self._handle_related,
            "status": self._handle_status,
        }
        ...

    def _on_file_change(self, path: Path, event_type: str):
        """Incrementally re-index changed file."""
        ...
```

#### 3.2 Daemon Database (SQLite with sqlite-vec + FTS5)

**New file:** `src/know/daemon_db.py`

```sql
-- Schema for the unified daemon database: .know/index.db

-- Code chunks (Tree-sitter extracted)
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL,
    chunk_name TEXT NOT NULL,          -- function/class name
    chunk_type TEXT NOT NULL,          -- 'function', 'class', 'method'
    language TEXT NOT NULL,            -- 'python', 'typescript', etc.
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT,                    -- function signature only
    body TEXT NOT NULL,                -- full body text
    body_hash TEXT NOT NULL,           -- xxhash for change detection
    token_count INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

-- FTS5 index on chunk content (BM25 search — the DEFAULT search)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_name, signature, body,
    content='chunks', content_rowid='id'
);

-- Optional: vector embeddings via sqlite-vec (only if fastembed installed)
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    embedding float[384]
);

-- Import graph (fully-qualified edges)
CREATE TABLE imports (
    source_module TEXT NOT NULL,       -- fully-qualified: "know.scanner"
    target_module TEXT NOT NULL,       -- fully-qualified: "know.models"
    import_type TEXT NOT NULL,         -- 'import', 'from_import'
    PRIMARY KEY (source_module, target_module)
);

-- Cross-session memories
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT,                         -- JSON array
    source_type TEXT DEFAULT 'manual', -- 'manual', 'auto', 'agent'
    quality_score REAL DEFAULT 1.0,    -- 0.0-1.0
    access_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL,
    content_hash TEXT NOT NULL         -- for deduplication
);

-- FTS5 on memories
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='rowid'
);

-- Optional: memory vectors
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    embedding float[384]
);

-- File metadata (change detection)
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT,
    chunk_count INTEGER DEFAULT 0
);
```

#### 3.3 Thin CLI Client

**Modify:** `src/know/cli.py`

Every command that hits the daemon follows this pattern:

```python
def _daemon_call(method: str, params: dict, project_root: Path) -> dict:
    """Call daemon over Unix socket. Auto-start if not running. Fallback to sync."""
    sock_path = _get_socket_path(project_root)

    # Try connecting to existing daemon
    try:
        return _rpc_call(sock_path, method, params, timeout=5.0)
    except ConnectionRefusedError:
        pass

    # Auto-start daemon
    _start_daemon(project_root)
    _wait_for_socket(sock_path, timeout=10.0)
    return _rpc_call(sock_path, method, params, timeout=5.0)

def _sync_fallback(method: str, params: dict, config: Config) -> dict:
    """Synchronous fallback for CI/CD or when daemon fails."""
    # Uses direct function calls without daemon overhead
    # No embedding, FTS5/BM25 only for search
    ...
```

#### 3.4 MCP Server as Daemon Proxy

**Modify:** `src/know/mcp_server.py`

- MCP server becomes a thin proxy to the daemon
- All tool handlers forward to daemon via Unix socket
- Truly async — no blocking operations in the event loop
- Connection pooling to daemon socket

```python
@server.tool()
async def get_context(query: str, budget: int = 8000) -> str:
    """Get token-budgeted context for a coding task."""
    result = await _async_daemon_call("context", {"query": query, "budget": budget})
    return result["context"]
```

#### 3.5 File Watcher Integration

**Modify:** `src/know/watcher.py`

- Debounce file events at 200ms
- Filter by `.gitignore` patterns (already partially implemented)
- On change: re-chunk file via Tree-sitter, update `chunks` table, update FTS5, optionally re-embed
- On delete: remove file's chunks from all tables
- Batch changes during rapid saves (e.g., `git checkout`)

---

### Phase 4: Cross-Session Memory as Core Product (Week 4-5)

#### 4.1 Hybrid Memory Retrieval

**Modify:** `src/know/knowledge_base.py`

```python
class KnowledgeBase:
    def recall(self, query: str, limit: int = 10) -> list[Memory]:
        """Hybrid recall: FTS5 (BM25) + optional vector search."""
        # 1. BM25 search via FTS5 (always available, fast)
        fts_results = self._fts5_search(query, limit=limit * 2)

        # 2. Optional: vector search via sqlite-vec
        vec_results = []
        if self._embedder and self._has_vectors:
            vec_results = self._vector_search(query, limit=limit * 2)

        # 3. Reciprocal Rank Fusion to combine results
        combined = self._rrf_merge(fts_results, vec_results)

        # 4. Filter stale memories (quality_score < 0.3 AND age > 90 days)
        filtered = [m for m in combined if not self._is_stale(m)]

        # 5. Update access counts
        self._bump_access_counts([m.id for m in filtered[:limit]])

        return filtered[:limit]
```

#### 4.2 Semantic Deduplication

```python
def remember(self, content: str, tags: list[str] = None, source: str = "manual") -> str:
    """Store a memory with semantic deduplication."""
    content_hash = xxhash.xxh64(content).hexdigest()

    # Exact duplicate check
    if self._exists_by_hash(content_hash):
        return self._update_existing(content_hash)

    # Semantic duplicate check (if embeddings available)
    if self._embedder:
        similar = self._vector_search(content, limit=1)
        if similar and similar[0].score > 0.92:
            return self._merge_memories(similar[0], content, source)

    # New memory
    return self._insert(content, tags, source)
```

#### 4.3 Quality Scoring and Staleness

```python
def _compute_quality(self, memory: Memory) -> float:
    """Quality = recency * frequency * source_weight."""
    days_old = (time.time() - memory.created_at) / 86400
    recency = max(0.1, 1.0 - (days_old / 365))  # Decay over 1 year
    frequency = min(1.0, memory.access_count / 10)  # Saturate at 10 accesses
    source_weight = {"manual": 1.0, "agent": 0.8, "auto": 0.6}.get(memory.source_type, 0.5)
    return recency * 0.4 + frequency * 0.3 + source_weight * 0.3

def _is_stale(self, memory: Memory) -> bool:
    """Memory is stale if quality < 0.3 AND older than 90 days."""
    return memory.quality_score < 0.3 and (time.time() - memory.created_at) > 86400 * 90
```

#### 4.4 Database Migration

**New file:** `src/know/migrations.py`

- Detect schema version from a `meta` table
- Auto-migrate on daemon startup
- Backup old database before migration
- v0.3 -> v2.0: Move memories from `knowledge.db` to unified `index.db`, add FTS5 + sqlite-vec tables

---

### Phase 5: New Agent-Optimized APIs (Week 5-6)

#### 5.1 Incremental Context Commands

**New CLI commands (thin clients to daemon):**

```bash
# Return the single most relevant file path for a query
# Accepts --exclude to avoid previously seen files
know next-file "fix auth bug" --exclude src/auth/middleware.py --exclude src/auth/tokens.py

# Return just function/class signatures for a file (no bodies)
know signatures src/auth/middleware.py

# Return import dependencies and dependents for a file
know related src/auth/middleware.py

# Traditional bundled context (still available, now fast)
know context "fix auth bug" --budget 4000
```

**MCP tools (matching CLI commands):**

```python
@server.tool()
async def next_file(query: str, exclude: list[str] = []) -> str:
    """Get the next most relevant file for a task. Use --exclude for files already seen."""

@server.tool()
async def get_signatures(file_path: str) -> str:
    """Get function/class signatures for a file (no bodies)."""

@server.tool()
async def get_related(file_path: str) -> str:
    """Get import dependencies and dependents for a file."""
```

#### 5.2 Static Context File Generation

**New command:** `know generate-context`

Generates `.know/CONTEXT.md` with:
- Project name, languages, file count
- Top-level module structure (directories with file counts)
- Key public function/class signatures (top 50 by import count)
- Stored memories (top 20 by quality score)
- Recent git activity summary

```bash
know generate-context              # Generate .know/CONTEXT.md
know generate-context --budget 4000  # Limit to 4000 tokens
```

Auto-regenerated on `know init` and `know reindex`. NOT on every file change (would create git noise). Add `.know/CONTEXT.md` to `.gitignore` by default.

#### 5.3 Zero-Config Auto-Init

**Modify:** `src/know/cli.py`

- Remove the hard requirement for `know init` before other commands
- On first `know context` / `know search` / etc., auto-detect project root (walk up looking for `.git/`)
- Auto-create `.know/` directory with default config
- Auto-start daemon, begin indexing in background
- Return results immediately using whatever is indexed so far (graceful degradation)

```python
def _ensure_initialized(project_root: Path) -> Config:
    """Auto-initialize if .know/ doesn't exist. Zero-config."""
    know_dir = project_root / ".know"
    if not know_dir.exists():
        config = Config.create_default(project_root)
        config.save()
        logger.info("Auto-initialized know-cli in %s", project_root)
    return Config.load(project_root)
```

---

### Phase 6: Polish and Performance (Week 6-7)

#### 6.1 Binary Quantization for Embeddings (Optional Path)

When `fastembed` is installed:
- Store both full-precision (float32) and binary (1-bit sign) embeddings
- Search: query binary index first (bitwise XOR, extremely fast), get top-100 candidates
- Re-rank candidates using full-precision cosine similarity
- Result: ~40% faster search with >99.9% recall parity

#### 6.2 Connection and Resource Management

- Add `atexit` handlers for all SQLite connections
- Implement proper context manager protocol on all database classes
- Daemon graceful shutdown on SIGTERM/SIGINT
- Stale PID file detection (check if PID is alive before declaring daemon running)

#### 6.3 Error Handling Overhaul

- Replace all `except Exception: pass` with specific exception types
- Add structured logging for swallowed errors (log at DEBUG level, not silently ignore)
- Add error boundaries in MCP tool handlers (return structured errors, not unhandled exceptions)
- Add `--debug` flag that enables full tracebacks

#### 6.4 Test Suite

```
tests/
  test_chunker.py          # Tree-sitter chunking for all 8 languages
  test_import_graph.py     # Fully-qualified import resolution
  test_token_counter.py    # Accurate counting with tiktoken
  test_embeddings.py       # Centralized EmbeddingManager
  test_knowledge_base.py   # Memory CRUD, deduplication, staleness
  test_daemon.py           # Daemon lifecycle, IPC, crash recovery
  test_daemon_db.py        # Schema, migrations, FTS5, sqlite-vec
  test_cli_integration.py  # End-to-end CLI commands
  test_mcp_server.py       # MCP tool responses
  test_performance.py      # Latency benchmarks (<100ms assertions)
```

#### 6.5 Dependency Cleanup

**Core dependencies (always installed):**
```
click, rich, pyyaml, watchdog, pathspec, xxhash, tiktoken
tree-sitter, tree-sitter-{python,javascript,typescript,go,rust,java,ruby,c,cpp}
```

**Optional groups:**
```
[search]  → fastembed, numpy, sqlite-vec
[ai]      → anthropic
[mcp]     → mcp
[dev]     → pytest, pytest-cov, pytest-asyncio, ruff, mypy
```

Remove `httpx` from core (only needed if we add HTTP transport later).

---

## Alternative Approaches Considered

### 1. Rewrite in Rust/Go for Single-Binary Distribution

**Rejected because:** The Python ecosystem (fastembed, tree-sitter, mcp SDK, tiktoken) is mature and well-maintained. A Rust rewrite would take 3-6 months and lose access to these libraries. The daemon architecture eliminates the Python startup cost concern (daemon stays warm).

### 2. Use an External Vector DB (Qdrant, ChromaDB)

**Rejected because:** sqlite-vec is zero-config, single-file, and fast enough for project-scale data (< 100K chunks). An external DB adds deployment complexity that kills adoption. The entire value prop is local-first, zero-config.

### 3. Abandon Daemon, Use Pre-Built Static Files Only

**Rejected because:** Static files go stale immediately. A daemon with file watching keeps indexes fresh in real-time. The daemon also enables the MCP server to be a thin proxy (truly async, no blocking I/O).

### 4. Use Language Server Protocol (LSP) Instead of Custom Daemon

**Considered but deferred:** LSP is well-established but designed for editor interactions (completions, hover, diagnostics), not agent context queries. A custom daemon with JSON-RPC gives us the exact API we need. Could adopt LSP later if editors want integration.

---

## Acceptance Criteria

### Functional Requirements

- [x] `know context "query" --budget 4000` returns relevant code chunks in <200ms (daemon warm)
- [x] `know search "query"` returns results in <100ms via FTS5/BM25
- [x] `know remember` / `know recall` works across sessions with semantic deduplication
- [x] `know next-file`, `know signatures`, `know related` provide iterative context
- [x] `know generate-context` produces a static `.know/CONTEXT.md`
- [x] Tree-sitter chunking works for Python, TypeScript, Go, Rust, Java, Ruby, C, C++
- [x] Import graph uses fully-qualified names, no false positives from leaf-name matching
- [x] Token counting uses tiktoken with provider-aware calibration
- [x] Daemon auto-starts on first CLI call, auto-shuts down after 30 min idle
- [ ] MCP server proxies to daemon (truly async, no event loop blocking)
- [ ] Works without `know init` (zero-config auto-initialization)
- [ ] CI/CD fallback mode (synchronous, no daemon, no embeddings)

### Non-Functional Requirements

- [ ] Daemon warm query latency: p50 <50ms, p99 <200ms
- [ ] Cold start (no daemon, sync mode): <5s for 1000-file project
- [ ] Memory footprint: <200MB per daemon (without embeddings), <400MB with
- [ ] Initial indexing: <30s for 1000-file project
- [ ] Incremental re-index on file change: <500ms
- [ ] Zero mandatory network calls (all features work offline)
- [ ] Install to first value: <30 seconds (`pip install know-cli && know context "query"`)

### Quality Gates

- [x] All existing tests pass (backward compatibility for core features)
- [x] New test suite covers all 8 languages for chunking
- [ ] Performance benchmark tests with <100ms assertions
- [x] No `except Exception: pass` in new code
- [ ] `ruff` and `mypy --strict` pass
- [ ] Database migration tested: v0.3 -> v2.0 preserves all memories

---

## Success Metrics

| Metric | v0.3 (Current) | v2.0 Target |
|--------|----------------|-------------|
| `know context` latency (warm) | 5-30 seconds | <200ms |
| `know search` latency | 3-10 seconds | <100ms |
| Token count accuracy | ~60-80% (heuristic) | >95% (tiktoken) |
| Languages supported | 1 (Python) | 8 |
| Import graph accuracy | ~50% (leaf-name) | >95% (qualified) |
| Setup friction | `know init` required | Zero-config |
| Embedding model loads | 3 per process | 1 (singleton) |
| Memory management | Basic store/recall | Dedup + staleness + quality |

---

## Dependencies & Prerequisites

- `sqlite-vec` Python package (for optional vector search)
- Tree-sitter grammar packages for 8 languages
- `tiktoken` package (for token counting)
- Unix domain socket support (Linux, macOS — Windows uses named pipes)

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Daemon stability issues | High | High | Synchronous fallback always available. Crash recovery via PID detection. SQLite WAL mode prevents corruption. |
| Tree-sitter grammar breaking changes | Low | Medium | Pin grammar versions. Test against each language grammar in CI. |
| sqlite-vec not available on all platforms | Medium | Low | Vector search is optional. FTS5/BM25 is the default. |
| Unix sockets not available (Windows) | Medium | Medium | Fall back to TCP localhost or named pipes on Windows. |
| Large codebase inotify limits | Medium | Medium | Detect limit and fall back to polling. Warn user to increase `fs.inotify.max_user_watches`. |
| Migration breaks existing memories | Low | High | Backup before migration. Auto-rollback on failure. Test migration in CI. |

---

## Implementation Phases Summary

| Phase | Duration | Key Deliverable | Dependencies |
|-------|----------|----------------|--------------|
| 1. Fix Core | Week 1-2 | Correct token counting, import graph, embedding singleton, dead code removal | None |
| 2. Tree-sitter | Week 2-3 | Multi-language chunking, context engine uses Tree-sitter | Phase 1 (embeddings) |
| 3. Daemon | Week 3-5 | Background daemon, Unix socket IPC, thin CLI client, async MCP | Phase 2 (chunking) |
| 4. Memory | Week 4-5 | Hybrid retrieval, deduplication, quality scoring, migrations | Phase 3 (daemon DB) |
| 5. New APIs | Week 5-6 | `next-file`, `signatures`, `related`, `generate-context`, zero-config | Phase 3 (daemon) |
| 6. Polish | Week 6-7 | Binary quantization, error handling, test suite, benchmarks | All phases |

---

## Future Considerations

- **Team knowledge sharing** — Sync memories across team members via git-tracked `.know/shared-memories.json`
- **LSP integration** — Expose daemon as an LSP server for editor integration
- **Remote daemon** — Run daemon on a powerful machine, connect from thin laptops
- **Custom embedding models** — Allow users to bring their own embedding model
- **Streaming responses** — For very large context bundles, stream chunks as they're found

---

## References & Research

### Internal References
- Audit findings: CLI Agent UX (Agent 1) and Technical Architecture (Agent 2)
- Current architecture: `src/know/context_engine.py` (main pipeline)
- Current MCP: `src/know/mcp_server.py` (FastMCP implementation)
- Current parsers: `src/know/parsers.py` (ParserFactory with strategy pattern)

### External References
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [rust-analyzer: Three Architectures for a Responsive IDE](https://rust-analyzer.github.io//blog/2020/07/20/three-architectures-for-responsive-ide.html)
- [Augment Code: Quantized Vector Search](https://www.augmentcode.com/blog/repo-scale-100M-line-codebase-quantized-vector-search)
- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [cAST: Structural Chunking via AST](https://arxiv.org/html/2506.15655v1)
- [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service)
- [Evil Martians: 6 Things Developer Tools Must Have](https://evilmartians.com/chronicles/six-things-developer-tools-must-have-to-earn-trust-and-adoption)
- [Reverse Engineering Claude's Token Counter](https://grohan.co/2026/02/10/ctoc/)
