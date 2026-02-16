"""Unified daemon database — SQLite with FTS5 for search.

Stores code chunks, import graph, and cross-session memories in a
single database. FTS5 provides BM25 keyword search as the default;
vector embeddings are optional via know-cli[search].
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xxhash

from know.logger import get_logger
from know.token_counter import count_tokens

logger = get_logger()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
-- Code chunks (Tree-sitter extracted)
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    chunk_name TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    language TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT DEFAULT '',
    body TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(body_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_path_name ON chunks(file_path, chunk_name, start_line);

-- FTS5 index on chunk content (BM25F search with field weighting)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_name, file_path, signature, body,
    content='chunks', content_rowid='id',
    prefix='2,3'
);

-- FTS5 vocabulary table for zero-result diagnostics
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab(chunks_fts, row);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_name, file_path, signature, body)
    VALUES (new.id, new.chunk_name, new.file_path, new.signature, new.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_name, file_path, signature, body)
    VALUES('delete', old.id, old.chunk_name, old.file_path, old.signature, old.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_name, file_path, signature, body)
    VALUES('delete', old.id, old.chunk_name, old.file_path, old.signature, old.body);
    INSERT INTO chunks_fts(rowid, chunk_name, file_path, signature, body)
    VALUES (new.id, new.chunk_name, new.file_path, new.signature, new.body);
END;

-- Import graph (fully-qualified edges)
CREATE TABLE IF NOT EXISTS imports (
    source_module TEXT NOT NULL,
    target_module TEXT NOT NULL,
    import_type TEXT NOT NULL DEFAULT 'import',
    PRIMARY KEY (source_module, target_module)
);

CREATE INDEX IF NOT EXISTS idx_imports_source ON imports(source_module);
CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_module);

-- Cross-session memories
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    source_type TEXT DEFAULT 'manual',
    quality_score REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed_at REAL,
    content_hash TEXT NOT NULL,
    embedding BLOB DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);

-- FTS5 on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
    VALUES('delete', old.rowid, old.content, old.tags);
END;

-- File tracking for incremental indexing
CREATE TABLE IF NOT EXISTS file_index (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    indexed_at REAL NOT NULL
);

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

-- Module importance scores (in-degree / PageRank)
CREATE TABLE IF NOT EXISTS module_importance (
    module_name TEXT PRIMARY KEY,
    in_degree INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    computed_at REAL NOT NULL
);
"""


MAX_SEARCH_TERMS = 12


class _BatchContext:
    """Defers commits until the outermost batch exits, then commits once.

    Supports nesting: inner batches are no-ops.  Thread-safe: batch depth
    is tracked per-thread via ``threading.local()``.
    """

    def __init__(self, db: "DaemonDB"):
        self._db = db

    def __enter__(self):
        local = self._db._local
        depth = getattr(local, "batch_depth", 0)
        local.batch_depth = depth + 1
        return self._db

    def __exit__(self, exc_type, exc_val, exc_tb):
        local = self._db._local
        local.batch_depth -= 1
        if local.batch_depth == 0:
            conn = self._db._get_conn()
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        return False  # don't suppress exceptions


class DaemonDB:
    """Unified project database with FTS5 search."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.db_path = project_root / ".know" / "daemon.db"
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            with self._lock:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _commit(self, conn: sqlite3.Connection) -> None:
        """Commit unless inside a batch() context (thread-safe)."""
        if getattr(self._local, "batch_depth", 0) == 0:
            conn.commit()

    def batch(self):
        """Context manager for batched writes — single commit at the end.

        Wraps all writes in one transaction, replacing per-operation commits
        with a single commit on exit.  Provides ~3-5x speedup for bulk
        indexing by eliminating per-file fsync overhead.

        Usage::

            with db.batch():
                for file in files:
                    db.upsert_chunks(...)
                    db.update_file_index(...)
            # single commit happens here
        """
        return _BatchContext(self)

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Build OR-based FTS5 query from natural language string.

        Uses query understanding to strip stop words, detect identifiers,
        and build a smarter FTS5 query.  Falls back to raw split if the
        query module is unavailable.
        """
        try:
            from know.query import analyze_query, build_fts_or_query
            plan = analyze_query(query)
            return build_fts_or_query(plan.all_search_terms)
        except Exception:
            # Fallback: raw whitespace split (pre-v3 behavior)
            terms = query.strip().split()[:MAX_SEARCH_TERMS]
            if not terms:
                return ""
            return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)

    def _init_db(self):
        conn = self._get_conn()
        # Migrate FTS5 schema BEFORE running _SCHEMA (which has CREATE IF NOT EXISTS)
        self._migrate_fts_schema(conn)
        conn.executescript(_SCHEMA)
        conn.commit()
        self._migrate(conn)
        # Cache FTS column count (schema doesn't change after init)
        self._fts_cols = self._fts_column_count(conn)

    def _needs_fts_migration(self, conn: sqlite3.Connection) -> bool:
        """Check if FTS5 table needs migration to add file_path column.

        CRITICAL: PRAGMA table_info FAILS on FTS5 virtual tables (returns empty).
        Use sqlite_master to inspect the CREATE statement instead.
        """
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if row is None:
            return False  # Table doesn't exist yet; _SCHEMA will create it
        return 'file_path' not in row[0]

    def _migrate_fts_schema(self, conn: sqlite3.Connection):
        """Migrate FTS5 from 3 columns to 4 columns (add file_path).

        CRITICAL: Triggers MUST be dropped FIRST — they reference old column
        positions. Without this, file_path values go into signature column
        and signature into body (silent data corruption).
        """
        if not self._needs_fts_migration(conn):
            return

        logger.info("Migrating FTS5 schema: adding file_path column")

        # Drop triggers FIRST (they reference old column positions)
        conn.execute("DROP TRIGGER IF EXISTS chunks_ai")
        conn.execute("DROP TRIGGER IF EXISTS chunks_ad")
        conn.execute("DROP TRIGGER IF EXISTS chunks_au")

        # Drop old FTS5 virtual table and vocab table
        conn.execute("DROP TABLE IF EXISTS chunks_fts_vocab")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.commit()

        # NOTE: Do NOT use executescript() — it implicitly COMMITs,
        # breaking transaction control. _SCHEMA's CREATE IF NOT EXISTS
        # will create the new table with 4 columns + new triggers.

    def _migrate(self, conn: sqlite3.Connection):
        """Run schema migrations for backwards compatibility."""
        # Add embedding column if missing (pre-v2 databases)
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        if "embedding" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB DEFAULT NULL")
            conn.commit()

        # Rebuild FTS5 index if it was just migrated (has 0 rows but chunks exist)
        try:
            fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if fts_count == 0 and chunk_count > 0:
                logger.info(f"Rebuilding FTS5 index ({chunk_count} chunks)")
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                conn.commit()
        except sqlite3.OperationalError:
            pass  # FTS table may not exist yet on first run

        # Track schema version
        try:
            row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            current = row[0] if row else 0
            if current < 3:
                # v3: full source bodies stored — force reindex by clearing all indexed data
                try:
                    conn.execute("DELETE FROM file_index")
                    conn.execute("DELETE FROM chunks")
                    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                except sqlite3.OperationalError:
                    pass
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (3, time.time()),
                )
                conn.commit()
        except sqlite3.OperationalError:
            pass  # Table may not exist yet

    def close(self):
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------
    def upsert_chunks(self, file_path: str, language: str, chunks: List[Dict[str, Any]]):
        """Replace all chunks for a file with new ones."""
        conn = self._get_conn()
        now = time.time()

        # Delete old chunks for this file
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

        for chunk in chunks:
            body = chunk.get("body", "")
            body_hash = xxhash.xxh64(body.encode()).hexdigest()
            token_count = count_tokens(body)

            conn.execute(
                """INSERT INTO chunks
                   (file_path, chunk_name, chunk_type, language,
                    start_line, end_line, signature, body,
                    body_hash, token_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_path,
                    chunk.get("name", ""),
                    chunk.get("type", "module"),
                    language,
                    chunk.get("start_line", 0),
                    chunk.get("end_line", 0),
                    chunk.get("signature", ""),
                    body,
                    body_hash,
                    token_count,
                    now,
                ),
            )
        self._commit(conn)

    def _fts_column_count(self, conn: sqlite3.Connection) -> int:
        """Detect number of columns in FTS5 table for weight vector selection."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        if row is None:
            return 0
        # Count columns in CREATE VIRTUAL TABLE statement
        sql = row[0]
        # Columns are between the first '(' and the first keyword like 'content='
        if 'file_path' in sql:
            return 4  # chunk_name, file_path, signature, body
        return 3  # chunk_name, signature, body (old schema)

    def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Dual-lane search: OR + AND + exact-name match, fused via RRF.

        Lane 1: FTS5 BM25F with OR query (broad recall)
        Lane 2: FTS5 BM25F with AND query (high precision — all terms must match)
        Lane 3: LIKE match on chunk_name for detected identifiers (exact match)

        Results fused via Reciprocal Rank Fusion with lane-specific weights:
        AND results get 2x weight, exact name matches get 3x.
        """
        from know.query import analyze_query, build_fts_or_query, build_fts_and_query
        from know.ranking import fuse_rankings

        plan = analyze_query(query)
        conn = self._get_conn()
        col_count = getattr(self, '_fts_cols', None) or self._fts_column_count(conn)

        lanes: List[List[Dict[str, Any]]] = []

        # Lane 1: OR query (broad recall)
        or_query = build_fts_or_query(plan.all_search_terms)
        if or_query:
            lane1 = self._fts_search(conn, or_query, col_count, limit * 3)
            if lane1:
                lanes.append(lane1)

        # Lane 2: AND query (high precision)
        and_query = build_fts_and_query(plan.all_search_terms)
        if and_query:
            lane2 = self._fts_search(conn, and_query, col_count, limit * 2)
            if lane2:
                lanes.append(lane2)

        # Lane 3: Exact name match for identifiers
        lane3: List[Dict[str, Any]] = []
        for ident in plan.identifiers:
            try:
                rows = conn.execute(
                    """SELECT c.*, 100.0 AS score FROM chunks c
                       WHERE c.chunk_name LIKE ? LIMIT ?""",
                    (f"%{ident}%", limit),
                ).fetchall()
                lane3.extend(dict(r) for r in rows)
            except sqlite3.OperationalError:
                pass

        # Also match file_path components for identifiers
        for ident in plan.identifiers:
            try:
                rows = conn.execute(
                    """SELECT c.*, 80.0 AS score FROM chunks c
                       WHERE c.file_path LIKE ? LIMIT ?""",
                    (f"%{ident}%", limit),
                ).fetchall()
                lane3.extend(dict(r) for r in rows)
            except sqlite3.OperationalError:
                pass

        if lane3:
            lanes.append(lane3)

        if not lanes:
            return []

        # If only one lane returned results, skip fusion
        if len(lanes) == 1:
            return lanes[0][:limit]

        # RRF fusion with lane-specific weights
        # Convert to (chunk_key, score) format for fuse_rankings
        ranked_lists = []
        for i, lane in enumerate(lanes):
            # Lane weights: lane1 (OR)=1x, lane2 (AND)=2x, lane3 (exact)=3x
            weight = [1, 2, 3][min(i, 2)]
            keyed = []
            for chunk in lane:
                key = f"{chunk['file_path']}:{chunk['chunk_name']}:{chunk.get('start_line', 0)}"
                keyed.append((key, chunk.get("score", 0)))
            # Repeat the lane to give it more weight in RRF
            for _ in range(weight):
                ranked_lists.append(keyed)

        fused = fuse_rankings(ranked_lists)

        # Map back to full chunk dicts
        chunk_map: Dict[str, Dict] = {}
        for lane in lanes:
            for chunk in lane:
                key = f"{chunk['file_path']}:{chunk['chunk_name']}:{chunk.get('start_line', 0)}"
                if key not in chunk_map:
                    chunk_map[key] = chunk

        results = []
        for key, fused_score in fused[:limit]:
            if key in chunk_map:
                chunk = chunk_map[key]
                chunk["score"] = fused_score
                results.append(chunk)

        return results

    def _fts_search(
        self, conn: sqlite3.Connection, fts_query: str,
        col_count: int, limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute a single FTS5 search query and return results."""
        try:
            if col_count == 4:
                rows = conn.execute(
                    """SELECT c.*, -bm25(chunks_fts, 5.0, 5.0, 3.0, 1.0) AS score
                       FROM chunks_fts
                       JOIN chunks c ON chunks_fts.rowid = c.id
                       WHERE chunks_fts MATCH ?
                       ORDER BY score DESC
                       LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT c.*, rank AS score
                       FROM chunks_fts
                       JOIN chunks c ON chunks_fts.rowid = c.id
                       WHERE chunks_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, limit),
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def get_chunks_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks for a file."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE file_path = ? ORDER BY start_line",
            (file_path,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_signatures(self, file_path: Optional[str] = None) -> List[Dict[str, str]]:
        """Get function/class signatures, optionally filtered by file."""
        conn = self._get_conn()
        if file_path:
            rows = conn.execute(
                "SELECT file_path, chunk_name, chunk_type, signature, start_line "
                "FROM chunks WHERE file_path = ? ORDER BY start_line",
                (file_path,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT file_path, chunk_name, chunk_type, signature, start_line "
                "FROM chunks ORDER BY file_path, start_line"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # File index (change detection)
    # ------------------------------------------------------------------
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get stored hash for a file."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content_hash FROM file_index WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row["content_hash"] if row else None

    def update_file_index(self, file_path: str, content_hash: str,
                          language: str, chunk_count: int):
        """Update file index entry."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO file_index
               (file_path, content_hash, language, chunk_count, indexed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (file_path, content_hash, language, chunk_count, time.time()),
        )
        self._commit(conn)

    def remove_file(self, file_path: str):
        """Remove a file and its chunks from the index."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM file_index WHERE file_path = ?", (file_path,))
        self._commit(conn)

    # ------------------------------------------------------------------
    # Import graph
    # ------------------------------------------------------------------
    def set_imports(self, source: str, targets: List[Tuple[str, str]]):
        """Set import edges for a module (replaces existing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM imports WHERE source_module = ?", (source,))
        if targets:
            conn.executemany(
                "INSERT OR REPLACE INTO imports (source_module, target_module, import_type) "
                "VALUES (?, ?, ?)",
                [(source, t, itype) for t, itype in targets],
            )
        self._commit(conn)

    def get_imports_of(self, module: str) -> List[str]:
        """Get modules imported by the given module."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_module FROM imports WHERE source_module = ?",
            (module,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_imported_by(self, module: str) -> List[str]:
        """Get modules that import the given module."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_module FROM imports WHERE target_module = ?",
            (module,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Return all (source, target) import edges."""
        conn = self._get_conn()
        rows = conn.execute("SELECT source_module, target_module FROM imports").fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_imports_batch(self, modules: List[str]) -> Dict[str, List[str]]:
        """Get imports for multiple modules in one query."""
        if not modules:
            return {}
        conn = self._get_conn()
        placeholders = ','.join('?' * len(modules))
        rows = conn.execute(
            f"SELECT source_module, target_module FROM imports WHERE source_module IN ({placeholders})",
            modules,
        ).fetchall()
        result: Dict[str, List[str]] = {}
        for s, t in rows:
            result.setdefault(s, []).append(t)
        return result

    def get_imported_by_batch(self, modules: List[str]) -> Dict[str, List[str]]:
        """Get reverse imports for multiple modules in one query."""
        if not modules:
            return {}
        conn = self._get_conn()
        placeholders = ','.join('?' * len(modules))
        rows = conn.execute(
            f"SELECT target_module, source_module FROM imports WHERE target_module IN ({placeholders})",
            modules,
        ).fetchall()
        result: Dict[str, List[str]] = {}
        for t, s in rows:
            result.setdefault(t, []).append(s)
        return result

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------
    def store_memory(self, memory_id: str, content: str,
                     tags: str = "[]", source_type: str = "manual",
                     embedding: Optional[bytes] = None) -> bool:
        """Store a memory. Returns False if duplicate content exists."""
        content_hash = xxhash.xxh64(content.encode()).hexdigest()

        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            return False

        conn.execute(
            """INSERT INTO memories (id, content, tags, source_type,
                                     created_at, content_hash, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, content, tags, source_type, time.time(), content_hash, embedding),
        )
        self._commit(conn)
        return True

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by its ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._commit(conn)
        return cursor.rowcount > 0

    def list_memories(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all memories, optionally filtered by source_type."""
        conn = self._get_conn()
        if source:
            rows = conn.execute(
                "SELECT * FROM memories WHERE source_type = ? ORDER BY created_at DESC",
                (source,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_memories(self, source: Optional[str] = None) -> int:
        """Count memories, optionally filtered by source_type."""
        conn = self._get_conn()
        if source:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE source_type = ?", (source,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def recall_memories_semantic(self, query_embedding: bytes, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories by cosine similarity against stored embeddings."""
        import numpy as np

        query_arr = np.frombuffer(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_arr)
        if q_norm == 0:
            return []
        query_arr = query_arr / q_norm

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            e_norm = np.linalg.norm(emb)
            if e_norm == 0:
                continue
            sim = float(np.dot(query_arr, emb / e_norm))
            scored.append((sim, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def recall_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using FTS5 BM25."""
        conn = self._get_conn()
        safe_query = self._build_fts_query(query)
        if not safe_query:
            return []
        try:
            rows = conn.execute(
                """SELECT m.*, rank
                   FROM memories_fts
                   JOIN memories m ON memories_fts.rowid = m.rowid
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Module importance (in-degree scoring)
    # ------------------------------------------------------------------
    def compute_importance(self) -> Dict[str, float]:
        """Compute in-degree importance scores for all modules.

        In-degree = number of modules that import this module.
        Normalized to 0-1 range. Stored in module_importance table.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT target_module, COUNT(*) as in_degree "
            "FROM imports GROUP BY target_module"
        ).fetchall()
        if not rows:
            return {}
        max_deg = max(r[1] for r in rows)
        if max_deg == 0:
            return {}
        raw_degrees = {r[0]: r[1] for r in rows}
        scores = {name: deg / max_deg for name, deg in raw_degrees.items()}
        now = time.time()
        # Store in DB
        conn.execute("DELETE FROM module_importance")
        conn.executemany(
            "INSERT INTO module_importance (module_name, in_degree, score, computed_at) "
            "VALUES (?, ?, ?, ?)",
            [(name, raw_degrees[name], score, now) for name, score in scores.items()],
        )
        self._commit(conn)
        return scores

    def get_importance(self, module_name: str) -> float:
        """Get cached importance score for a module (0-1)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT score FROM module_importance WHERE module_name = ?",
            (module_name,),
        ).fetchone()
        return row[0] if row else 0.0

    def get_importance_batch(self, modules: List[str]) -> Dict[str, float]:
        """Get importance scores for multiple modules."""
        if not modules:
            return {}
        conn = self._get_conn()
        placeholders = ','.join('?' * len(modules))
        rows = conn.execute(
            f"SELECT module_name, score FROM module_importance WHERE module_name IN ({placeholders})",
            modules,
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # Zero-result intelligence
    # ------------------------------------------------------------------
    def get_nearest_terms(self, query: str, limit: int = 5) -> List[str]:
        """Find terms in the FTS index closest to query terms via fts5vocab prefix matching."""
        conn = self._get_conn()
        terms = query.strip().split()[:5]
        results = []
        for term in terms:
            prefix = term[:3].lower()
            if len(prefix) < 2:
                continue
            try:
                rows = conn.execute(
                    "SELECT term, doc FROM chunks_fts_vocab "
                    "WHERE term >= ? AND term < ? ORDER BY doc DESC LIMIT ?",
                    (prefix, prefix + '\uffff', limit),
                ).fetchall()
                results.extend(r[0] for r in rows)
            except sqlite3.OperationalError:
                pass  # fts5vocab table may not exist yet
        # Deduplicate preserving order
        return list(dict.fromkeys(results))[:limit]

    def get_matching_file_names(self, query: str, limit: int = 5) -> List[str]:
        """Find file paths that contain query terms."""
        conn = self._get_conn()
        terms = query.strip().split()[:5]
        if not terms:
            return []
        conditions = " OR ".join(["file_path LIKE ?"] * len(terms))
        like_params = [f"%{t}%" for t in terms]
        rows = conn.execute(
            f"SELECT DISTINCT file_path FROM chunks WHERE {conditions} LIMIT ?",
            (*like_params, limit),
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_conn()
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        files = conn.execute("SELECT COUNT(*) FROM file_index").fetchone()[0]
        memories = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        imports = conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM chunks"
        ).fetchone()[0]

        return {
            "chunks": chunks,
            "files": files,
            "memories": memories,
            "import_edges": imports,
            "total_tokens": total_tokens,
        }
