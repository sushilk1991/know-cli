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

-- FTS5 index on chunk content (BM25 search)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_name, signature, body,
    content='chunks', content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_name, signature, body)
    VALUES (new.id, new.chunk_name, new.signature, new.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_name, signature, body)
    VALUES('delete', old.id, old.chunk_name, old.signature, old.body);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, chunk_name, signature, body)
    VALUES('delete', old.id, old.chunk_name, old.signature, old.body);
    INSERT INTO chunks_fts(rowid, chunk_name, signature, body)
    VALUES (new.id, new.chunk_name, new.signature, new.body);
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
    content_hash TEXT NOT NULL
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
"""


class DaemonDB:
    """Unified project database with FTS5 search."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.db_path = project_root / ".know" / "daemon.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)
                    self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

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
        conn.commit()

    def search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """BM25 full-text search over code chunks."""
        conn = self._get_conn()
        # Escape FTS5 special characters by quoting the query
        safe_query = '"' + query.replace('"', '""') + '"'
        try:
            rows = conn.execute(
                """SELECT c.*, rank
                   FROM chunks_fts
                   JOIN chunks c ON chunks_fts.rowid = c.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

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
        conn.commit()

    def remove_file(self, file_path: str):
        """Remove a file and its chunks from the index."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        conn.execute("DELETE FROM file_index WHERE file_path = ?", (file_path,))
        conn.commit()

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
        conn.commit()

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

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------
    def store_memory(self, memory_id: str, content: str,
                     tags: str = "[]", source_type: str = "manual") -> bool:
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
                                     created_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (memory_id, content, tags, source_type, time.time(), content_hash),
        )
        conn.commit()
        return True

    def recall_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using FTS5 BM25."""
        conn = self._get_conn()
        # Escape FTS5 special characters by quoting the query
        safe_query = '"' + query.replace('"', '""') + '"'
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

        # Batch update access counts
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE memories SET access_count = access_count + 1, "
                f"last_accessed_at = ? WHERE id IN ({placeholders})",
                [time.time(), *ids],
            )
            conn.commit()
        return [dict(r) for r in rows]

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
