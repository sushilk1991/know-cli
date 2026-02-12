"""Knowledge base: cross-session memory for AI agents.

Persists codebase understanding in a project-local SQLite database.
Supports semantic recall (fastembed) with text-match fallback.
Each memory is project-scoped and stored in `.know/knowledge.db`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config

logger = get_logger()


@dataclass
class Memory:
    """A single memory entry."""
    id: int
    text: str
    source: str = "manual"  # manual, auto-explain, auto-digest
    tags: str = ""
    created_at: str = ""
    project_root: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at,
            "project_root": self.project_root,
        }


class KnowledgeBase:
    """Project-local knowledge base backed by SQLite.

    Stores memories with optional embedding vectors for semantic recall.
    Falls back to text-matching when fastembed is unavailable.
    """
    
    # Class-level embedding model cache
    _embedding_model_cache: Dict[str, Any] = {}
    _embedding_lock: Optional[threading.Lock] = None
    _cache_initialized: bool = False

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.db_path = self.root / ".know" / "knowledge.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                tags TEXT DEFAULT '',
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                project_root TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
            CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_root);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def remember(
        self,
        text: str,
        source: str = "manual",
        tags: str = "",
    ) -> int:
        """Store a new memory.  Returns the memory ID."""
        conn = self._get_conn()
        embedding_blob = self._embed_text(text)

        cursor = conn.execute(
            """INSERT INTO memories (text, source, tags, embedding, project_root)
               VALUES (?, ?, ?, ?, ?)""",
            (text, source, tags, embedding_blob, str(self.root)),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def forget(self, memory_id: int) -> bool:
        """Delete a memory by ID.  Returns True if deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM memories WHERE id = ? AND project_root = ?",
            (memory_id, str(self.root)),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a single memory by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, text, source, tags, created_at, project_root "
            "FROM memories WHERE id = ? AND project_root = ?",
            (memory_id, str(self.root)),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def list_all(self, source: Optional[str] = None) -> List[Memory]:
        """List all memories for this project, optionally filtered by source."""
        conn = self._get_conn()
        if source:
            rows = conn.execute(
                "SELECT id, text, source, tags, created_at, project_root "
                "FROM memories WHERE project_root = ? AND source = ? "
                "ORDER BY created_at DESC",
                (str(self.root), source),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, text, source, tags, created_at, project_root "
                "FROM memories WHERE project_root = ? ORDER BY created_at DESC",
                (str(self.root),),
            ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count(self, source: Optional[str] = None) -> int:
        """Count memories for this project."""
        conn = self._get_conn()
        if source:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_root = ? AND source = ?",
                (str(self.root), source),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_root = ?",
                (str(self.root),),
            ).fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Semantic recall
    # ------------------------------------------------------------------
    def recall(self, query: str, top_k: int = 5) -> List[Memory]:
        """Recall memories semantically similar to *query*.

        Tries embedding-based cosine similarity first; falls back to
        text-based word-overlap scoring.
        """
        try:
            return self._recall_semantic(query, top_k)
        except Exception as e:
            logger.debug(f"Semantic recall unavailable ({e}), using text fallback")
            return self._recall_text(query, top_k)

    def _recall_semantic(self, query: str, top_k: int) -> List[Memory]:
        """Cosine-similarity search over embedding vectors."""
        import numpy as np

        query_vec = self._embed_text(query)
        if query_vec is None:
            raise RuntimeError("No embedding model available")

        query_arr = np.frombuffer(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(query_arr)
        if q_norm == 0:
            return []
        query_arr = query_arr / q_norm

        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, text, source, tags, created_at, project_root, embedding "
            "FROM memories WHERE project_root = ? AND embedding IS NOT NULL",
            (str(self.root),),
        ).fetchall()

        if not rows:
            return self._recall_text(query, top_k)

        scored: list[tuple[float, Memory]] = []
        for row in rows:
            emb_blob = row["embedding"]
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            e_norm = np.linalg.norm(emb)
            if e_norm == 0:
                continue
            emb = emb / e_norm
            sim = float(np.dot(query_arr, emb))
            mem = self._row_to_memory(row)
            scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def _recall_text(self, query: str, top_k: int) -> List[Memory]:
        """Fallback: word-overlap scoring."""
        memories = self.list_all()
        if not memories:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return memories[:top_k]

        scored: list[tuple[float, Memory]] = []
        for mem in memories:
            mem_words = set(re.findall(r"\w+", mem.text.lower()))
            if not mem_words:
                continue
            overlap = len(query_words & mem_words)
            score = overlap / len(query_words)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored[:top_k] if s > 0]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------
    def export_json(self) -> str:
        """Export all memories as JSON."""
        memories = self.list_all()
        return json.dumps([m.to_dict() for m in memories], indent=2)

    def import_json(self, data: str) -> int:
        """Import memories from JSON string with batch transaction.  Returns count imported."""
        records = json.loads(data)
        if not records:
            return 0
        
        conn = self._get_conn()
        count = 0
        
        try:
            with conn:
                # Batch insert for performance
                values = []
                for rec in records:
                    text = rec["text"]
                    source = rec.get("source", "manual")
                    tags = rec.get("tags", "")
                    embedding_blob = self._embed_text(text)
                    
                    values.append((
                        text,
                        source,
                        tags,
                        embedding_blob,
                        str(self.root)
                    ))
                
                conn.executemany(
                    """INSERT INTO memories (text, source, tags, embedding, project_root)
                       VALUES (?, ?, ?, ?, ?)""",
                    values
                )
                count = conn.total_changes
        except Exception:
            # Fallback to individual inserts on error
            for rec in records:
                try:
                    self.remember(
                        text=rec["text"],
                        source=rec.get("source", "manual"),
                        tags=rec.get("tags", ""),
                    )
                    count += 1
                except Exception:
                    pass
        
        return count

    # ------------------------------------------------------------------
    # Relevant memories for context engine
    # ------------------------------------------------------------------
    def get_relevant_context(self, query: str, max_tokens: int = 500) -> str:
        """Return relevant memories formatted for context engine injection."""
        from know.token_counter import count_tokens

        memories = self.recall(query, top_k=10)
        if not memories:
            return ""

        lines = ["## Relevant Memories", ""]
        used = count_tokens("\n".join(lines))

        for mem in memories:
            entry = f"- [{mem.source}] {mem.text}"
            entry_tokens = count_tokens(entry)
            if used + entry_tokens > max_tokens:
                break
            lines.append(entry)
            used += entry_tokens

        if len(lines) <= 2:  # only header
            return ""
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    _embedding_model_cache: Dict[str, Any] = {}
    _cache_initialized = False
    
    def _get_embedding_model(self):
        """Get cached embedding model (thread-safe)."""
        model_name = "BAAI/bge-small-en-v1.5"
        
        # Lazy initialization of lock
        if not KnowledgeBase._cache_initialized:
            KnowledgeBase._embedding_lock = threading.Lock()
            KnowledgeBase._cache_initialized = True
        
        # Check cache first
        if model_name in KnowledgeBase._embedding_model_cache:
            return KnowledgeBase._embedding_model_cache[model_name]
        
        # Load and cache
        try:
            from fastembed import TextEmbedding
            
            with KnowledgeBase._embedding_lock:
                # Double-check after acquiring lock
                if model_name in KnowledgeBase._embedding_model_cache:
                    return KnowledgeBase._embedding_model_cache[model_name]
                
                model = TextEmbedding(model_name=model_name)
                KnowledgeBase._embedding_model_cache[model_name] = model
                return model
        except Exception:
            return None
    
    def _embed_text(self, text: str) -> Optional[bytes]:
        """Embed text using fastembed with model caching. Returns raw bytes or None."""
        try:
            import numpy as np
            
            model = self._get_embedding_model()
            if model is None:
                return None
            
            # Truncate to 8000 chars for efficiency
            text = text[:8000]
            emb = np.array(list(model.embed([text]))[0], dtype=np.float32)
            return emb.tobytes()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_memory(row) -> Memory:
        return Memory(
            id=row["id"],
            text=row["text"],
            source=row["source"] or "manual",
            tags=row["tags"] or "",
            created_at=row["created_at"] or "",
            project_root=row["project_root"] or "",
        )
