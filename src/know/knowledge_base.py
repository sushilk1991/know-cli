"""Knowledge base: cross-session memory for AI agents.

Thin wrapper around DaemonDB — the single source of truth for all
project data. Translates between integer display IDs (used by CLI)
and text UUIDs (used by DaemonDB).
"""

from __future__ import annotations

import json
import re
import uuid

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
    """Project-local knowledge base backed by DaemonDB.

    Maintains an in-memory int→text ID map so CLI users can reference
    memories by short integer IDs (e.g., ``know forget 3``).
    """

    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        from know.daemon_db import DaemonDB
        self._db = DaemonDB(config.root)
        # Build int→text ID map from existing memories
        self._id_map: Dict[int, str] = {}  # int display id → text UUID
        self._reverse_map: Dict[str, int] = {}  # text UUID → int display id
        self._next_id = 1
        self._rebuild_id_map()

    def _rebuild_id_map(self):
        """Build sequential integer IDs from all existing memories."""
        self._id_map.clear()
        self._reverse_map.clear()
        memories = self._db.list_memories()
        # Sort by created_at ascending so oldest gets lowest ID
        memories.sort(key=lambda m: m.get("created_at", 0))
        for i, mem in enumerate(memories, 1):
            text_id = mem["id"]
            self._id_map[i] = text_id
            self._reverse_map[text_id] = i
        self._next_id = len(memories) + 1

    def _assign_id(self, text_id: str) -> int:
        """Assign the next sequential integer ID to a text UUID."""
        int_id = self._next_id
        self._id_map[int_id] = text_id
        self._reverse_map[text_id] = int_id
        self._next_id += 1
        return int_id

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def remember(
        self,
        text: str,
        source: str = "manual",
        tags: str = "",
    ) -> int:
        """Store a new memory. Returns the integer display ID."""
        embedding_blob = self._embed_text(text)
        text_id = str(uuid.uuid4())[:8]

        # Map source names: DaemonDB uses source_type
        stored = self._db.store_memory(
            text_id, text, tags=tags, source_type=source,
            embedding=embedding_blob,
        )
        if not stored:
            # Duplicate content — find existing and return its display ID
            for int_id, tid in self._id_map.items():
                mem = self._db.get_memory_by_id(tid)
                if mem and mem["content"] == text:
                    return int_id
            # Shouldn't reach here, but assign new ID as fallback
            return self._assign_id(text_id)

        return self._assign_id(text_id)

    def forget(self, memory_id: int) -> bool:
        """Delete a memory by integer display ID. Returns True if deleted."""
        text_id = self._id_map.get(memory_id)
        if not text_id:
            return False
        deleted = self._db.delete_memory(text_id)
        if deleted:
            del self._id_map[memory_id]
            del self._reverse_map[text_id]
        return deleted

    def get(self, memory_id: int) -> Optional[Memory]:
        """Get a single memory by integer display ID."""
        text_id = self._id_map.get(memory_id)
        if not text_id:
            return None
        row = self._db.get_memory_by_id(text_id)
        if row is None:
            return None
        return self._dict_to_memory(row, memory_id)

    def list_all(self, source: Optional[str] = None) -> List[Memory]:
        """List all memories, optionally filtered by source."""
        rows = self._db.list_memories(source=source)
        result = []
        for row in rows:
            text_id = row["id"]
            int_id = self._reverse_map.get(text_id)
            if int_id is None:
                int_id = self._assign_id(text_id)
            result.append(self._dict_to_memory(row, int_id))
        return result

    def count(self, source: Optional[str] = None) -> int:
        """Count memories."""
        return self._db.count_memories(source=source)

    # ------------------------------------------------------------------
    # Semantic recall
    # ------------------------------------------------------------------
    def recall(self, query: str, top_k: int = 5) -> List[Memory]:
        """Recall memories semantically similar to *query*.

        Tries embedding-based cosine similarity first; falls back to
        DaemonDB FTS5, then text-based word-overlap scoring.
        """
        try:
            return self._recall_semantic(query, top_k)
        except Exception as e:
            logger.debug(f"Semantic recall unavailable ({e}), using FTS fallback")
            return self._recall_fts(query, top_k)

    def _recall_semantic(self, query: str, top_k: int) -> List[Memory]:
        """Cosine-similarity search over embedding vectors via DaemonDB."""
        query_vec = self._embed_text(query)
        if query_vec is None:
            raise RuntimeError("No embedding model available")

        rows = self._db.recall_memories_semantic(query_vec, limit=top_k)
        if not rows:
            return self._recall_fts(query, top_k)
        return self._rows_to_memories(rows)

    def _recall_fts(self, query: str, top_k: int) -> List[Memory]:
        """FTS5 BM25 search via DaemonDB, with text fallback."""
        rows = self._db.recall_memories(query, limit=top_k)
        if rows:
            return self._rows_to_memories(rows)
        return self._recall_text(query, top_k)

    def _recall_text(self, query: str, top_k: int) -> List[Memory]:
        """Fallback: word-overlap + prefix scoring."""
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
            for qw in query_words:
                if qw in (query_words & mem_words):
                    continue
                for mw in mem_words:
                    if qw.startswith(mw) or mw.startswith(qw):
                        overlap += 0.5
                        break
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
        """Import memories from JSON string. Returns count imported."""
        records = json.loads(data)
        if not records:
            return 0

        count = 0
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
    # Embedding helpers — uses centralized manager
    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> Optional[bytes]:
        """Embed text using centralized embedding manager. Returns raw bytes or None."""
        from know.embeddings import embed_text
        return embed_text(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _dict_to_memory(self, row: Dict[str, Any], int_id: int) -> Memory:
        """Convert a DaemonDB dict to a Memory dataclass."""
        created = row.get("created_at", "")
        # DaemonDB stores created_at as float epoch; convert to string
        if isinstance(created, (int, float)):
            created = datetime.fromtimestamp(created).isoformat()
        return Memory(
            id=int_id,
            text=row.get("content", ""),
            source=row.get("source_type", "manual"),
            tags=row.get("tags", ""),
            created_at=str(created),
            project_root=str(self.root),
        )

    def _rows_to_memories(self, rows: List[Dict[str, Any]]) -> List[Memory]:
        """Convert a list of DaemonDB dicts to Memory objects."""
        result = []
        for row in rows:
            text_id = row["id"]
            int_id = self._reverse_map.get(text_id)
            if int_id is None:
                int_id = self._assign_id(text_id)
            result.append(self._dict_to_memory(row, int_id))
        return result
