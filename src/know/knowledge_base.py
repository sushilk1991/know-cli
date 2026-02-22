"""Knowledge base: cross-session memory for AI agents.

Thin wrapper around DaemonDB — the single source of truth for all
project data. Translates between integer display IDs (used by CLI)
and text UUIDs (used by DaemonDB).
"""

from __future__ import annotations

import json
import re
import time
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
    memory_type: str = "note"  # note, decision, constraint, fact, todo, risk
    decision_status: str = "active"  # active, resolved, superseded, rejected
    confidence: float = 0.5
    evidence: str = ""
    session_id: str = ""
    agent: str = ""
    trust_level: str = "local_verified"  # local_verified, imported_unverified, blocked
    supersedes_id: str = ""
    tags: str = ""
    created_at: str = ""
    resolved_at: str = ""
    expires_at: str = ""
    project_root: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "memory_type": self.memory_type,
            "decision_status": self.decision_status,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "session_id": self.session_id,
            "agent": self.agent,
            "trust_level": self.trust_level,
            "supersedes_id": self.supersedes_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "expires_at": self.expires_at,
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
        memory_type: str = "note",
        decision_status: str = "active",
        confidence: float = 0.5,
        evidence: str = "",
        session_id: str = "",
        agent: str = "",
        trust_level: str = "local_verified",
        supersedes_id: str = "",
        expires_at: Optional[float] = None,
    ) -> int:
        """Store a new memory. Returns the integer display ID."""
        trust_level = self._normalize_trust_level(text, trust_level)
        embedding_blob = self._embed_text(text)
        text_id = str(uuid.uuid4())[:8]

        # Map source names: DaemonDB uses source_type
        stored = self._db.store_memory(
            text_id, text, tags=tags, source_type=source,
            embedding=embedding_blob,
            memory_type=memory_type,
            decision_status=decision_status,
            confidence=confidence,
            evidence=evidence,
            session_id=session_id,
            agent=agent,
            trust_level=trust_level,
            supersedes_id=supersedes_id,
            expires_at=expires_at,
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

    def list_all(
        self,
        source: Optional[str] = None,
        memory_type: Optional[str] = None,
        decision_status: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Memory]:
        """List memories with optional structured filters."""
        rows = self._db.list_memories(
            source=source,
            memory_type=memory_type,
            decision_status=decision_status,
            session_id=session_id,
        )
        result = []
        for row in rows:
            text_id = row["id"]
            int_id = self._reverse_map.get(text_id)
            if int_id is None:
                int_id = self._assign_id(text_id)
            result.append(self._dict_to_memory(row, int_id))
        return result

    def count(
        self,
        source: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> int:
        """Count memories."""
        return self._db.count_memories(source=source, memory_type=memory_type)

    def resolve(self, memory_id: int, status: str = "resolved") -> bool:
        """Resolve/supersede/reject a memory by display ID."""
        text_id = self._id_map.get(memory_id)
        if not text_id:
            return False
        resolved_at = time.time() if status in {"resolved", "superseded", "rejected"} else None
        return self._db.update_memory(
            text_id,
            decision_status=status,
            resolved_at=resolved_at,
        )

    # ------------------------------------------------------------------
    # Semantic recall
    # ------------------------------------------------------------------
    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        decision_status: Optional[str] = None,
        include_blocked: bool = False,
        include_expired: bool = False,
    ) -> List[Memory]:
        """Recall memories using hybrid retrieval (semantic + lexical + text + priors)."""
        try:
            return self._recall_hybrid(
                query,
                top_k=top_k,
                memory_type=memory_type,
                decision_status=decision_status,
                include_blocked=include_blocked,
                include_expired=include_expired,
            )
        except Exception as e:
            logger.debug(f"Hybrid memory recall failed ({e}), using lexical fallback")
            rows = self._db.recall_memories(query, limit=max(top_k, 10))
            rows = self._filter_rows(
                rows,
                memory_type=memory_type,
                decision_status=decision_status,
                include_blocked=include_blocked,
                include_expired=include_expired,
            )
            memories = self._rows_to_memories(rows[:top_k])
            self._touch_memories(memories)
            return memories

    def _recall_hybrid(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[str],
        decision_status: Optional[str],
        include_blocked: bool,
        include_expired: bool,
    ) -> List[Memory]:
        from know.ranking import fuse_rankings

        lane_limit = max(20, top_k * 6)

        lexical_rows = self._db.recall_memories(query, limit=lane_limit)

        semantic_rows: List[Dict[str, Any]] = []
        query_vec = self._embed_text(query)
        if query_vec is not None:
            semantic_rows = self._db.recall_memories_semantic(query_vec, limit=lane_limit)

        text_lane = self._recall_text_with_scores(query, lane_limit)

        candidate_rows: Dict[str, Dict[str, Any]] = {}
        for row in lexical_rows + semantic_rows:
            rid = row.get("id")
            if rid:
                candidate_rows[rid] = row
        for text_id, _score in text_lane:
            if text_id not in candidate_rows:
                row = self._db.get_memory_by_id(text_id)
                if row:
                    candidate_rows[text_id] = row

        if not candidate_rows:
            return []

        ranked_lists: List[List[tuple[str, float]]] = []
        if lexical_rows:
            lex_rank = [(row["id"], float(-i)) for i, row in enumerate(lexical_rows)]
            for _ in range(3):
                ranked_lists.append(lex_rank)
        if semantic_rows:
            sem_rank = [(row["id"], float(-i)) for i, row in enumerate(semantic_rows)]
            for _ in range(2):
                ranked_lists.append(sem_rank)
        if text_lane:
            text_rank = [(rid, score) for rid, score in text_lane]
            ranked_lists.append(text_rank)

        # Prior lane: recency + quality + access_count + confidence.
        prior = sorted(
            candidate_rows.values(),
            key=self._prior_utility,
            reverse=True,
        )
        if prior:
            ranked_lists.append([(row["id"], float(-i)) for i, row in enumerate(prior)])

        fused = fuse_rankings(ranked_lists) if ranked_lists else []
        if not fused:
            fused = [(rid, 0.0) for rid in candidate_rows.keys()]

        fused_rows = []
        for rid, _score in fused:
            row = candidate_rows.get(rid)
            if row is not None:
                fused_rows.append(row)
        fused_rows = self._filter_rows(
            fused_rows,
            memory_type=memory_type,
            decision_status=decision_status,
            include_blocked=include_blocked,
            include_expired=include_expired,
        )

        memories = self._rows_to_memories(fused_rows[:top_k])
        self._touch_memories(memories)
        return memories

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

    def _recall_text_with_scores(self, query: str, top_k: int) -> List[tuple[str, float]]:
        """Return (text_id, score) tuples from word-overlap fallback lane."""
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return []

        scored: list[tuple[str, float]] = []
        for mem in self.list_all():
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
            score = overlap / max(1, len(query_words))
            if score <= 0:
                continue
            text_id = self._id_map.get(mem.id)
            if text_id:
                scored.append((text_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _prior_utility(row: Dict[str, Any]) -> float:
        now = time.time()
        created_at = float(row.get("created_at", now) or now)
        age_days = max(0.0, (now - created_at) / 86400.0)
        recency = 1.0 / (1.0 + age_days / 30.0)
        quality = float(row.get("quality_score", 1.0) or 1.0)
        confidence = float(row.get("confidence", 0.5) or 0.5)
        access_count = float(row.get("access_count", 0) or 0)
        access = min(access_count, 20.0) / 20.0
        return 0.45 * recency + 0.25 * quality + 0.20 * confidence + 0.10 * access

    def _filter_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        memory_type: Optional[str],
        decision_status: Optional[str],
        include_blocked: bool,
        include_expired: bool,
    ) -> List[Dict[str, Any]]:
        out = []
        now = time.time()
        for row in rows:
            if memory_type and row.get("memory_type", "note") != memory_type:
                continue
            if decision_status and row.get("decision_status", "active") != decision_status:
                continue
            if not include_blocked and row.get("trust_level", "local_verified") == "blocked":
                continue
            expires_at = row.get("expires_at")
            if not include_expired and expires_at is not None:
                try:
                    if float(expires_at) <= now:
                        continue
                except Exception:
                    pass
            out.append(row)
        return out

    def _touch_memories(self, memories: List[Memory]) -> None:
        text_ids = []
        for mem in memories:
            tid = self._id_map.get(mem.id)
            if tid:
                text_ids.append(tid)
        if text_ids:
            try:
                self._db.touch_memories(text_ids)
            except Exception as e:
                logger.debug(f"touch_memories failed: {e}")

    @staticmethod
    def _normalize_trust_level(text: str, requested: str) -> str:
        level = (requested or "local_verified").strip()
        if level != "local_verified":
            return level

        lowered = (text or "").lower()
        suspicious_markers = (
            "ignore previous instructions",
            "ignore all previous",
            "system prompt",
            "leak secret",
            "exfiltrate",
            "bypass safeguards",
        )
        if any(marker in lowered for marker in suspicious_markers):
            return "blocked"
        return level

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
                    memory_type=rec.get("memory_type", "note"),
                    decision_status=rec.get("decision_status", "active"),
                    confidence=float(rec.get("confidence", 0.5) or 0.5),
                    evidence=rec.get("evidence", ""),
                    session_id=rec.get("session_id", ""),
                    agent=rec.get("agent", ""),
                    trust_level=rec.get("trust_level", "imported_unverified"),
                    supersedes_id=rec.get("supersedes_id", ""),
                    expires_at=rec.get("expires_at"),
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
    @staticmethod
    def _fmt_ts(value: Any) -> str:
        if value in (None, "", 0):
            return ""
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value).isoformat()
            except Exception:
                return ""
        return str(value)

    def _dict_to_memory(self, row: Dict[str, Any], int_id: int) -> Memory:
        """Convert a DaemonDB dict to a Memory dataclass."""
        created = self._fmt_ts(row.get("created_at", ""))
        resolved = self._fmt_ts(row.get("resolved_at", ""))
        expires = self._fmt_ts(row.get("expires_at", ""))
        return Memory(
            id=int_id,
            text=row.get("content", ""),
            source=row.get("source_type", "manual"),
            memory_type=row.get("memory_type", "note"),
            decision_status=row.get("decision_status", "active"),
            confidence=float(row.get("confidence", 0.5) or 0.5),
            evidence=row.get("evidence", ""),
            session_id=row.get("session_id", ""),
            agent=row.get("agent", ""),
            trust_level=row.get("trust_level", "local_verified"),
            supersedes_id=row.get("supersedes_id", ""),
            tags=row.get("tags", ""),
            created_at=str(created),
            resolved_at=str(resolved),
            expires_at=str(expires),
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
