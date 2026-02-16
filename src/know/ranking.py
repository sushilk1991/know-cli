"""Score fusion for combining multiple ranking signals.

This module contains ONLY pure functions. No DB connections,
no filesystem access, no side effects. Trivially testable.
"""

from typing import Dict, List, Tuple


def fuse_rankings(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion across N ranked lists.

    Each list is [(chunk_id, score), ...] sorted by score descending.
    k=60 is the standard RRF constant from Cormack et al.
    Returns fused [(chunk_id, fused_score)] sorted descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def apply_relevance_floor(
    chunks: List[Dict],
    top_score_ratio: float = 0.3,
) -> List[Dict]:
    """Drop chunks below relevance floor.

    Returns under budget rather than filling with noise.
    Every irrelevant token actively degrades agent reasoning.
    """
    if not chunks:
        return chunks
    top_score = chunks[0].get("score", 1.0)
    if top_score <= 0:
        return chunks
    floor = top_score * top_score_ratio
    return [c for c in chunks if c.get("score", 0) >= floor]
