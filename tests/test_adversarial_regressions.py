"""Regression tests for adversarial boundary and corruption cases."""

from __future__ import annotations

import gc
import hashlib
import sqlite3
import weakref
from contextlib import closing

import numpy as np
import pytest

from know.config import Config
from know.daemon_db import DaemonDB
from know.index import CodebaseIndex
from know.ranking import fuse_rankings
from know.semantic_search import EmbeddingCache, SemanticSearcher
from know.stats import StatsTracker
from know.token_counter import count_tokens, truncate_to_budget


@pytest.mark.parametrize("owner_type", ["daemon", "index", "stats"])
def test_discarded_database_owner_closes_its_connection(tmp_path, owner_type):
    """Short-lived API objects must not leak one SQLite handle per call."""
    config = Config.create_default(tmp_path)
    if owner_type == "daemon":
        owner = DaemonDB(tmp_path)
        connection = owner._get_conn()
    elif owner_type == "index":
        owner = CodebaseIndex(config)
        connection = owner._get_connection()
    else:
        owner = StatsTracker(config)
        connection = owner._get_conn()

    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()

    assert owner_ref() is None
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


@pytest.mark.parametrize("budget", [-10, 0, 1, 2, 3, 4, 5])
def test_truncate_to_budget_never_exceeds_tiny_or_non_positive_budget(budget):
    result = truncate_to_budget("alpha beta gamma delta", budget)

    assert count_tokens(result) <= max(0, budget)
    if budget <= 0:
        assert result == ""


@pytest.mark.parametrize("method_name", ["search", "search_chunks"])
@pytest.mark.parametrize("top_k", [-1, 0])
def test_semantic_search_non_positive_limit_returns_no_results(
    tmp_path, method_name, top_k
):
    class _Cache:
        def get_all_embeddings(self, _model_name):
            return ["a.py::one::1", "b.py::two::1"], np.array(
                [[1.0, 0.0], [0.5, 0.5]], dtype=np.float32
            )

    searcher = object.__new__(SemanticSearcher)
    searcher.model_name = "test-model"
    searcher.project_root = tmp_path
    searcher.cache = _Cache()
    searcher._get_embedding = lambda _query: np.array([1.0, 0.0], dtype=np.float32)

    if method_name == "search":
        results = searcher.search("query", top_k=top_k)
    else:
        results = searcher.search_chunks(
            "query", tmp_path, top_k=top_k, auto_index=False
        )

    assert results == []


def test_embedding_cache_skips_corrupt_rows_across_read_paths(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)
    model = "test-model"
    cache.set(
        "valid.py",
        "valid-hash",
        np.array([1.0, 2.0], dtype=np.float32),
        model,
    )

    corrupt_path = "corrupt.py"
    corrupt_hash = "corrupt-hash"
    file_hash = hashlib.sha256(
        f"{corrupt_path}:{corrupt_hash}:{model}".encode()
    ).hexdigest()
    with closing(sqlite3.connect(cache.db_path)) as connection:
        connection.execute(
            f"""INSERT INTO {cache._table}
                (file_hash, file_path, content_hash, embedding, model, dim)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (
                file_hash,
                corrupt_path,
                corrupt_hash,
                np.array([9.0], dtype=np.float32).tobytes(),
                model,
                2,
            ),
        )
        connection.commit()

    assert cache.get(corrupt_path, corrupt_hash, model) is None
    assert cache.batch_get([(corrupt_path, corrupt_hash, model)]) == {}
    paths, matrix = cache.get_all_embeddings(model)
    assert paths == ["valid.py"]
    assert matrix.shape == (1, 2)


def test_semantic_memory_recall_skips_wrong_dimension_rows(tmp_path):
    with DaemonDB(tmp_path) as db:
        db.store_memory(
            "valid",
            "valid memory",
            embedding=np.array([1.0, 0.0], dtype=np.float32).tobytes(),
        )
        db.store_memory(
            "corrupt",
            "corrupt memory",
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
        )

        results = db.recall_memories_semantic(
            np.array([1.0, 0.0], dtype=np.float32).tobytes(), limit=10
        )

    assert [row["id"] for row in results] == ["valid"]


def test_session_seen_pads_missing_token_counts_instead_of_dropping_chunks(tmp_path):
    with DaemonDB(tmp_path) as db:
        db.mark_session_seen("session", ["one", "two", "three"], [7])

        assert db.get_session_stats("session") == {
            "chunks_seen": 3,
            "tokens_provided": 7,
        }


def test_rrf_counts_a_candidate_once_per_ranked_list():
    fused = dict(
        fuse_rankings(
            [[("duplicate", 10.0), ("duplicate", 9.0), ("other", 8.0)]],
            k=60,
        )
    )

    assert fused["duplicate"] == pytest.approx(1 / 61)
    # A duplicate is ignored, including for rank position: the next distinct
    # candidate is still the second item in this ranking.
    assert fused["other"] == pytest.approx(1 / 62)


def test_rrf_rejects_nonpositive_smoothing_constant():
    with pytest.raises(ValueError, match="greater than zero"):
        fuse_rankings([[('candidate', 1.0)]], k=0)


@pytest.mark.parametrize("limit", [-1, 0, 1001, True, "10"])
def test_daemon_search_rejects_invalid_or_resource_exhausting_limits(tmp_path, limit):
    with DaemonDB(tmp_path) as db:
        with pytest.raises(ValueError, match="limit must be an integer between 1 and 1000"):
            db.search_chunks("auth", limit=limit)
