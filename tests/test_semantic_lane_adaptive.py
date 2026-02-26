"""TDD coverage for adaptive semantic lane performance guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.config import Config
import know.context_engine as context_engine_mod
from know.context_engine import ContextEngine


class _FakeModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        for text in batch:
            seed = float((sum(text.encode("utf-8", errors="ignore")) % 13) + 1)
            yield [seed, 1.0, 0.5, 0.25]


def _mk_config(tmp_path: Path) -> Config:
    (tmp_path / ".know").mkdir()
    cfg = Config.create_default(tmp_path)
    cfg.root = tmp_path
    cfg.save(tmp_path / ".know" / "config.yaml")
    return cfg


def _mk_candidate(i: int) -> dict:
    return {
        "file_path": f"src/m{i % 7}.py",
        "chunk_name": f"fn_{i}",
        "signature": f"def fn_{i}(x):",
        "body": f"def fn_{i}(x):\n    return x + {i}\n",
        "start_line": i + 1,
        "score": float(100 - i),
    }


def test_semantic_lane_caps_embeddings_for_large_repos(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    cfg = _mk_config(tmp_path)
    engine = ContextEngine(cfg)
    model = _FakeModel()

    context_engine_mod._clear_semantic_embedding_caches()
    monkeypatch.setattr(context_engine_mod, "_get_cached_embedding_model", lambda: model)
    monkeypatch.setenv("KNOW_SEMANTIC_MAX_CANDIDATES", "10")
    monkeypatch.setenv("KNOW_SEMANTIC_BATCH_SIZE", "4")

    candidates = [_mk_candidate(i) for i in range(100)]
    scored = engine._semantic_rerank_lane(
        "agent provider mapping",
        candidates,
        limit=30,
        repo_file_count=5000,
    )

    assert scored  # still returns usable semantic ranking
    # One query embed + at most 10 candidate embeds.
    total_embedded = sum(len(batch) for batch in model.calls)
    assert total_embedded <= 11


def test_semantic_lane_reuses_cached_embeddings_across_calls(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    cfg = _mk_config(tmp_path)
    engine = ContextEngine(cfg)
    model = _FakeModel()

    context_engine_mod._clear_semantic_embedding_caches()
    monkeypatch.setattr(context_engine_mod, "_get_cached_embedding_model", lambda: model)
    monkeypatch.setenv("KNOW_SEMANTIC_MAX_CANDIDATES", "20")
    monkeypatch.setenv("KNOW_SEMANTIC_BATCH_SIZE", "8")

    candidates = [_mk_candidate(i) for i in range(20)]
    first = engine._semantic_rerank_lane("credential lookup", candidates, limit=20, repo_file_count=2000)
    assert first
    first_total = sum(len(batch) for batch in model.calls)
    assert first_total >= 2  # query + candidate embeddings

    model.calls.clear()
    second = engine._semantic_rerank_lane("credential lookup", candidates, limit=20, repo_file_count=2000)
    assert second
    second_total = sum(len(batch) for batch in model.calls)
    # Cache should eliminate re-embedding all candidates on repeated query.
    assert second_total <= 1


def test_retrieve_hybrid_candidates_fast_profile_skips_semantic(tmp_path, monkeypatch):
    cfg = _mk_config(tmp_path)
    engine = ContextEngine(cfg)

    class _DB:
        def search_chunks(self, _query, limit=100):
            return [_mk_candidate(i) for i in range(min(20, limit))]

    db = _DB()
    semantic_calls = {"n": 0}

    def _fake_semantic(*args, **kwargs):
        semantic_calls["n"] += 1
        return []

    monkeypatch.setattr(engine, "_graph_expand_lane", lambda *_a, **_k: [])
    monkeypatch.setattr(engine, "_semantic_rerank_lane", _fake_semantic)

    out = engine._retrieve_hybrid_candidates(
        db,
        "provider credentials",
        limit=40,
        repo_file_count=2000,
        retrieval_profile="fast",
        semantic_max_ms=1000,
    )

    assert out
    assert semantic_calls["n"] == 0
