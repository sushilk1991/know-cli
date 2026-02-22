"""Resilience tests for embedding model initialization."""

import sys
from types import SimpleNamespace


def test_get_model_returns_none_on_runtime_model_error(monkeypatch):
    """Corrupted ONNX/cache errors should not crash callers."""
    import know.embeddings as emb

    emb._model_cache.clear()

    class BrokenEmbedding:
        def __init__(self, model_name=None):
            raise RuntimeError(
                "Load model from /tmp/fastembed_cache/BAAI_bge/model_optimized.onnx "
                "failed: File doesn't exist"
            )

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=BrokenEmbedding))
    monkeypatch.setattr(emb, "_repair_fastembed_cache", lambda _e: False)

    model = emb.get_model("broken-model")
    assert model is None


def test_get_model_retries_once_after_cache_repair(monkeypatch):
    """If cache repair succeeds, model init should retry once and recover."""
    import know.embeddings as emb

    emb._model_cache.clear()
    attempts = {"n": 0}

    class FlakyEmbedding:
        def __init__(self, model_name=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError(
                    "Load model from /tmp/fastembed_cache/BAAI_bge/model_optimized.onnx "
                    "failed: File doesn't exist"
                )

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FlakyEmbedding))
    monkeypatch.setattr(emb, "_repair_fastembed_cache", lambda _e: True)

    model = emb.get_model("flaky-model")
    assert model is not None
    assert attempts["n"] == 2

