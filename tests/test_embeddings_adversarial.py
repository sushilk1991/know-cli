"""Adversarial tests for automatic FastEmbed cache repair."""

from pathlib import Path

from know import embeddings


def _cache_error(onnx_path: Path) -> RuntimeError:
    return RuntimeError(f"Load model from {onnx_path} failed: File doesn't exist")


def test_cache_repair_removes_only_the_model_directory(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed_cache"
    model_dir = cache_root / "models--qdrant--bge-small-en-v1.5-onnx-q"
    onnx_path = model_dir / "onnx" / "model_optimized.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_bytes(b"corrupt")
    sibling = cache_root / "models--qdrant--another-model"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    assert embeddings._repair_fastembed_cache(_cache_error(onnx_path)) is True

    assert not model_dir.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert cache_root.is_dir()


def test_cache_repair_rejects_parent_traversal(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed_cache"
    cache_root.mkdir()
    important = tmp_path / "important"
    important.mkdir()
    onnx_path = important / "model_optimized.onnx"
    onnx_path.write_bytes(b"valuable")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    traversal_path = cache_root / ".." / "important" / onnx_path.name
    assert embeddings._repair_fastembed_cache(_cache_error(traversal_path)) is False
    assert onnx_path.read_bytes() == b"valuable"


def test_cache_repair_rejects_lookalike_sibling(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed_cache"
    cache_root.mkdir()
    lookalike_model = tmp_path / "fastembed_cache_notes" / "model"
    onnx_path = lookalike_model / "model_optimized.onnx"
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_bytes(b"valuable")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    assert embeddings._repair_fastembed_cache(_cache_error(onnx_path)) is False
    assert onnx_path.read_bytes() == b"valuable"


def test_cache_repair_rejects_model_symlink_escape(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed_cache"
    cache_root.mkdir()
    outside_model = tmp_path / "outside-model"
    outside_model.mkdir()
    onnx_path = outside_model / "model_optimized.onnx"
    onnx_path.write_bytes(b"valuable")
    (cache_root / "models--escaped").symlink_to(outside_model, target_is_directory=True)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    reported_path = cache_root / "models--escaped" / onnx_path.name
    assert embeddings._repair_fastembed_cache(_cache_error(reported_path)) is False
    assert onnx_path.read_bytes() == b"valuable"
    assert outside_model.is_dir()


def test_cache_repair_rejects_relative_error_path(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed_cache"
    model_dir = cache_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model_optimized.onnx").write_bytes(b"valuable")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    relative = Path("fastembed_cache/model/model_optimized.onnx")
    assert embeddings._repair_fastembed_cache(_cache_error(relative)) is False
    assert model_dir.is_dir()


def test_cache_repair_never_selects_the_cache_root_itself(tmp_path, monkeypatch):
    cache_root = tmp_path / "fastembed"
    cache_root.mkdir()
    onnx_path = cache_root / "model_optimized.onnx"
    onnx_path.write_bytes(b"corrupt")
    keep = cache_root / "keep.txt"
    keep.write_text("valuable", encoding="utf-8")
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache_root))

    assert embeddings._repair_fastembed_cache(_cache_error(onnx_path)) is False
    assert keep.read_text(encoding="utf-8") == "valuable"
