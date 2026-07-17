"""Adversarial lifecycle tests for semantic embedding state."""

from __future__ import annotations

import numpy as np

from know.semantic_search import EmbeddingCache, SemanticSearcher


def test_set_replaces_stale_content_version_for_logical_path(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)

    cache.set("file:/repo/a.py", "old", np.array([1.0, 0.0], dtype=np.float32), "model")
    cache.set("file:/repo/a.py", "new", np.array([0.0, 1.0], dtype=np.float32), "model")

    assert cache.get("file:/repo/a.py", "old", "model") is None
    paths, embeddings = cache.get_all_embeddings("model", path_prefix="file:")
    assert paths == ["file:/repo/a.py"]
    np.testing.assert_array_equal(embeddings, [[0.0, 1.0]])


def test_batch_set_keeps_only_latest_item_per_logical_path(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)

    cache.batch_set([
        ("chunk:a.py::f::1", "old", np.array([1.0, 0.0], dtype=np.float32), "model"),
        ("chunk:a.py::f::1", "new", np.array([0.0, 1.0], dtype=np.float32), "model"),
    ])

    assert cache.get("chunk:a.py::f::1", "old", "model") is None
    paths, embeddings = cache.get_all_embeddings("model", path_prefix="chunk:")
    assert paths == ["chunk:a.py::f::1"]
    np.testing.assert_array_equal(embeddings, [[0.0, 1.0]])


def test_file_and_chunk_searches_do_not_cross_contaminate(tmp_path):
    searcher = object.__new__(SemanticSearcher)
    searcher.model_name = "model"
    searcher.cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)
    searcher.cache.set(
        "file:/repo/a.py", "file", np.array([1.0, 0.0], dtype=np.float32), "model",
    )
    searcher.cache.set(
        "chunk:src/b.py::work::4", "chunk", np.array([1.0, 0.0], dtype=np.float32), "model",
    )
    searcher._get_embedding = lambda _query: np.array([1.0, 0.0], dtype=np.float32)

    assert searcher.search("query") == [("/repo/a.py", 1.0)]
    assert searcher.search_chunks("query", tmp_path, auto_index=False) == [{
        "path": "src/b.py",
        "name": "work",
        "line_start": 4,
        "score": 1.0,
        "chunk_key": "src/b.py::work::4",
    }]


def test_repo_parent_named_build_does_not_exclude_source_files(tmp_path, monkeypatch):
    root = tmp_path / "build" / "repo"
    source = root / "src" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("def main():\n    return 1\n")
    searcher = SemanticSearcher(project_root=root)
    indexed = []
    monkeypatch.setattr(searcher, "index_file", lambda path: indexed.append(path) or True)

    assert searcher.index_directory(root, extensions=[".py"]) == 1
    assert indexed == [source]


def test_authoritative_prune_removes_deleted_paths_only_in_requested_namespace(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    cache.set("file:/repo/removed.py", "x", vector, "model")
    cache.set("file:/repo/kept.py", "x", vector, "model")
    cache.set("chunk:removed.py::f::1", "x", vector, "model")

    cache.prune_paths("model", "file:", {"file:/repo/kept.py"})

    file_paths, _ = cache.get_all_embeddings("model", path_prefix="file:")
    chunk_paths, _ = cache.get_all_embeddings("model", path_prefix="chunk:")
    assert file_paths == ["file:/repo/kept.py"]
    assert chunk_paths == ["chunk:removed.py::f::1"]


def test_search_uses_current_query_dimension_when_stale_dimension_is_dominant(tmp_path):
    searcher = object.__new__(SemanticSearcher)
    searcher.model_name = "model"
    searcher.cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)
    # Two stale rows make the old dimension dominant; the current-model row is
    # the only one compatible with the query and must still be searchable.
    searcher.cache.set(
        "file:/repo/stale-a.py", "a", np.array([1.0, 0.0], dtype=np.float32), "model",
    )
    searcher.cache.set(
        "file:/repo/stale-b.py", "b", np.array([0.0, 1.0], dtype=np.float32), "model",
    )
    searcher.cache.set(
        "file:/repo/current.py", "c", np.array([1.0, 0.0, 0.0], dtype=np.float32), "model",
    )
    searcher._get_embedding = lambda _query: np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert searcher.search("query") == [("/repo/current.py", 1.0)]


def test_search_returns_empty_when_every_cached_vector_has_stale_dimension(tmp_path):
    searcher = object.__new__(SemanticSearcher)
    searcher.model_name = "model"
    searcher.cache = EmbeddingCache(cache_dir=tmp_path / "cache", project_root=tmp_path)
    searcher.cache.set(
        "file:/repo/stale.py", "a", np.array([1.0, 0.0], dtype=np.float32), "model",
    )
    searcher._get_embedding = lambda _query: np.array([1.0, 0.0, 0.0], dtype=np.float32)

    assert searcher.search("query") == []
    paths, matrix = searcher.cache.get_all_embeddings(
        "model", path_prefix="file:", expected_dim=3,
    )
    assert paths == []
    assert matrix.size == 0


def test_reindexing_empty_file_removes_its_last_good_embedding(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    searcher = SemanticSearcher(project_root=tmp_path)
    monkeypatch.setattr(
        searcher,
        "_get_embedding",
        lambda _text: np.array([1.0, 0.0], dtype=np.float32),
    )

    assert searcher.index_file(source) is True
    source.write_text("\n", encoding="utf-8")
    assert searcher.index_file(source) is False

    paths, _matrix = searcher.cache.get_all_embeddings(
        searcher.model_name,
        path_prefix=searcher.FILE_CACHE_PREFIX,
    )
    assert f"file:{source}" not in paths


def test_directory_reindex_does_not_retain_non_indexable_discovered_path(
    tmp_path, monkeypatch
):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    searcher = SemanticSearcher(project_root=tmp_path)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    searcher.cache.set(f"file:{source}", "old", vector, searcher.model_name)
    monkeypatch.setattr(searcher, "index_file", lambda _path: False)

    assert searcher.index_directory(tmp_path, extensions=[".py"]) == 0
    paths, _matrix = searcher.cache.get_all_embeddings(
        searcher.model_name,
        path_prefix=searcher.FILE_CACHE_PREFIX,
    )
    assert f"file:{source}" not in paths


def test_semantic_indexers_reject_source_symlink_that_escapes_root(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("def secret():\n    return 'outside'\n", encoding="utf-8")
    escaped = root / "escape.py"
    escaped.symlink_to(outside)
    searcher = SemanticSearcher(project_root=root)
    monkeypatch.setattr(
        searcher,
        "_get_embedding",
        lambda _text: np.array([1.0, 0.0], dtype=np.float32),
    )

    assert searcher.index_file(escaped) is False
    assert searcher.index_directory(root, extensions=[".py"]) == 0
    assert searcher.index_chunks(root, extensions=[".py"]) == 0
    file_paths, _ = searcher.cache.get_all_embeddings(
        searcher.model_name, path_prefix=searcher.FILE_CACHE_PREFIX,
    )
    chunk_paths, _ = searcher.cache.get_all_embeddings(
        searcher.model_name, path_prefix=searcher.CHUNK_CACHE_PREFIX,
    )
    assert file_paths == []
    assert chunk_paths == []
