"""Adversarial regressions for context assembly and path handling."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from know.context_engine import (
    _GIT_RECENCY_CACHE,
    _GIT_RECENCY_CACHE_TIME,
    CodeChunk,
    ContextEngine,
    _get_batch_file_recency,
    extract_chunks_from_file,
)
from know.diff import ArchitectureDiff
from know.file_categories import categorize_file
from know.token_counter import count_tokens


def _engine(root: Path) -> ContextEngine:
    config = SimpleNamespace(
        root=root,
        project=SimpleNamespace(name="test", description=""),
    )
    return ContextEngine(config)


def _db_chunk(file_path: str, name: str, *, score: float = 1.0) -> dict:
    body = f"def {name}():\n    return True"
    return {
        "file_path": file_path,
        "chunk_name": name,
        "chunk_type": "function",
        "language": "python",
        "start_line": 1,
        "end_line": 2,
        "signature": f"def {name}():",
        "body": body,
        "token_count": count_tokens(body),
        "score": score,
    }


def test_v3_no_tests_filters_retrieved_and_expanded_chunks(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    source = _db_chunk("src/service.py", "service")
    retrieved_test = _db_chunk("tests/test_service.py", "test_service")
    expanded_test = _db_chunk("src/tests/test_neighbor.py", "test_neighbor")
    db = MagicMock()
    db.get_stats.return_value = {"files": 3}
    db.get_importance_batch.return_value = {}

    monkeypatch.setattr(
        engine,
        "_retrieve_hybrid_candidates",
        lambda *args, **kwargs: [source, retrieved_test],
    )

    def expand(_db, chunks, used, _budget, _seen):
        return chunks + [expanded_test], used + expanded_test["token_count"]

    monkeypatch.setattr(engine, "_expand_context", expand)
    monkeypatch.setattr(engine, "_bundle_metadata", lambda *args: None)
    monkeypatch.setattr(engine, "_build_summaries_from_db", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_project_overview", lambda *args: "")

    result = engine._build_context_v3_inner(
        db,
        "service",
        budget=1000,
        include_tests=False,
        include_imports=False,
        include_patterns=None,
        exclude_patterns=None,
        chunk_types=None,
    )

    assert [chunk.file_path for chunk in result["code_chunks"]] == ["src/service.py"]
    assert result["used_tokens"] == result["code_chunks"][0].tokens


def test_legacy_no_tests_excludes_test_files_from_primary_code(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    source = CodeChunk("src/service.py", "service", "function", 1, 2, "source", tokens=1)
    test = CodeChunk("tests/test_service.py", "test_service", "function", 1, 2, "test", tokens=1)

    monkeypatch.setattr(engine, "_collect_all_chunks", lambda: [source, test])
    monkeypatch.setattr(engine, "_score_chunks", lambda _query, chunks: chunks)
    monkeypatch.setattr(engine, "_apply_recency_boost", lambda _chunks: None)
    monkeypatch.setattr(engine, "_build_summaries", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_project_overview", lambda _budget: "")

    result = engine._build_context_legacy(
        "service", budget=100, include_tests=False, include_imports=False,
    )

    assert [chunk.file_path for chunk in result["code_chunks"]] == ["src/service.py"]


def test_legacy_no_tests_excludes_test_files_from_import_expansion(
    tmp_path, monkeypatch,
):
    engine = _engine(tmp_path)
    source = CodeChunk(
        "src/service.py", "service", "function", 1, 2, "source", tokens=1,
    )
    test_dependency = tmp_path / "tests" / "test_helper.py"
    test_dependency.parent.mkdir()
    test_dependency.write_text("def helper():\n    return True\n")

    class Graph:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imports_of(self, _module):
            return ["tests.test_helper"]

        def file_for_module(self, _module):
            return test_dependency

    monkeypatch.setattr("know.import_graph.ImportGraph", Graph)
    monkeypatch.setattr(engine, "_collect_all_chunks", lambda: [source])
    monkeypatch.setattr(engine, "_score_chunks", lambda _query, chunks: chunks)
    monkeypatch.setattr(engine, "_apply_recency_boost", lambda _chunks: None)
    monkeypatch.setattr(engine, "_find_tests", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_build_summaries", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_project_overview", lambda _budget: "")

    with_tests = engine._build_context_legacy(
        "service", budget=1000, include_tests=True, include_imports=True,
    )
    without_tests = engine._build_context_legacy(
        "service", budget=1000, include_tests=False, include_imports=True,
    )

    assert [
        chunk.file_path for chunk in with_tests["dependency_chunks"]
    ] == ["tests/test_helper.py"]
    assert without_tests["dependency_chunks"] == []


def test_v3_no_tests_excludes_test_dependency_signatures(tmp_path):
    engine = _engine(tmp_path)

    class DB:
        def get_imports_batch(self, _modules):
            return {"src.service": ["src.dependency", "tests.test_helper"]}

        def get_signatures(self, file_path):
            return [{"signature": f"def from_{Path(file_path).stem}():"}]

    dependencies, _used = engine._get_dependency_sigs(
        DB(),
        {"src/service.py"},
        budget=1000,
        include_tests=False,
    )

    assert [chunk["file_path"] for chunk in dependencies] == ["src/dependency.py"]


class _DeepDB:
    def __init__(self, target_path: str = "src/target.py"):
        self.target = _db_chunk(target_path, "target")
        self.test_helper = _db_chunk("tests/test_helpers.py", "helper")
        self.caller = _db_chunk("src/caller.py", "caller")

    def get_chunks_by_name(self, name, limit=50):
        return {
            "target": [self.target],
            "helper": [self.test_helper],
            "caller": [self.caller],
        }.get(name, [])

    def get_method_chunks_by_suffix(self, _name, limit=50):
        return []

    def get_callees(self, name, limit=30):
        if name != "target":
            return []
        return [
            {"ref_name": "helper", "file_path": "src/target.py", "line_number": 2},
            {"ref_name": "helper", "file_path": "src/target.py", "line_number": 3},
        ]

    def get_callers(self, name, limit=30):
        if name != "target":
            return []
        return [
            {"containing_chunk": "caller", "file_path": "src/caller.py", "line_number": 4},
            {"containing_chunk": "caller", "file_path": "src/caller.py", "line_number": 8},
        ]

    def search_chunks(self, _name, limit=5):
        return []

    def has_symbol_refs(self):
        return True


def test_deep_no_tests_filters_related_chunks_and_dedupes_call_sites(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(engine, "_collect_deep_refresh_candidates", lambda *args: [])

    result = engine.build_deep_context(
        "target", budget=1000, include_tests=False, db=_DeepDB(),
    )

    assert result["target"]["file"] == "src/target.py"
    assert result["callees"] == []
    assert [caller["name"] for caller in result["callers"]] == ["caller"]
    assert result["callers"][0]["call_site_line"] == 4
    assert result["overflow_signatures"] == []


def test_deep_no_tests_rejects_test_only_direct_target(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(engine, "_collect_deep_refresh_candidates", lambda *args: [])

    result = engine.build_deep_context(
        "target", budget=1000, include_tests=False,
        db=_DeepDB(target_path="tests/test_target.py"),
    )

    assert result == {"error": "not_found", "nearest": []}


def test_git_recency_cache_serves_paths_not_requested_by_first_caller(tmp_path, monkeypatch):
    _GIT_RECENCY_CACHE.clear()
    _GIT_RECENCY_CACHE_TIME.clear()
    now = int(time.time())
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            returncode=0,
            stdout=f"{now}\na.py\nb.py\n",
        )

    monkeypatch.setattr("know.context_engine.subprocess.run", fake_run)

    assert _get_batch_file_recency(tmp_path, ["a.py"])["a.py"] > 0.9
    assert _get_batch_file_recency(tmp_path, ["b.py"])["b.py"] > 0.9
    assert calls == 1


def test_absolute_path_is_categorized_relative_to_explicit_repo_root(tmp_path):
    root = tmp_path / "build" / "repo"
    assert categorize_file(root / "src" / "main.py", root=root) == "source"
    assert categorize_file(root / "vendor" / "lib.py", root=root) == "vendor"


def test_class_chunk_token_count_matches_returned_full_body(tmp_path):
    source = tmp_path / "large.py"
    source.write_text(
        "class Large:\n"
        "    def method(self):\n"
        "        return [\n"
        + "".join(f"            {i},\n" for i in range(100))
        + "        ]\n"
    )

    class_chunk = next(
        chunk for chunk in extract_chunks_from_file(source, tmp_path)
        if chunk.chunk_type == "class"
    )

    assert class_chunk.tokens == count_tokens(class_chunk.body)


def test_root_commit_reports_its_changed_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "first.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "first.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "root"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    assert ArchitectureDiff(tmp_path).get_changed_files(commit) == ["first.py"]
