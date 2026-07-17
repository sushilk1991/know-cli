"""Adversarial regressions for indexing, persistence, and refresh state."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from threading import Barrier, Event, local

import pytest

from know.config import Config, OutputConfig
from know.daemon import (
    KnowDaemon,
    _collect_project_file_mtimes,
    populate_index,
    refresh_file_if_stale,
)
from know.daemon_db import DaemonDB
from know.import_graph import ImportGraph
from know.index import CodebaseIndex, read_source_snapshot
from know.models import FunctionInfo, ModuleInfo
from know.parsers import PythonParser
from know.scanner import CodebaseScanner


def _config(root: Path) -> Config:
    (root / ".know" / "cache").mkdir(parents=True, exist_ok=True)
    return Config(
        root=root,
        exclude=[],
        include=[],
        output=OutputConfig(directory="docs", watch=OutputConfig.WatchConfig()),
    )


def _chunk(name: str) -> dict:
    return {
        "name": name,
        "type": "function",
        "start_line": 1,
        "end_line": 1,
        "signature": f"{name}()",
        "body": f"def {name}(): pass",
    }


@pytest.mark.parametrize("operation", ["chunks", "imports", "symbol_refs"])
def test_delete_then_replace_is_atomic_on_malformed_later_row(tmp_path, operation):
    db = DaemonDB(tmp_path)
    try:
        if operation == "chunks":
            db.upsert_chunks("a.py", "python", [_chunk("old")])
            with pytest.raises(AttributeError):
                db.upsert_chunks("a.py", "python", [_chunk("new"), None])
            assert [row["chunk_name"] for row in db.get_chunks_for_file("a.py")] == ["old"]
        elif operation == "imports":
            db.set_imports("pkg.a", [("pkg.old", "import")])
            with pytest.raises(ValueError):
                db.set_imports("pkg.a", [("pkg.new", "import"), ("broken",)])
            assert db.get_imports_of("pkg.a") == ["pkg.old"]
        else:
            db.upsert_symbol_refs("a.py", [{
                "ref_name": "old", "ref_type": "call",
                "line_number": 1, "containing_chunk": "caller",
            }])
            with pytest.raises(KeyError):
                db.upsert_symbol_refs("a.py", [{
                    "ref_name": "new", "ref_type": "call",
                    "line_number": 2, "containing_chunk": "caller",
                }, {"ref_name": "broken"}])
            assert len(db.get_callers("old")) == 1
            assert not db.get_callers("new")
    finally:
        db.close()


def test_failed_replace_can_be_caught_inside_outer_batch_without_partial_commit(tmp_path):
    db = DaemonDB(tmp_path)
    try:
        db.upsert_chunks("a.py", "python", [_chunk("old")])
        with db.batch():
            try:
                db.upsert_chunks("a.py", "python", [_chunk("new"), None])
            except AttributeError:
                pass
            db.update_file_index("other.py", "hash", "python", 0)

        assert [row["chunk_name"] for row in db.get_chunks_for_file("a.py")] == ["old"]
        assert db.get_file_hash("other.py") == "hash"
    finally:
        db.close()


def test_malformed_session_tokens_rollback_session_and_allow_later_write(tmp_path):
    db = DaemonDB(tmp_path)
    try:
        with pytest.raises(sqlite3.ProgrammingError):
            db.mark_session_seen("recoverable", ["broken"], [object()])

        conn = db._get_conn()
        assert conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", ("recoverable",),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM session_seen WHERE session_id = ?", ("recoverable",),
        ).fetchone() is None

        db.mark_session_seen("recoverable", ["healthy"], [3])
        assert db.get_session_seen("recoverable") == {"healthy"}
        assert db.get_session_stats("recoverable") == {
            "chunks_seen": 1,
            "tokens_provided": 3,
        }
    finally:
        db.close()


def test_concurrent_duplicate_memory_insert_has_one_winner(tmp_path):
    db_a = DaemonDB(tmp_path)
    db_b = DaemonDB(tmp_path)
    gate = Barrier(2)

    def store(db: DaemonDB, memory_id: str) -> bool:
        gate.wait()
        try:
            return db.store_memory(memory_id, "same concurrent content")
        finally:
            db.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(store, (db_a, db_b), ("a", "b")))
        assert sorted(results) == [False, True]
        with DaemonDB(tmp_path) as verifier:
            assert verifier.count_memories() == 1
    finally:
        db_a.close()
        db_b.close()


def test_memory_display_ids_are_stable_and_never_reused_after_deletion(tmp_path):
    db = DaemonDB(tmp_path)
    try:
        assert db.store_memory("a", "alpha memory")
        assert db.store_memory("b", "beta memory")
        assert db.store_memory("c", "gamma memory")
        assert db.delete_memory("b")

        assert {row["_display_id"] for row in db.list_memories()} == {1, 3}
        assert db.get_memory_by_id("c")["_display_id"] == 3
        assert db.recall_memories("gamma")[0]["_display_id"] == 3

        assert db.delete_memory("c")
        assert db.store_memory("d", "delta memory")
        assert db.get_memory_by_id("d")["_display_id"] == 4
        assert {row["_display_id"] for row in db.list_memories()} == {1, 4}
    finally:
        db.close()


def test_pre_v7_display_id_migration_preserves_created_at_order(tmp_path):
    db = DaemonDB(tmp_path)
    conn = db._get_conn()
    try:
        # Deliberately insert the newer memory first so rowid order disagrees
        # with the historical created_at-based display-ID contract.
        conn.executemany(
            """INSERT INTO memories (id, content, created_at, content_hash)
               VALUES (?, ?, ?, ?)""",
            [
                ("newer", "newer memory", 200.0, "newer-hash"),
                ("older", "older memory", 100.0, "older-hash"),
            ],
        )
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (6, 0)",
        )
        conn.commit()
    finally:
        db.close()

    migrated = DaemonDB(tmp_path)
    try:
        rows = migrated._get_conn().execute(
            "SELECT display_id, memory_id FROM memory_display_ids "
            "ORDER BY display_id",
        ).fetchall()
        assert [tuple(row) for row in rows] == [(1, "older"), (2, "newer")]
    finally:
        migrated.close()


def test_scanner_rejects_source_symlink_that_escapes_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("def secret(): pass\n", encoding="utf-8")
    (root / "escape.py").symlink_to(outside)

    scanner = CodebaseScanner(_config(root))
    try:
        assert list(scanner._discover_files()) == []
    finally:
        scanner.close()


def test_refresh_accepts_project_root_reached_through_symlink_alias(tmp_path):
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    alias_root = tmp_path / "project-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    source = real_root / "module.py"
    source.write_text("def current():\n    return 1\n", encoding="utf-8")
    config = Config.create_default(alias_root)

    with DaemonDB(real_root) as db:
        result = refresh_file_if_stale(alias_root, config, db, "module.py")

    assert result["updated"] is True
    assert result["file_path"] == "module.py"


def test_scan_files_rebuilds_state_and_counts_without_duplicates(tmp_path):
    root = tmp_path
    (root / "a.py").write_text("def old(): pass\n", encoding="utf-8")
    (root / "b.py").write_text("class B:\n    def method(self): pass\n", encoding="utf-8")
    scanner = CodebaseScanner(_config(root))
    try:
        assert scanner.scan(max_workers=1)["modules"] == 2
        (root / "a.py").write_text("def fresh(): pass\n", encoding="utf-8")

        first = scanner.scan_files([Path("a.py")])
        second = scanner.scan_files([root / "a.py"])

        assert first == second
        assert second["modules"] == 2
        assert second["functions"] == 2  # fresh plus B.method
        assert [m.name for m in scanner.modules].count("a") == 1
        refreshed = next(module for module in scanner.modules if module.name == "a")
        assert refreshed.functions[0].name == "fresh"
    finally:
        scanner.close()


def test_scan_files_retains_last_good_module_after_parse_failure(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("def stable(): pass\n", encoding="utf-8")
    scanner = CodebaseScanner(_config(tmp_path))
    try:
        scanner.scan(max_workers=1)
        source.write_text("def broken(:\n", encoding="utf-8")

        stats = scanner.scan_files([source])

        assert stats["modules"] == 1
        assert scanner.modules[0].functions[0].name == "stable"
    finally:
        scanner.close()


def test_python_parser_preserves_relative_aliases_and_full_signature(tmp_path):
    source = tmp_path / "service.py"
    source.write_text(
        "from . import utils, helpers\n"
        "def f(a, /, b=1, *args, c: int = 2, **kwargs): pass\n",
        encoding="utf-8",
    )
    module = PythonParser().parse(source, tmp_path)

    assert module.imports == [".utils", ".helpers"]
    assert module.functions[0].signature == "f(a, /, b=1, *args, c: int=2, **kwargs)"


def test_import_graph_empty_modules_is_authoritative_not_filesystem_fallback(tmp_path):
    (tmp_path / "a.py").write_text("import b\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = ImportGraph(_config(tmp_path))
    try:
        assert graph.build() == 1
        assert graph.get_all_edges() == [("a", "b")]
        assert graph.build([]) == 0
        assert graph.get_all_edges() == []
    finally:
        graph.close()


def test_import_graph_ignores_hard_excluded_ancestors_outside_root(tmp_path):
    root = tmp_path / "build" / "repo"
    root.mkdir(parents=True)
    (root / "a.py").write_text("import b\n", encoding="utf-8")
    (root / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = ImportGraph(_config(root))
    try:
        assert graph.build() == 1
        assert graph.get_all_edges() == [("a", "b")]
    finally:
        graph.close()


def test_import_graph_resolves_from_dot_import_alias(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "service.py").write_text("from . import utils\n", encoding="utf-8")
    (pkg / "utils.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = ImportGraph(_config(tmp_path))
    try:
        graph.build()
        assert graph.imports_of("pkg.service") == ["pkg.utils"]
    finally:
        graph.close()


def test_import_graph_mixed_rebuild_preserves_failed_clears_empty_and_removes_absent(
    tmp_path, monkeypatch,
):
    failing = tmp_path / "a.py"
    emptied = tmp_path / "b.py"
    removed = tmp_path / "d.py"
    failing.write_text("import c\n", encoding="utf-8")
    emptied.write_text("import c\n", encoding="utf-8")
    removed.write_text("import c\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = ImportGraph(_config(tmp_path))
    try:
        assert graph.build() == 3
        emptied.write_text("VALUE = 2\n", encoding="utf-8")
        removed.unlink()
        real_read_text = Path.read_text

        def fail_one_read(path, *args, **kwargs):
            if path == failing:
                raise OSError("transient read failure")
            return real_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_one_read)

        assert graph.build() == 0
        assert graph.get_all_edges() == [("a", "c")]
    finally:
        graph.close()


def test_import_graph_preserves_indexed_module_omitted_after_full_scan_parse_failure(
    tmp_path,
):
    failing = tmp_path / "a.py"
    failing.write_text("import b\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    graph = ImportGraph(_config(tmp_path))
    try:
        assert graph.build() == 1
        graph._db.update_file_index("a.py", "old-a", "python", 1)
        graph._db.update_file_index("b.py", "old-b", "python", 1)

        failing.write_text("def broken(:\n", encoding="utf-8")
        assert graph.build([{"path": "b.py", "name": "b"}]) == 0
        assert graph.get_all_edges() == [("a", "b")]
    finally:
        graph.close()


def test_concurrent_import_build_cannot_publish_an_older_snapshot_last(
    tmp_path, monkeypatch,
):
    source = tmp_path / "a.py"
    source.write_text("import b\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 'b'\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("VALUE = 'c'\n", encoding="utf-8")
    config = _config(tmp_path)
    baseline = ImportGraph(config)
    old_graph = ImportGraph(config)
    new_graph = ImportGraph(config)
    old_snapshot_read = Event()
    release_old = Event()
    worker = local()
    real_parse = __import__("ast").parse

    def controlled_parse(text, *args, **kwargs):
        if getattr(worker, "generation", "") == "old" and text == "import b\n":
            old_snapshot_read.set()
            if not release_old.wait(5):
                raise TimeoutError("test did not release old import build")
        return real_parse(text, *args, **kwargs)

    def run_build(graph, generation):
        worker.generation = generation
        return graph.build()

    monkeypatch.setattr("know.import_graph.ast.parse", controlled_parse)
    try:
        assert baseline.build() == 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            old_future = pool.submit(run_build, old_graph, "old")
            if not old_snapshot_read.wait(5):
                release_old.set()
                pytest.fail("old import build never parsed its source snapshot")

            source.write_text("import c\n", encoding="utf-8")
            new_future = pool.submit(run_build, new_graph, "new")
            try:
                # Without build serialization, the new generation publishes
                # while the old parser is paused. The old build then regresses
                # the graph when released.
                new_future.result(timeout=1)
            except FutureTimeout:
                pass
            finally:
                release_old.set()

            old_future.result(timeout=5)
            new_future.result(timeout=5)

        assert baseline.imports_of("a") == ["c"]
    finally:
        release_old.set()
        baseline.close()
        old_graph.close()
        new_graph.close()


def test_populate_index_keeps_last_good_data_on_transient_parse_failure(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("def working(): return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    db = DaemonDB(tmp_path)
    try:
        populate_index(tmp_path, config, db)
        old_hash = db.get_file_hash("service.py")
        assert db.get_chunks_by_name("working")

        source.write_text("def broken(:\n", encoding="utf-8")
        populate_index(tmp_path, config, db)

        assert db.get_file_hash("service.py") == old_hash
        assert db.get_chunks_by_name("working")
    finally:
        db.close()


def test_full_index_cannot_overwrite_a_newer_refresh_generation(
    tmp_path, monkeypatch,
):
    source = tmp_path / "service.py"
    source.write_text("def older(): return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    old_db = DaemonDB(tmp_path)
    new_db = DaemonDB(tmp_path)
    old_ready_to_publish = Event()
    release_old = Event()
    real_old_batch = old_db.batch

    def paused_old_batch():
        old_ready_to_publish.set()
        if not release_old.wait(5):
            raise TimeoutError("test did not release old index publication")
        return real_old_batch()

    monkeypatch.setattr(old_db, "batch", paused_old_batch)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            old_future = pool.submit(populate_index, tmp_path, config, old_db)
            if not old_ready_to_publish.wait(5):
                release_old.set()
                pytest.fail("full index never reached its publication boundary")

            source.write_text("def newer(): return 2\n", encoding="utf-8")
            new_future = pool.submit(
                refresh_file_if_stale,
                tmp_path,
                config,
                new_db,
                source,
                force=True,
            )
            try:
                # An unversioned publisher can commit this newer refresh first,
                # then let the paused full index overwrite it with old chunks.
                new_future.result(timeout=1)
            except FutureTimeout:
                pass
            finally:
                release_old.set()

            old_future.result(timeout=5)
            result = new_future.result(timeout=5)

        assert result["reason"] == "reindexed"
        expected = read_source_snapshot(source)
        with DaemonDB(tmp_path) as verifier:
            assert verifier.get_file_hash("service.py") == expected["content_hash"]
            assert [
                row["chunk_name"]
                for row in verifier.get_chunks_for_file("service.py")
            ] == ["newer"]
    finally:
        release_old.set()
        old_db.close()
        new_db.close()


def test_refresh_discards_parse_result_if_file_changes_mid_parse(tmp_path, monkeypatch):
    source = tmp_path / "service.py"
    version_a = "def version_a(): return 1\n"
    source.write_text(version_a, encoding="utf-8")
    config = _config(tmp_path)
    db = DaemonDB(tmp_path)
    module_a = ModuleInfo(
        path=Path("service.py"),
        name="service",
        docstring=None,
        functions=[FunctionInfo(
            name="version_a", line_number=1, end_line=1, docstring=None,
            signature="version_a()", is_async=False, is_method=False,
        )],
        classes=[],
        imports=[],
    )

    class MutatingParser:
        def parse(self, _path, _root):
            source.write_text("def version_b(): return 2\n", encoding="utf-8")
            return module_a

        def extract_call_refs(self, _content, _module):
            return []

    monkeypatch.setattr(
        "know.parsers.ParserFactory.get_parser_for_file",
        lambda _path: MutatingParser(),
    )
    try:
        result = refresh_file_if_stale(tmp_path, config, db, source)
        assert result["reason"] == "changed_during_parse"
        assert db.get_file_hash("service.py") is None
        assert not db.get_chunks_by_name("version_a")
    finally:
        db.close()


def test_refresh_rejects_aba_source_generation(tmp_path, monkeypatch):
    source = tmp_path / "service.py"
    version_a = "def stable(): return 1\n"
    version_b = "def mutant(): return 2\n"
    assert len(version_a) == len(version_b)
    source.write_text(version_a, encoding="utf-8")
    original_stat = source.stat()
    db = DaemonDB(tmp_path)

    class ABAParser:
        def parse(self, path, root):
            source.write_text(version_b, encoding="utf-8")
            module = PythonParser().parse(path, root)
            source.write_text(version_a, encoding="utf-8")
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return module

        def extract_call_refs(self, _content, _module):
            return []

    monkeypatch.setattr(
        "know.parsers.ParserFactory.get_parser_for_file", lambda _path: ABAParser(),
    )
    try:
        result = refresh_file_if_stale(tmp_path, _config(tmp_path), db, source)
        assert result["reason"] == "changed_during_parse"
        assert db.get_file_hash("service.py") is None
        assert db.get_chunks_by_name("mutant") == []
    finally:
        db.close()


def test_incremental_scanner_rejects_aba_source_generation(tmp_path, monkeypatch):
    source = tmp_path / "service.py"
    version_a = "def stable(): return 1\n"
    version_b = "def mutant(): return 2\n"
    assert len(version_a) == len(version_b)
    source.write_text(version_a, encoding="utf-8")
    original_stat = source.stat()

    class ABAParser:
        language = "python"

        def parse(self, path, root):
            source.write_text(version_b, encoding="utf-8")
            module = PythonParser().parse(path, root)
            source.write_text(version_a, encoding="utf-8")
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            return module

    monkeypatch.setattr(
        "know.parsers.ParserFactory.get_parser_for_file",
        lambda _path, _use_treesitter=True: ABAParser(),
    )
    scanner = CodebaseScanner(_config(tmp_path))
    try:
        stats = scanner.scan_files([source])
        assert stats["modules"] == 0
        assert scanner.modules == []
        assert scanner._get_index().get_file_metadata(source) is None
    finally:
        scanner.close()


def test_old_cache_snapshot_is_never_treated_as_current_after_race(tmp_path):
    source = tmp_path / "service.py"
    source.write_text("def version_a(): pass\n", encoding="utf-8")
    index = CodebaseIndex(_config(tmp_path))
    try:
        snapshot_a = read_source_snapshot(source)
        source.write_text("def version_b(): pass\n", encoding="utf-8")
        index.cache_file(source, "python", {"path": "service.py", "name": "service"}, snapshot=snapshot_a)

        assert index.is_file_changed(source)
        assert index.get_cached_module(source) is None
    finally:
        index.close()


@pytest.mark.parametrize("failure", ["serialization", "sqlite_write"])
def test_cache_file_failure_keeps_old_file_and_module_snapshot_stale(
    tmp_path, failure,
):
    source = tmp_path / "service.py"
    source.write_text("def old(): pass\n", encoding="utf-8")
    index = CodebaseIndex(_config(tmp_path))
    try:
        old_module = {"path": "service.py", "name": "service"}
        index.cache_file(source, "python", old_module)
        old_metadata = index.get_file_metadata(source)
        source.write_text("def new(): pass\n", encoding="utf-8")

        if failure == "serialization":
            new_module = {
                "path": "service.py", "name": "service", "bad": object(),
            }
        else:
            index._get_connection().executescript(
                """CREATE TRIGGER reject_module_cache
                   BEFORE INSERT ON modules
                   BEGIN
                       SELECT RAISE(ABORT, 'reject module write');
                   END;"""
            )
            new_module = {"path": "service.py", "name": "service"}

        index.cache_file(source, "python", new_module)

        assert index.get_file_metadata(source)["content_hash"] == old_metadata["content_hash"]
        assert index.is_file_changed(source)
        assert index.get_cached_module(source) is None
    finally:
        index.close()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"path": "other.py", "name": "service"},
    ],
    ids=["non_mapping", "missing_identity", "wrong_path"],
)
def test_invalid_cached_module_payload_is_treated_as_cache_miss(tmp_path, payload):
    source = tmp_path / "service.py"
    source.write_text("def current(): pass\n", encoding="utf-8")
    index = CodebaseIndex(_config(tmp_path))
    try:
        index.cache_file(
            source, "python", {"path": "service.py", "name": "service"},
        )
        conn = index._get_connection()
        conn.execute(
            "UPDATE modules SET data = ? WHERE path = ?",
            (json.dumps(payload), "service.py"),
        )
        conn.commit()

        assert index.get_cached_module_snapshot(source) is None
        changed, cached, snapshots = index.get_changed_files([source])
        assert changed == [source]
        assert cached == []
        assert snapshots == {}
    finally:
        index.close()


def test_incremental_refresh_updates_imports_and_clears_symbol_refs(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("from b import helper\ndef run(): return helper()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def helper(): return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    try:
        daemon._full_index_sync()
        assert daemon.db.get_imports_of("a") == ["b"]
        assert daemon.db.get_callers("helper")
        daemon._file_mtime_snapshot = _collect_project_file_mtimes(config)

        old_mtime_ns = source.stat().st_mtime_ns
        source.write_text("def run(): return 1\n", encoding="utf-8")
        changed_mtime_ns = old_mtime_ns + 2_000_000_000
        os.utime(source, ns=(changed_mtime_ns, changed_mtime_ns))
        summary = daemon._incremental_refresh_once_sync()

        assert summary["refreshed"] == 1
        assert daemon.db.get_imports_of("a") == []
        assert daemon.db.get_callers("helper") == []
    finally:
        daemon.db.close()


def test_failed_incremental_parse_preserves_last_good_imports(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("import b\ndef stable(): return b.VALUE\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    try:
        daemon._full_index_sync()
        old_hash = daemon.db.get_file_hash("a.py")
        assert daemon.db.get_imports_of("a") == ["b"]
        daemon._file_mtime_snapshot = _collect_project_file_mtimes(config)

        old_mtime_ns = source.stat().st_mtime_ns
        source.write_text("import b\ndef broken(:\n", encoding="utf-8")
        changed_mtime_ns = old_mtime_ns + 2_000_000_000
        os.utime(source, ns=(changed_mtime_ns, changed_mtime_ns))

        summary = daemon._incremental_refresh_once_sync()

        assert summary["refreshed"] == 0
        assert summary["skipped"] == 1
        assert daemon.db.get_file_hash("a.py") == old_hash
        assert daemon.db.get_imports_of("a") == ["b"]
    finally:
        daemon.db.close()


def test_failed_incremental_refresh_keeps_snapshot_for_retry(tmp_path, monkeypatch):
    source = tmp_path / "a.py"
    source.write_text("def a(): return 1\n", encoding="utf-8")
    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    try:
        old_snapshot = {"a.py": 1}
        daemon._file_mtime_snapshot = old_snapshot.copy()
        monkeypatch.setattr(
            "know.daemon._collect_project_file_mtimes", lambda _config: {"a.py": 2}
        )
        monkeypatch.setattr(
            "know.daemon.refresh_files_if_stale",
            lambda *_args, **_kwargs: {
                "refreshed": 0,
                "removed": 0,
                "skipped": 1,
                "results": [{
                    "updated": False,
                    "file_path": "a.py",
                    "reason": "parse_failed:ParseError",
                }],
            },
        )

        daemon._incremental_refresh_once_sync()
        assert daemon._file_mtime_snapshot == old_snapshot
    finally:
        daemon.db.close()


def test_auto_refresh_loop_continues_after_iteration_exception(tmp_path, monkeypatch):
    import know.daemon as daemon_module

    daemon = KnowDaemon(tmp_path, _config(tmp_path))
    daemon._auto_refresh_interval = 15
    calls = {"refresh": 0, "sleep": 0}

    async def immediate_to_thread(func, *args):
        return func(*args)

    async def bounded_sleep(_seconds):
        calls["sleep"] += 1
        if calls["sleep"] >= 3:
            raise asyncio.CancelledError

    def flaky_refresh():
        calls["refresh"] += 1
        if calls["refresh"] == 1:
            raise OSError("transient")
        return {"refreshed": 0, "removed": 0, "skipped": 0, "results": []}

    monkeypatch.setattr(daemon_module.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(daemon_module.asyncio, "sleep", bounded_sleep)
    monkeypatch.setattr(daemon_module, "_collect_project_file_mtimes", lambda _config: {})
    monkeypatch.setattr(daemon, "_incremental_refresh_once_sync", flaky_refresh)
    try:
        asyncio.run(daemon._auto_refresh_loop())
        assert calls["refresh"] == 2
    finally:
        daemon.db.close()


def test_daemon_read_only_context_does_not_touch_memory_metadata(tmp_path, monkeypatch):
    import know.context_engine as context_module
    import know.knowledge_base as kb_module

    observed = {}

    class FakeEngine:
        def __init__(self, _config):
            pass

        def build_context(self, *_args, **_kwargs):
            return {}

        def format_agent_json(self, _result):
            return "{}"

    class FakeKB:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            observed["closed"] = True

        def get_relevant_context(self, _query, *, max_tokens, touch):
            observed.update(max_tokens=max_tokens, touch=touch)
            return ""

    monkeypatch.setattr(context_module, "ContextEngine", FakeEngine)
    monkeypatch.setattr(kb_module, "KnowledgeBase", FakeKB)
    daemon = KnowDaemon(tmp_path, _config(tmp_path))
    try:
        asyncio.run(daemon._handle_context({
            "query": "read without mutation",
            "budget": 1000,
            "read_only": True,
        }))
        assert observed == {"max_tokens": 100, "touch": False, "closed": True}
    finally:
        daemon.db.close()
