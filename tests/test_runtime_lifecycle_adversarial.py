"""Adversarial regressions for scanner, watcher, and daemon lifecycles."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from know.config import Config


def _config(root: Path) -> Config:
    (root / ".know").mkdir(exist_ok=True)
    config = Config.create_default(root)
    config.root = root
    config.project.name = "runtime-lifecycle-test"
    config.save(root / ".know" / "config.yaml")
    return config


def test_typescript_cap_is_applied_after_test_files_are_deprioritized(tmp_path, monkeypatch):
    from know import scanner as scanner_module
    from know.scanner import CodebaseScanner

    root = tmp_path / "project"
    root.mkdir()
    config = _config(root)
    tests = root / "tests"
    source = root / "src"
    tests.mkdir()
    source.mkdir()
    # Create test files first: raw filesystem order must not decide the cap.
    (tests / "alpha.test.ts").write_text("export const alpha = 1\n")
    (tests / "beta.spec.tsx").write_text("export const beta = 2\n")
    (source / "app.ts").write_text("export const app = 3\n")
    (source / "core.tsx").write_text("export const core = 4\n")

    monkeypatch.setattr(scanner_module, "MAX_TS_FILES", 2)
    scanner = CodebaseScanner(config)
    try:
        selected = {
            str(path.relative_to(root)).replace("\\", "/")
            for path, _language in scanner._discover_files()
        }
    finally:
        scanner.close()

    assert selected == {"src/app.ts", "src/core.tsx"}


def test_scanner_pathspec_construction_emits_no_deprecation_warning(tmp_path):
    from know.scanner import CodebaseScanner

    config = _config(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scanner = CodebaseScanner(config)
        scanner.close()

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_watcher_flushes_trailing_debounce_without_another_event(tmp_path):
    from know.watcher import DocUpdateHandler

    config = _config(tmp_path)
    config.output.watch.debounce_seconds = 0.03
    changed = tmp_path / "src.py"
    changed.write_text("x = 1\n")
    handler = DocUpdateHandler(config)
    triggered = threading.Event()
    handler.last_update = time.time()
    handler._trigger_update_serialized = triggered.set  # type: ignore[method-assign]
    try:
        handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(changed)))
        assert triggered.wait(0.5), "pending trailing update was never flushed"
    finally:
        handler.close()


def test_watcher_close_cancels_pending_trailing_update(tmp_path):
    from know.watcher import DocUpdateHandler

    config = _config(tmp_path)
    config.output.watch.debounce_seconds = 0.05
    changed = tmp_path / "src.py"
    changed.write_text("x = 1\n")
    handler = DocUpdateHandler(config)
    triggered = threading.Event()
    handler.last_update = time.time()
    handler._trigger_update_serialized = triggered.set  # type: ignore[method-assign]
    handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(changed)))
    handler.close()

    assert not triggered.wait(0.15)


def test_watcher_trailing_update_waits_for_slow_scan(tmp_path, monkeypatch):
    from know.watcher import DocUpdateHandler

    config = _config(tmp_path)
    config.output.watch.debounce_seconds = 0.02
    first_path = tmp_path / "first.py"
    second_path = tmp_path / "second.py"
    first_path.write_text("first = 1\n")
    second_path.write_text("second = 2\n")

    state_lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    active_scans = 0
    max_active_scans = 0
    scan_count = 0

    class Scanner:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def scan_files(self, _paths):
            nonlocal active_scans, max_active_scans, scan_count
            with state_lock:
                scan_count += 1
                current_scan = scan_count
                active_scans += 1
                max_active_scans = max(max_active_scans, active_scans)
            try:
                if current_scan == 1:
                    first_started.set()
                    assert release_first.wait(1), "test did not release the first scan"
            finally:
                with state_lock:
                    active_scans -= 1
            if current_scan == 2:
                second_finished.set()
            return {"files": 1, "changed_files": 1}

        def get_structure(self):
            return {}

    class Generator:
        def __init__(self, _config):
            pass

        def generate_readme(self, _structure):
            return None

    monkeypatch.setattr("know.scanner.CodebaseScanner", Scanner)
    monkeypatch.setattr("know.generator.DocGenerator", Generator)

    handler = DocUpdateHandler(config)
    leading_thread = threading.Thread(
        target=handler.on_modified,
        args=(SimpleNamespace(is_directory=False, src_path=str(first_path)),),
    )
    try:
        leading_thread.start()
        assert first_started.wait(0.5)

        # This event is flushed by the trailing timer while the leading scan is
        # deliberately blocked. It must wait rather than overlap that scan.
        handler.on_modified(
            SimpleNamespace(is_directory=False, src_path=str(second_path))
        )
        time.sleep(0.08)
        with state_lock:
            assert scan_count == 1
            assert max_active_scans == 1

        release_first.set()
        leading_thread.join(timeout=1)
        assert not leading_thread.is_alive()
        assert second_finished.wait(0.5)
        with state_lock:
            assert scan_count == 2
            assert max_active_scans == 1
    finally:
        release_first.set()
        leading_thread.join(timeout=1)
        handler.close()


def test_read_only_workflow_uses_compatible_daemon_without_local_writes(tmp_path, monkeypatch):
    from know.cli import cli

    config = _config(tmp_path)
    request = {}
    payload = {
        "query": "billing",
        "session_id": None,
        "read_only": True,
        "daemon_api_version": 2,
        "workflow_mode": "implement",
        "latency_budget_ms": 6000,
        "budgets": {"map_limit": 20, "context_budget": 4000, "deep_budget": 3000},
        "selected_deep_target": None,
        "map": {"results": [], "count": 0, "truncated": False, "tokens": 0},
        "context": {"code": [], "used_tokens": 0},
        "deep": {"error": "no_target"},
        "total_tokens": 0,
        "latency_ms": {"map": 1, "context": 1, "deep": 0, "total": 2},
        "degraded_by_latency": False,
    }

    class _Client:
        def call_sync(self, method, params):
            request.update({"method": method, "params": params})
            return payload

    monkeypatch.setattr("know.cli.agent._get_daemon_client", lambda _config: _Client())
    monkeypatch.setattr(
        "know.cli.agent._get_db_fallback",
        lambda _config: (_ for _ in ()).throw(AssertionError("cold fallback used")),
    )

    with (
        patch("know.memory_capture.capture_workflow_decision") as capture,
        patch("know.stats.StatsTracker.record_workflow") as record_workflow,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "--config", str(tmp_path / ".know" / "config.yaml"),
                "--json", "workflow", "billing", "--json-full", "--read-only",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["read_only"] is True
    assert request["method"] == "workflow"
    assert request["params"]["read_only"] is True
    assert request["params"]["session_id"] is None
    capture.assert_not_called()
    record_workflow.assert_not_called()


def test_failed_initial_index_keeps_cold_refresh_baseline_for_retry(tmp_path, monkeypatch):
    import know.daemon as daemon_module
    from know.daemon import KnowDaemon

    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    observed = {}

    async def run() -> None:
        async def fail_index():
            raise RuntimeError("transient cold-index failure")

        async def no_wait(_seconds):
            return None

        daemon._index_task = asyncio.create_task(fail_index())

        def inspect_baseline():
            observed["baseline"] = dict(daemon._file_mtime_snapshot)
            raise asyncio.CancelledError

        monkeypatch.setattr(daemon, "_incremental_refresh_once_sync", inspect_baseline)
        monkeypatch.setattr(asyncio, "sleep", no_wait)
        await daemon._auto_refresh_loop()

    monkeypatch.setattr(
        daemon_module,
        "_collect_project_file_mtimes",
        lambda _config: {"src/main.py": 101},
    )
    monkeypatch.setattr(daemon.db, "list_indexed_files", lambda: [])
    try:
        asyncio.run(run())
    finally:
        daemon.db.close()

    assert observed["baseline"] == {}


def test_full_index_performs_low_frequency_session_cleanup(tmp_path, monkeypatch):
    import know.daemon as daemon_module
    from know.daemon import KnowDaemon

    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    cleanups = []
    monkeypatch.setattr(daemon.db, "cleanup_expired_sessions", lambda: cleanups.append(True) or 0)
    monkeypatch.setattr(daemon_module, "populate_index", lambda *_args: (0, []))
    monkeypatch.setattr(daemon.db, "compute_importance", lambda: {})

    class _Graph:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def build(self, _modules):
            return None

    monkeypatch.setattr("know.import_graph.ImportGraph", _Graph)
    try:
        daemon._full_index_sync()
    finally:
        daemon.db.close()

    assert cleanups == [True]


def test_pid_for_unusable_socket_is_not_treated_as_running(tmp_path, monkeypatch):
    import know.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "PID_DIR", tmp_path / "pids")
    monkeypatch.setattr(daemon_module, "SOCKET_DIR", tmp_path / "sockets")
    daemon_module.PID_DIR.mkdir()
    daemon_module.SOCKET_DIR.mkdir()
    pf = daemon_module.pid_path(tmp_path)
    pf.write_text(str(os.getpid()))

    assert daemon_module.is_daemon_running(tmp_path) is False


def _install_probe_response(monkeypatch, daemon_module, response) -> None:
    framed = json.dumps(response).encode()
    incoming = bytearray(len(framed).to_bytes(4, "big") + framed)

    class _Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            pass

        def sendall(self, _payload):
            pass

        def recv(self, size):
            chunk = bytes(incoming[:size])
            del incoming[:size]
            return chunk

    monkeypatch.setattr(daemon_module.socket, "socket", lambda *_args, **_kwargs: _Socket())


@pytest.mark.parametrize(
    "response",
    [
        [],
        "running",
        {"jsonrpc": "2.0", "id": 1, "result": []},
        {"jsonrpc": "2.0", "id": 1, "result": "running"},
        {"jsonrpc": "2.0", "id": True, "result": {"running": True, "project": "/tmp"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"running": True, "project": "/tmp"}},
        {"jsonrpc": "1.0", "id": 1, "result": {"running": True, "project": "/tmp"}},
        {"jsonrpc": "2.0", "id": 1, "result": {"running": "yes", "project": "/tmp"}},
        {"jsonrpc": "2.0", "id": 1, "result": {"running": True, "project": "  "}},
    ],
)
def test_probe_daemon_rejects_malformed_or_incompatible_responses(
    tmp_path,
    monkeypatch,
    response,
):
    import know.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "SOCKET_DIR", tmp_path / "sockets")
    daemon_module.SOCKET_DIR.mkdir()
    daemon_module.socket_path(tmp_path).touch()
    _install_probe_response(monkeypatch, daemon_module, response)

    assert daemon_module._probe_daemon(tmp_path) is False


def test_probe_daemon_accepts_matching_well_formed_status(tmp_path, monkeypatch):
    import know.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "SOCKET_DIR", tmp_path / "sockets")
    daemon_module.SOCKET_DIR.mkdir()
    daemon_module.socket_path(tmp_path).touch()
    _install_probe_response(
        monkeypatch,
        daemon_module,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"running": True, "project": str(tmp_path)},
        },
    )

    assert daemon_module._probe_daemon(tmp_path) is True


def test_concurrent_ensure_daemon_starts_only_one_process(tmp_path, monkeypatch):
    import know.daemon as daemon_module

    config = _config(tmp_path)
    monkeypatch.setattr(daemon_module, "PID_DIR", tmp_path / "pids")
    monkeypatch.setattr(daemon_module, "SOCKET_DIR", tmp_path / "sockets")
    state = {"running": False, "starts": 0}
    state_lock = threading.Lock()

    def fake_running(_root):
        with state_lock:
            running = state["running"]
        if not running:
            time.sleep(0.04)
        return running

    def fake_start(root, _config):
        with state_lock:
            state["starts"] += 1
            state["running"] = True
        daemon_module.socket_path(root).parent.mkdir(parents=True, exist_ok=True)
        daemon_module.socket_path(root).touch()

    monkeypatch.setattr(daemon_module, "is_daemon_running", fake_running)
    monkeypatch.setattr(daemon_module, "_start_daemon_background", fake_start)
    with ThreadPoolExecutor(max_workers=2) as pool:
        clients = list(pool.map(lambda _item: daemon_module.ensure_daemon(tmp_path, config), range(2)))

    assert state["starts"] == 1
    assert all(client.root == tmp_path for client in clients)


def test_ensure_daemon_never_replaces_a_live_unresponsive_process(tmp_path, monkeypatch):
    import know.daemon as daemon_module

    config = _config(tmp_path)
    monkeypatch.setattr(daemon_module, "PID_DIR", tmp_path / "pids")
    monkeypatch.setattr(daemon_module, "SOCKET_DIR", tmp_path / "sockets")
    daemon_module.PID_DIR.mkdir()
    daemon_module.SOCKET_DIR.mkdir()
    daemon_module.pid_path(tmp_path).write_text(str(os.getpid()))
    monkeypatch.setattr(daemon_module, "_probe_daemon", lambda _root: False)
    monkeypatch.setattr(daemon_module.time, "sleep", lambda _seconds: None)
    starts = []
    monkeypatch.setattr(
        daemon_module,
        "_start_daemon_background",
        lambda *_args: starts.append(True),
    )

    with pytest.raises(RuntimeError, match="refusing to replace it"):
        daemon_module.ensure_daemon(tmp_path, config)

    assert starts == []


def test_background_spawn_is_reaped_without_blocking_caller(tmp_path, monkeypatch):
    import subprocess
    import know.daemon as daemon_module

    waited = threading.Event()
    caller_thread = threading.get_ident()
    waiter_threads = []

    class _Process:
        pid = 12345

        def wait(self):
            waiter_threads.append(threading.get_ident())
            waited.set()
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: _Process())
    daemon_module._start_daemon_background(tmp_path, _config(tmp_path))

    assert waited.wait(0.5)
    assert len(waiter_threads) == 1
    assert waiter_threads[0] != caller_thread


def test_next_file_ranking_is_invariant_to_raw_score_scale():
    from know.cli.agent import _rank_file_candidates

    rows = [
        {"file_path": "backend/auth.py", "score": 0.016},
        {"file_path": "web/components/auth.tsx", "score": 0.015},
    ]
    scaled_rows = [
        {"file_path": row["file_path"], "score": row["score"] * 10000 + 77}
        for row in rows
    ]

    first = _rank_file_candidates(rows, "frontend")
    scaled = _rank_file_candidates(scaled_rows, "frontend")
    assert list(first) == list(scaled)
    assert first == scaled


def test_next_file_ranking_ignores_nonfinite_duplicate_scores():
    from know.cli.agent import _rank_file_candidates

    ranked = _rank_file_candidates(
        [
            {"file_path": "b.py", "score": 0.8},
            {"file_path": "a.py", "score": float("nan")},
            {"file_path": "a.py", "score": 0.9},
            {"file_path": "c.py", "score": float("inf")},
        ],
        "mixed",
    )

    assert list(ranked) == ["a.py", "b.py", "c.py"]


def test_daemon_db_close_releases_connections_created_by_worker_threads(tmp_path):
    from know.daemon_db import DaemonDB

    db = DaemonDB(tmp_path)
    barrier = threading.Barrier(3)

    def open_worker_connection(_index):
        connection = db._get_conn()
        barrier.wait(timeout=2)
        connection.execute("SELECT 1").fetchone()
        return connection

    with ThreadPoolExecutor(max_workers=3) as pool:
        connections = list(pool.map(open_worker_connection, range(3)))

    assert len({id(connection) for connection in connections}) == 3
    db.close()

    for connection in connections:
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError("DaemonDB.close() left a worker-thread connection open")


def test_daemon_db_prunes_connections_owned_by_finished_threads(tmp_path):
    from know.daemon_db import DaemonDB

    db = DaemonDB(tmp_path)
    worker_connections = []

    def open_connection():
        connection = db._get_conn()
        connection.execute("SELECT 1").fetchone()
        worker_connections.append(connection)

    try:
        for _ in range(40):
            worker = threading.Thread(target=open_connection)
            worker.start()
            worker.join(timeout=1)
            assert not worker.is_alive()

        # The main-thread connection plus at most the most recently retired
        # worker connection remain. Earlier dead-thread handles were closed.
        assert len(db._connections) <= 2
        for connection in worker_connections[:-1]:
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
    finally:
        db.close()

    for connection in worker_connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_daemon_db_thread_churn_stays_within_low_file_descriptor_limit(tmp_path):
    script = """
import resource
import threading
from pathlib import Path

from know.daemon_db import DaemonDB

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
limit = min(32, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
db = DaemonDB(Path.cwd() / "project")
errors = []

def query():
    try:
        db._get_conn().execute("SELECT 1").fetchone()
    except BaseException as error:
        errors.append(repr(error))

for _ in range(100):
    worker = threading.Thread(target=query)
    worker.start()
    worker.join()

if errors:
    raise AssertionError(errors)
if len(db._connections) > 2:
    raise AssertionError(f"retained {len(db._connections)} connections")
db.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_shutdown_prevents_cancelled_index_worker_from_reopening_database(
    tmp_path, monkeypatch,
):
    from know.daemon import KnowDaemon

    config = _config(tmp_path)
    daemon = KnowDaemon(tmp_path, config)
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    outcomes = []

    def blocked_index():
        worker_started.set()
        assert release_worker.wait(1), "test did not release indexing worker"
        try:
            daemon.db.store_memory("late-write", "must not persist")
        except RuntimeError as error:
            outcomes.append(str(error))
        else:
            outcomes.append("wrote")
        finally:
            worker_finished.set()
        return 1

    monkeypatch.setattr(daemon, "_full_index_sync", blocked_index)

    async def exercise_shutdown():
        daemon._index_task = asyncio.create_task(daemon._full_index())
        assert await asyncio.to_thread(worker_started.wait, 0.5)
        daemon._shutdown()
        release_worker.set()
        assert await asyncio.to_thread(worker_finished.wait, 0.5)
        with pytest.raises(asyncio.CancelledError):
            await daemon._index_task

    try:
        asyncio.run(exercise_shutdown())
    finally:
        release_worker.set()
        daemon.db.close()

    assert outcomes == ["DaemonDB is closed"]
    with pytest.raises(RuntimeError, match="closed"):
        daemon.db._get_conn()
    with closing(sqlite3.connect(daemon.db.db_path)) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM memories WHERE id = 'late-write'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("method", "field", "maximum", "base_params", "invalid_values"),
    [
        ("search", "limit", 100, {"query": "target"}, None),
        ("recall", "limit", 100, {"query": "target"}, None),
        ("callers", "limit", 100, {"function_name": "target"}, None),
        ("callees", "limit", 100, {"chunk_name": "target"}, None),
        ("map", "limit", 100, {"query": "target"}, None),
        ("context", "budget", 100_000, {"query": "target"}, None),
        ("deep", "budget", 100_000, {"name": "target"}, None),
        ("workflow", "map_limit", 100, {"query": "target"}, None),
        ("workflow", "context_budget", 100_000, {"query": "target"}, None),
        (
            "workflow",
            "deep_budget",
            100_000,
            {"query": "target"},
            (True, "1", -1, 100_001),
        ),
    ],
)
def test_daemon_rejects_malformed_or_unbounded_resource_parameters(
    tmp_path, method, field, maximum, base_params, invalid_values,
):
    from know.daemon import KnowDaemon

    daemon = KnowDaemon(tmp_path, _config(tmp_path))
    try:
        values = invalid_values or (True, "1", 0, -1, maximum + 1)
        for invalid in values:
            params = {**base_params, field: invalid}
            with pytest.raises(ValueError):
                asyncio.run(daemon._dispatch(method, params))
    finally:
        daemon.db.close()


def test_explore_workflow_retains_zero_deep_budget_skip_sentinel(
    tmp_path, monkeypatch,
):
    from know.daemon import KnowDaemon

    daemon = KnowDaemon(tmp_path, _config(tmp_path))
    monkeypatch.setattr(daemon.db, "search_signatures", lambda *_args: [])
    try:
        result = asyncio.run(daemon._handle_workflow({
            "query": "target",
            "mode": "explore",
            "deep_budget": 0,
            "max_latency_ms": 1,
            "read_only": True,
        }))
    finally:
        daemon.db.close()

    assert result["budgets"]["deep_budget"] == 0
    assert result["deep"]["error"] == "skipped_by_mode"


def test_cli_does_not_retain_closed_runner_stream_in_logger(tmp_path):
    from know.cli import cli

    config = _config(tmp_path)
    runner = CliRunner()
    args = ["--config", str(tmp_path / ".know" / "config.yaml"), "--json", "commands"]

    first = runner.invoke(cli, args)
    assert first.exit_code == 0, first.output

    know_logger = logging.getLogger("know")
    assert not [
        handler
        for handler in know_logger.handlers
        if getattr(getattr(handler, "stream", None), "closed", False)
    ]
    know_logger.error("post-command diagnostic")

    second = runner.invoke(cli, args)
    assert second.exit_code == 0, second.output
