"""Background daemon for maintaining hot indexes.

Lifecycle:
- Auto-started on first CLI call (no explicit `know start` needed)
- Listens on Unix socket at ~/.know/sockets/<project-hash>.sock
- PID file at ~/.know/pids/<project-hash>.pid
- Auto-shutdown after 30 min idle
- Crash recovery: stale PID detection, SQLite WAL mode

Protocol: JSON-RPC 2.0 over Unix domain socket
"""

import asyncio
import fcntl
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Iterator, List, Tuple

import xxhash

from know.config import Config
from know.daemon_db import DaemonDB
from know.logger import get_logger
from know.parsers import EXTENSION_TO_LANGUAGE

logger = get_logger()

IDLE_TIMEOUT = 1800  # 30 minutes
SOCKET_DIR = Path.home() / ".know" / "sockets"
PID_DIR = Path.home() / ".know" / "pids"
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_CHUNK_BODY_CHARS = 5000
DAEMON_API_VERSION = 2
MAX_RPC_RESULTS = 100
MAX_RPC_TOKEN_BUDGET = 100_000


def _bounded_rpc_int(name: str, value: Any, *, maximum: int) -> int:
    """Validate resource-shaping JSON-RPC integers without coercion."""
    # bool is an int subclass, while strings accepted by int() hide malformed
    # JSON clients. Both must be rejected at the protocol boundary.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value

_INDEX_LOCKS: Dict[str, threading.Lock] = {}
_INDEX_LOCKS_GUARD = threading.Lock()


@contextmanager
def _serialized_index(root: Path) -> Iterator[None]:
    """Serialize source validation and chunk publication per project."""
    key = str(root.resolve())
    with _INDEX_LOCKS_GUARD:
        thread_lock = _INDEX_LOCKS.setdefault(key, threading.Lock())

    with thread_lock:
        lock_path = root / ".know" / "index.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _collect_project_file_mtimes(config: Config) -> Dict[str, int]:
    """Collect mtime_ns for source files included by scanner filters."""
    from know.scanner import CodebaseScanner

    mtimes: Dict[str, int] = {}
    with CodebaseScanner(config) as scanner:
        for path, _lang in scanner._discover_files():
            try:
                rel = str(path.relative_to(config.root)).replace("\\", "/")
                mtimes[rel] = path.stat().st_mtime_ns
            except (FileNotFoundError, OSError, ValueError):
                continue
    return mtimes


def _diff_file_mtimes(previous: Dict[str, int], current: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """Return (changed_or_new_files, removed_files)."""
    changed = sorted([p for p, m in current.items() if previous.get(p) != m])
    removed = sorted([p for p in previous if p not in current])
    return changed, removed


def _extract_body(
    lines: list | None,
    content: str,
    start: int,
    end: int,
    fallback: str,
) -> tuple:
    """Extract source body from line range, with lazy line splitting.

    Returns (lines_list, body_text). The lines_list is passed back so the
    caller can reuse it across multiple calls without re-splitting.
    """
    if end > 0 and end >= start:
        if lines is None:
            lines = content.split("\n")
        body = "\n".join(lines[start - 1 : end])
        if len(body) > MAX_CHUNK_BODY_CHARS:
            cut = body.rfind("\n", 0, MAX_CHUNK_BODY_CHARS)
            if cut > 0:
                body = body[:cut] + "\n# ... truncated"
            else:
                body = body[:MAX_CHUNK_BODY_CHARS] + "\n# ... truncated"
        return lines, body
    return lines, fallback


def _build_chunks_from_module(content: str, mod_info) -> list[dict]:
    """Build chunk rows for a parsed module."""
    chunks: list[dict] = []
    lines = None

    for func in mod_info.functions:
        start = func.line_number
        end = func.end_line if func.end_line >= start else start
        chunk_type = "constant" if "constant" in func.decorators else (
            "method" if func.is_method else "function"
        )
        fallback = f"{func.signature}\n{func.docstring or ''}"
        lines, body = _extract_body(lines, content, start, end, fallback)

        chunks.append({
            "name": func.name,
            "type": chunk_type,
            "start_line": start,
            "end_line": end,
            "signature": func.signature,
            "body": body,
        })

    for cls in mod_info.classes:
        start = cls.line_number
        end = cls.end_line if cls.end_line >= start else start
        fallback = f"class {cls.name}\n{cls.docstring or ''}"
        lines, body = _extract_body(lines, content, start, end, fallback)

        chunks.append({
            "name": cls.name,
            "type": "class",
            "start_line": start,
            "end_line": end,
            "signature": cls.name,
            "body": body,
        })

        # Python parser stores class methods on cls.methods (not module.functions).
        for method in getattr(cls, "methods", []) or []:
            m_start = method.line_number
            m_end = method.end_line if method.end_line >= m_start else m_start
            method_sig = method.signature or method.name
            if not method_sig.startswith(f"{cls.name}."):
                method_sig = f"{cls.name}.{method_sig}"
            fallback = f"{method_sig}\n{method.docstring or ''}"
            lines, method_body = _extract_body(lines, content, m_start, m_end, fallback)
            chunks.append({
                "name": f"{cls.name}.{method.name}",
                "type": "method",
                "start_line": m_start,
                "end_line": m_end,
                "signature": method_sig,
                "body": method_body,
            })

    if not chunks:
        chunks.append({
            "name": mod_info.name,
            "type": "module",
            "start_line": 1,
            "end_line": content.count("\n") + 1,
            "signature": mod_info.name,
            "body": content[:MAX_CHUNK_BODY_CHARS],
        })

    return chunks


def refresh_file_if_stale(
    root: Path,
    config: Config,
    db: DaemonDB,
    file_path: str | Path,
    *,
    force: bool = False,
    remove_missing: bool = False,
) -> Dict[str, Any]:
    """Re-index one file as an atomic, generation-ordered operation."""
    with _serialized_index(root):
        return _refresh_file_if_stale_locked(
            root,
            config,
            db,
            file_path,
            force=force,
            remove_missing=remove_missing,
        )


def _refresh_file_if_stale_locked(
    root: Path,
    config: Config,
    db: DaemonDB,
    file_path: str | Path,
    *,
    force: bool = False,
    remove_missing: bool = False,
) -> Dict[str, Any]:
    """Re-index a single file when missing or stale.

    Returns metadata with keys: updated(bool), removed(bool), file_path, reason,
    and optional chunks/call_refs counts.
    """
    from know.index import read_source_snapshot, source_snapshot_matches
    from know.parsers import ParserFactory

    candidate = Path(file_path)
    resolved_root = root.resolve()
    abs_path = candidate if candidate.is_absolute() else (resolved_root / candidate)
    abs_path = abs_path.resolve()

    try:
        rel_path = str(abs_path.relative_to(resolved_root)).replace("\\", "/")
    except ValueError:
        # Ignore paths outside project root
        return {
            "updated": False,
            "removed": False,
            "file_path": str(file_path),
            "reason": "outside_project_root",
        }

    if not abs_path.exists():
        if remove_missing:
            db.remove_file(rel_path)
            return {
                "updated": True,
                "removed": True,
                "file_path": rel_path,
                "reason": "file_missing_removed",
            }
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "file_missing_skipped",
        }

    lang = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), "")
    if not lang:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "unsupported_extension",
        }

    before = read_source_snapshot(abs_path)
    if before is None:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "unstable_snapshot",
        }
    content = before["content"]
    content_hash = before["content_hash"]
    stored_hash = db.get_file_hash(rel_path)
    if stored_hash == content_hash and not force:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "up_to_date",
        }

    parser = ParserFactory.get_parser_for_file(abs_path)
    if parser is None:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "no_parser",
        }

    try:
        mod_info = parser.parse(abs_path, resolved_root)
    except Exception as e:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": f"parse_failed:{e.__class__.__name__}",
        }

    after = read_source_snapshot(abs_path)
    if after is None or not source_snapshot_matches(before, after):
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": "changed_during_parse",
        }

    chunks = _build_chunks_from_module(content, mod_info)
    call_refs = []
    try:
        if hasattr(parser, "extract_call_refs"):
            call_refs = parser.extract_call_refs(content, mod_info) or []
    except Exception as e:
        logger.debug(f"Call extraction failed for {rel_path}: {e}")
        call_refs = []

    with db.batch():
        db.upsert_chunks(rel_path, lang, chunks)
        db.update_file_index(rel_path, content_hash, lang, len(chunks))
        # Always replace symbol refs so stale rows are removed on file edits.
        db.upsert_symbol_refs(rel_path, call_refs)

    return {
        "updated": True,
        "removed": False,
        "file_path": rel_path,
        "reason": "reindexed",
        "chunks": len(chunks),
        "call_refs": len(call_refs),
    }


def refresh_files_if_stale(
    root: Path,
    config: Config,
    db: DaemonDB,
    file_paths: Iterable[str | Path],
    *,
    force: bool = False,
    remove_missing: bool = False,
) -> Dict[str, Any]:
    """Refresh multiple files and return an aggregate summary."""
    refreshed = 0
    removed = 0
    skipped = 0
    results = []
    for fp in file_paths:
        result = refresh_file_if_stale(
            root, config, db, fp, force=force, remove_missing=remove_missing,
        )
        results.append(result)
        if result.get("updated"):
            refreshed += 1
            if result.get("removed"):
                removed += 1
        else:
            skipped += 1
    return {
        "refreshed": refreshed,
        "removed": removed,
        "skipped": skipped,
        "results": results,
    }


def populate_index(root: Path, config: Config, db: DaemonDB) -> tuple:
    """Populate chunks as an atomic, generation-ordered operation."""
    with _serialized_index(root):
        return _populate_index_locked(root, config, db)


def _populate_index_locked(root: Path, config: Config, db: DaemonDB) -> tuple:
    """Populate the daemon DB with code chunks (standalone, no daemon needed).

    Called by the context engine on first use when the DB is empty.
    Also used by KnowDaemon._full_index_sync().

    Uses scanner.modules (ModuleInfo objects) directly to avoid re-parsing
    every file.  The scanner already parsed each file during scan(); we only
    need to read the raw content for body extraction and content hashing.

    Returns (file_count, modules_list) so callers can build import graphs.
    """
    from know.scanner import CodebaseScanner

    with CodebaseScanner(config) as scanner:
        scanner.scan()
        modules = list(scanner.modules)  # List[ModuleInfo] — already parsed
        # Discovery scope, not successful parses, is authoritative for deletion.
        # A transient syntax/read error must not purge the last known-good index.
        allowed_paths = set(scanner.discovered_paths)
        source_snapshots = dict(scanner.source_snapshots)
    count = 0
    purged = 0

    with db.batch():
        # Purge previously indexed files that are no longer part of scanner scope
        # (for example old .venv/site-packages entries from prior versions).
        for existing in db.list_indexed_files():
            normalized = str(existing).replace("\\", "/")
            if normalized not in allowed_paths:
                db.remove_file(existing)
                purged += 1

        for mod_info in modules:
            path_str = str(mod_info.path)
            abs_path = root / path_str

            if not abs_path.exists():
                continue

            lang = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), "")
            if not lang:
                continue

            snapshot = source_snapshots.get(path_str.replace("\\", "/"))
            if snapshot is None:
                logger.debug(f"No coherent parse snapshot for {path_str}; deferring it")
                continue
            content = snapshot["content"]
            content_hash = snapshot["content_hash"]
            stored_hash = db.get_file_hash(path_str)

            if stored_hash == content_hash:
                continue

            chunks = _build_chunks_from_module(content, mod_info)

            db.upsert_chunks(path_str, lang, chunks)
            db.update_file_index(path_str, content_hash, lang, len(chunks))

            # Extract call references for symbol_refs table. Always replace,
            # including zero refs, so edits cannot leave phantom callers.
            call_refs = []
            try:
                from know.parsers import ParserFactory
                parser = ParserFactory.get_parser_for_file(abs_path)
                if parser and hasattr(parser, "extract_call_refs"):
                    call_refs = parser.extract_call_refs(content, mod_info) or []
            except Exception as e:
                logger.debug(f"Call extraction failed for {path_str}: {e}")
            db.upsert_symbol_refs(path_str, call_refs)

            count += 1

    if purged:
        logger.info(f"Purged {purged} out-of-scope indexed files")

    return count, modules


async def write_framed_message(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write a length-prefixed message (4-byte big-endian header)."""
    length = len(data)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})")
    writer.write(struct.pack(">I", length))
    writer.write(data)
    await writer.drain()


async def read_framed_message(reader: asyncio.StreamReader) -> Optional[bytes]:
    """Read a length-prefixed message. Returns None on clean disconnect."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack(">I", header)[0]
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})")
    try:
        data = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None
    return data


def _project_hash(root: Path) -> str:
    """Deterministic short hash for a project root."""
    return xxhash.xxh64(str(root.resolve()).encode()).hexdigest()[:12]


def socket_path(root: Path) -> Path:
    """Get the Unix socket path for a project."""
    return SOCKET_DIR / f"{_project_hash(root)}.sock"


def pid_path(root: Path) -> Path:
    """Get the PID file path for a project."""
    return PID_DIR / f"{_project_hash(root)}.pid"


def _probe_daemon(root: Path, timeout: float = 0.25) -> bool:
    """Validate that the project socket answers a usable status request."""
    sock_path = socket_path(root)
    if not sock_path.exists():
        return False

    request = {
        "jsonrpc": "2.0",
        "method": "status",
        "params": {},
        "id": 1,
    }
    payload = json.dumps(request).encode()

    def _recv_exact(conn: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            part = conn.recv(size - len(chunks))
            if not part:
                raise ConnectionError("daemon closed status probe")
            chunks.extend(part)
        return bytes(chunks)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(sock_path))
            conn.sendall(struct.pack(">I", len(payload)) + payload)
            header = _recv_exact(conn, 4)
            length = struct.unpack(">I", header)[0]
            if length <= 0 or length > MAX_MESSAGE_SIZE:
                return False
            response = json.loads(_recv_exact(conn, length).decode())

        if not isinstance(response, dict):
            return False
        response_id = response.get("id")
        if (
            response.get("jsonrpc") != "2.0"
            or type(response_id) is not int
            or response_id != request["id"]
            or "error" in response
        ):
            return False

        result = response.get("result")
        if not isinstance(result, dict):
            return False
        project = result.get("project")
        if (
            result.get("running") is not True
            or not isinstance(project, str)
            or not project.strip()
        ):
            return False
        return Path(project).resolve() == root.resolve()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _live_daemon_pid(root: Path) -> Optional[int]:
    """Return the recorded PID when it is alive, cleaning only stale state."""
    pf = pid_path(root)
    if not pf.exists():
        return None
    try:
        stored_pid = int(pf.read_text().strip())
        os.kill(stored_pid, 0)  # Check if process exists
    except (ValueError, OSError):
        # Stale PID file — clean up
        pf.unlink(missing_ok=True)
        socket_path(root).unlink(missing_ok=True)
        return None
    return stored_pid


def is_daemon_running(root: Path) -> bool:
    """Check for a live process *and* a usable project-specific daemon."""
    if _live_daemon_pid(root) is None:
        return False
    return _probe_daemon(root)


class KnowDaemon:
    """Background daemon that maintains hot indexes for a project."""

    def __init__(self, project_root: Path, config: Config):
        self.root = project_root
        self.config = config
        self.db = DaemonDB(project_root)
        self._started_at = time.time()
        self._last_activity = time.time()
        self._server: Optional[asyncio.AbstractServer] = None
        self._index_task: Optional[asyncio.Task] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._file_mtime_snapshot: Dict[str, int] = {}
        self._auto_refresh_explicit = "KNOW_DAEMON_AUTO_REFRESH" in os.environ
        self._auto_refresh_enabled = _env_bool("KNOW_DAEMON_AUTO_REFRESH", True)
        self._auto_refresh_interval = _env_int("KNOW_DAEMON_REFRESH_INTERVAL", 60, 15)
        self._auto_refresh_max_files = _env_int("KNOW_DAEMON_AUTO_REFRESH_MAX_FILES", 2500, 100)

    async def serve(self):
        """Main event loop: listen on socket, handle JSON-RPC requests."""
        sock = socket_path(self.root)
        sock.parent.mkdir(parents=True, exist_ok=True)
        sock.unlink(missing_ok=True)

        # Write PID file
        pf = pid_path(self.root)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(os.getpid()))

        # Start server FIRST — accept connections immediately
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=str(sock)
        )
        os.chmod(str(sock), 0o600)
        logger.info(f"Daemon listening on {sock}")

        # Index in background — queries use stale/cached data until done
        self._index_task = asyncio.create_task(self._full_index())
        if self._auto_refresh_enabled:
            self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

        # Set up idle timeout check
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                if time.time() - self._last_activity > IDLE_TIMEOUT:
                    logger.info("Idle timeout reached, shutting down")
                    break
        finally:
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
        if self._index_task and not self._index_task.done():
            self._index_task.cancel()
        if self._server:
            self._server.close()
        socket_path(self.root).unlink(missing_ok=True)
        pid_path(self.root).unlink(missing_ok=True)
        self.db.close()

    async def _auto_refresh_loop(self):
        """Background loop: keep index fresh for changed/deleted files."""
        try:
            initial_index_succeeded = True
            if self._index_task:
                try:
                    await self._index_task
                except Exception as e:
                    initial_index_succeeded = False
                    logger.debug(f"Initial index task failed before auto-refresh: {e}")

            current_snapshot = await asyncio.to_thread(
                _collect_project_file_mtimes, self.config,
            )
            # A failed cold index must make every discovered source look new on
            # the first refresh pass. Warming the baseline here would silently
            # suppress the retry until files changed again.
            self._file_mtime_snapshot = (
                current_snapshot if initial_index_succeeded else {}
            )
            # One-time hygiene: purge indexed files outside current scanner scope
            # (for example old virtualenv/site-packages artifacts).
            try:
                allowed = {str(p).replace("\\", "/") for p in current_snapshot.keys()}
                indexed = self.db.list_indexed_files()
                out_of_scope = [fp for fp in indexed if str(fp).replace("\\", "/") not in allowed]
                if out_of_scope:
                    with self.db.batch():
                        for fp in out_of_scope:
                            self.db.remove_file(fp)
                    logger.info("Auto-refresh startup purge removed %s out-of-scope files", len(out_of_scope))
            except Exception as e:
                logger.debug(f"Startup out-of-scope purge skipped: {e}")

            try:
                files = int(self.db.get_stats().get("files", 0))
            except Exception:
                files = 0
            if (
                not self._auto_refresh_explicit
                and files > self._auto_refresh_max_files
            ):
                logger.info(
                    "Auto-refresh suspended for large repo (%s files > %s). "
                    "Set KNOW_DAEMON_AUTO_REFRESH=1 to force-enable.",
                    files, self._auto_refresh_max_files,
                )
                return

            while True:
                await asyncio.sleep(self._auto_refresh_interval)
                try:
                    summary = await asyncio.to_thread(self._incremental_refresh_once_sync)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    # A transient filesystem/parser/SQLite failure must not
                    # permanently kill the daemon's refresh task.
                    logger.debug(f"Auto-refresh iteration failed; will retry: {e}")
                    continue
                if summary.get("refreshed", 0) or summary.get("removed", 0):
                    logger.info(
                        "Auto-refresh: refreshed=%s removed=%s skipped=%s",
                        summary.get("refreshed", 0),
                        summary.get("removed", 0),
                        summary.get("skipped", 0),
                    )
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.debug(f"Auto-refresh loop stopped: {e}")

    def _incremental_refresh_once_sync(self) -> Dict[str, Any]:
        """Perform one incremental refresh pass."""
        current = _collect_project_file_mtimes(self.config)
        changed, removed = _diff_file_mtimes(self._file_mtime_snapshot, current)

        if not changed and not removed:
            return {"refreshed": 0, "removed": 0, "skipped": 0, "results": []}

        paths = changed + removed
        summary = refresh_files_if_stale(
            self.root, self.config, self.db, paths, remove_missing=True,
        )

        # Import edges are part of the index contract, not optional derived
        # state. Rebuild authoritatively after every changed batch so removed
        # imports and newly added imports are visible immediately.
        from know.import_graph import ImportGraph
        with ImportGraph(self.config) as graph:
            graph.build()

        # Recomputing full graph importance on every tiny edit is expensive.
        # Refresh it only for larger batches or file removals.
        refresh_count = int(summary.get("refreshed", 0) or 0)
        removed_count = int(summary.get("removed", 0) or 0)
        if removed_count > 0 or refresh_count >= 10:
            try:
                self.db.compute_importance()
            except Exception as e:
                logger.debug(f"Importance recompute after auto-refresh failed: {e}")

        # Advance only successful paths. Parse/read failures retain their old
        # mtime (or remain absent) so the next iteration retries them.
        retry_paths = {
            str(result.get("file_path", ""))
            for result in summary.get("results", [])
            if not result.get("updated")
            and result.get("reason") != "up_to_date"
        }
        next_snapshot = dict(current)
        for path in retry_paths:
            if path in self._file_mtime_snapshot:
                next_snapshot[path] = self._file_mtime_snapshot[path]
            else:
                next_snapshot.pop(path, None)
        self._file_mtime_snapshot = next_snapshot

        return summary

    async def _handle_connection(self, reader: asyncio.StreamReader,
                                  writer: asyncio.StreamWriter):
        """Handle a single JSON-RPC connection with length-prefixed framing."""
        self._last_activity = time.time()
        try:
            data = await read_framed_message(reader)
            if data is None:
                return

            request = json.loads(data.decode())
            method = request.get("method", "")
            params = request.get("params", {})
            req_id = request.get("id", 1)

            try:
                result = await self._dispatch(method, params)
                response = {"jsonrpc": "2.0", "result": result, "id": req_id}
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": str(e)},
                    "id": req_id,
                }

            await write_framed_message(writer, json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, method: str, params: dict) -> Any:
        """Route JSON-RPC methods to handlers."""
        handlers = {
            "search": self._handle_search,
            "context": self._handle_context,
            "workflow": self._handle_workflow,
            "signatures": self._handle_signatures,
            "related": self._handle_related,
            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "reindex": self._handle_reindex,
            "status": self._handle_status,
            "callers": self._handle_callers,
            "callees": self._handle_callees,
            "map": self._handle_map,
            "deep": self._handle_deep,
        }
        handler = handlers.get(method)
        if not handler:
            raise ValueError(f"Unknown method: {method}")
        return await handler(params)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    async def _handle_search(self, params: dict) -> dict:
        """BM25 search over code chunks."""
        query = params.get("query", "")
        limit = _bounded_rpc_int(
            "limit", params.get("limit", 20), maximum=MAX_RPC_RESULTS,
        )
        results = self.db.search_chunks(query, limit)
        return {"results": results, "count": len(results)}

    async def _handle_context(self, params: dict) -> dict:
        """Build full v3 context for a query within a token budget."""
        query = params.get("query", "")
        budget = _bounded_rpc_int(
            "budget",
            params.get("budget", 8000),
            maximum=MAX_RPC_TOKEN_BUDGET,
        )
        include_tests = bool(params.get("include_tests", True))
        include_imports = bool(params.get("include_imports", True))
        include_patterns = params.get("include_patterns")
        exclude_patterns = params.get("exclude_patterns")
        chunk_types = params.get("chunk_types")
        session_id = params.get("session_id")
        include_markdown = bool(params.get("include_markdown", False))
        retrieval_profile = str(params.get("retrieval_profile", "balanced") or "balanced")
        semantic_max_ms = params.get("semantic_max_ms")
        read_only = bool(params.get("read_only", False))
        if semantic_max_ms is not None:
            semantic_max_ms = int(semantic_max_ms)

        from know.context_engine import ContextEngine

        engine = ContextEngine(self.config)
        result = engine.build_context(
            query,
            budget=budget,
            include_tests=include_tests,
            include_imports=include_imports,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            chunk_types=chunk_types,
            session_id=session_id,
            retrieval_profile=retrieval_profile,
            semantic_max_ms=semantic_max_ms,
            db=self.db,
        )

        # Inject relevant memories into context (same behavior as CLI fallback).
        try:
            from know.knowledge_base import KnowledgeBase
            with KnowledgeBase(self.config) as kb:
                memory_ctx = kb.get_relevant_context(
                    query,
                    max_tokens=min(500, budget // 10),
                    touch=not read_only,
                )
            if memory_ctx:
                result["memories_context"] = memory_ctx
        except Exception as e:
            logger.debug(f"Memory injection into daemon context failed: {e}")

        payload = json.loads(engine.format_agent_json(result))
        if include_markdown:
            payload["markdown"] = engine.format_markdown(result)
        return payload

    @staticmethod
    def _pick_deep_target(
        context_payload: Dict[str, Any],
        map_results: List[Dict[str, Any]],
    ) -> Optional[Tuple[str, Optional[str]]]:
        """Pick best deep target from context first, map as fallback.

        Returns (target_name, target_file_path|None).
        """
        code = context_payload.get("code", []) if isinstance(context_payload, dict) else []
        for preferred_type in ("function", "method", "class"):
            for chunk in code:
                if (chunk.get("type") or "").lower() == preferred_type and chunk.get("name"):
                    return str(chunk["name"]), chunk.get("file")

        for preferred_type in ("function", "method", "class", "constant", "module"):
            for row in map_results:
                if (row.get("chunk_type") or "").lower() == preferred_type:
                    name = row.get("chunk_name") or row.get("signature")
                    if name:
                        return str(name), row.get("file_path")

        return None

    async def _handle_workflow(self, params: dict) -> dict:
        """Single-call workflow: map -> context -> deep in one daemon request."""
        workflow_started = time.monotonic()
        query = params.get("query", "")
        mode = str(params.get("mode", "implement") or "implement").strip().lower()
        if mode not in {"explore", "implement", "thorough"}:
            mode = "implement"
        defaults = {
            "explore": {
                "map_limit": 30,
                "context_budget": 3500,
                "deep_budget": 0,
                "retrieval_profile": "fast",
                "max_latency_ms": 2500,
            },
            "implement": {
                "map_limit": 20,
                "context_budget": 4000,
                "deep_budget": 3000,
                "retrieval_profile": "balanced",
                "max_latency_ms": 6000,
            },
            "thorough": {
                "map_limit": 30,
                "context_budget": 6000,
                "deep_budget": 4500,
                "retrieval_profile": "thorough",
                "max_latency_ms": 15000,
            },
        }[mode]

        map_limit = _bounded_rpc_int(
            "map_limit",
            params.get("map_limit", defaults["map_limit"]),
            maximum=MAX_RPC_RESULTS,
        )
        context_budget = _bounded_rpc_int(
            "context_budget",
            params.get("context_budget", defaults["context_budget"]),
            maximum=MAX_RPC_TOKEN_BUDGET,
        )
        if "deep_budget" in params:
            requested_deep_budget = params["deep_budget"]
            if (
                mode == "explore"
                and not isinstance(requested_deep_budget, bool)
                and requested_deep_budget == 0
            ):
                # Existing workflow clients send the documented explore-mode
                # sentinel explicitly. It disables work rather than shaping a
                # query, so retain it while validating every active budget.
                deep_budget = 0
            else:
                deep_budget = _bounded_rpc_int(
                    "deep_budget",
                    requested_deep_budget,
                    maximum=MAX_RPC_TOKEN_BUDGET,
                )
        else:
            # Explore mode intentionally uses zero as an internal sentinel to
            # skip deep expansion; explicit RPC values remain strictly positive.
            deep_budget = defaults["deep_budget"]
        include_tests = bool(params.get("include_tests", False))
        include_imports = bool(params.get("include_imports", True))
        include_patterns = params.get("include_patterns")
        exclude_patterns = params.get("exclude_patterns")
        chunk_types = params.get("chunk_types")
        retrieval_profile = str(
            params.get("retrieval_profile", defaults["retrieval_profile"]) or defaults["retrieval_profile"]
        ).strip().lower()
        max_latency_ms = params.get("max_latency_ms", defaults["max_latency_ms"])
        max_latency_ms = int(max_latency_ms) if max_latency_ms is not None else 0
        if max_latency_ms <= 0:
            max_latency_ms = 0
        session_id = params.get("session_id")
        read_only = bool(params.get("read_only", False))
        if read_only:
            session_id = None
        explicit_deep_name = params.get("deep_name")

        deadline = workflow_started + (max_latency_ms / 1000.0) if max_latency_ms > 0 else None

        def _remaining_ms() -> int:
            if deadline is None:
                return 10**9
            return max(0, int((deadline - time.monotonic()) * 1000))

        if not read_only and not session_id:
            try:
                session_id = self.db.create_session()
            except Exception:
                session_id = None

        if not read_only and session_id:
            try:
                from know.runtime_context import set_active_session_id
                set_active_session_id(self.config, session_id)
            except Exception as e:
                logger.debug(f"Failed to persist daemon session id: {e}")

        map_t0 = time.monotonic()
        map_results = self.db.search_signatures(query, map_limit)
        map_elapsed_ms = int((time.monotonic() - map_t0) * 1000)

        context_t0 = time.monotonic()
        if _remaining_ms() <= 120:
            context_payload = {
                "query": query,
                "budget": context_budget,
                "used_tokens": 0,
                "budget_utilization": "0 / 0 (0%)",
                "indexing_status": "complete",
                "confidence": 0,
                "warnings": [
                    f"workflow latency budget exhausted before context (max_latency_ms={max_latency_ms})",
                ],
                "code": [],
                "dependencies": [],
                "tests": [],
                "summaries": [],
                "overview": "",
                "source_files": [],
                "error": "skipped_latency_budget",
            }
        else:
            semantic_max_ms = params.get("semantic_max_ms")
            if semantic_max_ms is None and deadline is not None:
                semantic_max_ms = max(200, int(_remaining_ms() * 0.75))
            if semantic_max_ms is not None:
                semantic_max_ms = int(semantic_max_ms)
            context_payload = await self._handle_context({
                "query": query,
                "budget": context_budget,
                "include_tests": include_tests,
                "include_imports": include_imports,
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
                "chunk_types": chunk_types,
                "session_id": session_id,
                "include_markdown": False,
                "retrieval_profile": retrieval_profile,
                "semantic_max_ms": semantic_max_ms,
                "read_only": read_only,
            })
        context_elapsed_ms = int((time.monotonic() - context_t0) * 1000)

        selected_target = explicit_deep_name
        selected_file = None
        if not selected_target:
            picked = self._pick_deep_target(context_payload, map_results)
            if picked:
                selected_target, selected_file = picked

        deep_t0 = time.monotonic()
        deep_result: Dict[str, Any]
        if mode == "explore" or deep_budget <= 0:
            deep_result = {
                "error": "skipped_by_mode",
                "reason": f"mode_{mode}",
            }
        elif _remaining_ms() <= 150:
            deep_result = {
                "error": "skipped_latency_budget",
                "reason": f"max_latency_ms={max_latency_ms}",
            }
        elif selected_target:
            from know.context_engine import ContextEngine
            engine = ContextEngine(self.config)
            deep_query = selected_target
            if selected_file:
                deep_query = f"{selected_file}:{selected_target}"

            deep_result = engine.build_deep_context(
                deep_query,
                budget=deep_budget,
                include_tests=include_tests,
                session_id=session_id,
                db=self.db,
            )
            if deep_result.get("error") == "ambiguous":
                # Retry with original target if file-qualified lookup was too strict.
                deep_result = engine.build_deep_context(
                    selected_target,
                    budget=deep_budget,
                    include_tests=include_tests,
                    session_id=session_id,
                    db=self.db,
                )
        else:
            deep_result = {"error": "no_target", "reason": "no_context_or_map_target"}
        deep_elapsed_ms = int((time.monotonic() - deep_t0) * 1000)

        from know.token_counter import count_tokens

        map_text = "\n".join(
            filter(
                None,
                [
                    f"{r.get('signature', '')}\n{r.get('docstring', '')}".strip()
                    for r in map_results
                ],
            )
        )
        map_tokens = count_tokens(map_text) if map_text else 0

        total_tokens = (
            map_tokens
            + int(context_payload.get("used_tokens", 0) or 0)
            + int(deep_result.get("budget_used", deep_result.get("used_tokens", 0)) or 0)
        )
        total_elapsed_ms = int((time.monotonic() - workflow_started) * 1000)
        degraded_by_latency = bool(
            context_payload.get("error") == "skipped_latency_budget"
            or deep_result.get("error") == "skipped_latency_budget"
        )

        payload = {
            "query": query,
            "session_id": session_id,
            "read_only": read_only,
            "daemon_api_version": DAEMON_API_VERSION,
            "workflow_mode": mode,
            "latency_budget_ms": max_latency_ms,
            "budgets": {
                "map_limit": map_limit,
                "context_budget": context_budget,
                "deep_budget": deep_budget,
            },
            "selected_deep_target": selected_target,
            "map": {
                "results": map_results,
                "count": len(map_results),
                "truncated": len(map_results) >= map_limit,
                "tokens": map_tokens,
            },
            "context": context_payload,
            "deep": deep_result,
            "total_tokens": total_tokens,
            "latency_ms": {
                "map": map_elapsed_ms,
                "context": context_elapsed_ms,
                "deep": deep_elapsed_ms,
                "total": total_elapsed_ms,
            },
            "degraded_by_latency": degraded_by_latency,
        }

        # Persist high-signal decision memory for future sessions.
        if not read_only:
            try:
                from know.memory_capture import capture_workflow_decision

                capture_workflow_decision(
                    self.config,
                    query,
                    payload,
                    session_id=session_id,
                    source="auto-workflow",
                    agent="daemon",
                )
            except Exception as e:
                logger.debug(f"Daemon workflow decision capture failed: {e}")

        return payload

    async def _handle_signatures(self, params: dict) -> dict:
        """Get function/class signatures."""
        file_path = params.get("file", None)
        sigs = self.db.get_signatures(file_path)
        return {"signatures": sigs}

    async def _handle_related(self, params: dict) -> dict:
        """Get related files via import graph."""
        module = params.get("module", "")
        imports = self.db.get_imports_of(module)
        imported_by = self.db.get_imported_by(module)
        return {"imports": imports, "imported_by": imported_by}

    async def _handle_remember(self, params: dict) -> dict:
        """Store a memory."""
        content = params.get("content", "")
        tags_val = params.get("tags", [])
        if isinstance(tags_val, list):
            tags = ",".join(str(t) for t in tags_val)
        else:
            tags = str(tags_val or "")
        source = params.get("source", "manual")
        session_id = params.get("session_id", "")
        agent = params.get("agent", "")
        try:
            from know.runtime_context import get_active_session_id, infer_agent_name
            if not session_id:
                session_id = get_active_session_id(self.config) or ""
            if not agent:
                agent = infer_agent_name("daemon")
        except Exception:
            pass

        try:
            from know.knowledge_base import KnowledgeBase

            with KnowledgeBase(self.config) as kb:
                mem_id = kb.remember(
                    content,
                    source=source,
                    tags=tags,
                    memory_type=params.get("memory_type", "note"),
                    decision_status=params.get("decision_status", "active"),
                    confidence=float(params.get("confidence", 0.5) or 0.5),
                    evidence=params.get("evidence", ""),
                    session_id=session_id,
                    agent=agent or "daemon",
                    trust_level=params.get("trust_level", "local_verified"),
                    supersedes_id=params.get("supersedes_id", ""),
                    expires_at=params.get("expires_at"),
                )
            return {"stored": True, "id": mem_id}
        except Exception as e:
            logger.debug(f"Daemon remember fallback path: {e}")
            return {"stored": False, "id": None}

    async def _handle_recall(self, params: dict) -> dict:
        """Recall memories by query."""
        query = params.get("query", "")
        limit = _bounded_rpc_int(
            "limit", params.get("limit", 10), maximum=MAX_RPC_RESULTS,
        )
        from know.knowledge_base import KnowledgeBase

        with KnowledgeBase(self.config) as kb:
            memories = kb.recall(
                query,
                top_k=limit,
                memory_type=params.get("memory_type"),
                decision_status=params.get("decision_status"),
                include_blocked=bool(params.get("include_blocked", False)),
                include_expired=bool(params.get("include_expired", False)),
            )
        return {"memories": [m.to_dict() for m in memories], "count": len(memories)}

    async def _handle_callers(self, params: dict) -> dict:
        """Find callers of a function."""
        function_name = params.get("function_name", "")
        limit = _bounded_rpc_int(
            "limit", params.get("limit", 50), maximum=MAX_RPC_RESULTS,
        )
        results = self.db.get_callers(function_name, limit)
        return {"callers": results, "count": len(results)}

    async def _handle_callees(self, params: dict) -> dict:
        """Find callees of a chunk."""
        chunk_name = params.get("chunk_name", "")
        limit = _bounded_rpc_int(
            "limit", params.get("limit", 50), maximum=MAX_RPC_RESULTS,
        )
        results = self.db.get_callees(chunk_name, limit)
        return {"callees": results, "count": len(results)}

    async def _handle_map(self, params: dict) -> dict:
        """Lightweight signature search — no bodies."""
        query = params.get("query", "")
        limit = _bounded_rpc_int(
            "limit", params.get("limit", 20), maximum=MAX_RPC_RESULTS,
        )
        chunk_type = params.get("chunk_type")
        results = self.db.search_signatures(query, limit, chunk_type)
        return {"results": results, "count": len(results)}

    async def _handle_deep(self, params: dict) -> dict:
        """Deep context: function body + callers + callees within budget."""
        name = params.get("name", "")
        budget = _bounded_rpc_int(
            "budget",
            params.get("budget", 3000),
            maximum=MAX_RPC_TOKEN_BUDGET,
        )
        include_tests = params.get("include_tests", False)
        session_id = params.get("session_id")

        from know.context_engine import ContextEngine
        engine = ContextEngine(self.config)
        result = engine.build_deep_context(
            name, budget=budget, include_tests=include_tests,
            session_id=session_id, db=self.db,
        )
        return result

    async def _handle_reindex(self, params: dict) -> dict:
        """Trigger full reindex."""
        count = await self._full_index()
        return {"files_indexed": count}

    async def _handle_status(self, params: dict) -> dict:
        """Get daemon status."""
        stats = self.db.get_stats()
        return {
            "running": True,
            "daemon_api_version": DAEMON_API_VERSION,
            "project": str(self.root),
            "uptime": time.time() - self._started_at,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    async def _full_index(self) -> int:
        """Index all supported files in the project.

        Runs synchronous file I/O and parsing in a thread to avoid
        blocking the async event loop.
        """
        return await asyncio.to_thread(self._full_index_sync)

    def _full_index_sync(self) -> int:
        """Synchronous indexing implementation (runs in thread pool)."""
        # Full indexing is a low-frequency lifecycle boundary and therefore a
        # safe place to expire session dedup state without adding request-path
        # latency or a separate maintenance thread.
        try:
            self.db.cleanup_expired_sessions()
        except Exception as e:
            logger.debug(f"Expired session cleanup skipped: {e}")

        count, modules = populate_index(self.root, self.config, self.db)

        # Build import graph so 'related' queries work
        try:
            from know.import_graph import ImportGraph
            with ImportGraph(self.config) as graph:
                graph.build(modules)
        except Exception as e:
            logger.debug(f"Import graph build failed: {e}")

        # Compute module importance scores (in-degree)
        try:
            scores = self.db.compute_importance()
            logger.info(f"Computed importance for {len(scores)} modules")
        except Exception as e:
            logger.debug(f"Importance computation failed: {e}")

        return count


# ---------------------------------------------------------------------------
# Client: connect to running daemon or start one
# ---------------------------------------------------------------------------
class DaemonClient:
    """Client for communicating with the daemon via Unix socket."""

    def __init__(self, project_root: Path):
        self.root = project_root
        self.sock_path = socket_path(project_root)

    async def call(self, method: str, params: Optional[dict] = None) -> dict:
        """Send a JSON-RPC request to the daemon with length-prefixed framing."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

        reader, writer = await asyncio.open_unix_connection(str(self.sock_path))
        try:
            await write_framed_message(writer, json.dumps(request).encode())
            data = await read_framed_message(reader)
            if data is None:
                raise RuntimeError("Daemon closed connection unexpectedly")
            response = json.loads(data.decode())
            if "error" in response:
                raise RuntimeError(response["error"]["message"])
            return response.get("result", {})
        finally:
            writer.close()
            await writer.wait_closed()

    def call_sync(self, method: str, params: Optional[dict] = None) -> dict:
        """Synchronous wrapper for call()."""
        return asyncio.run(self.call(method, params))


def ensure_daemon(root: Path, config: Config) -> DaemonClient:
    """Ensure daemon is running and return a client.

    Auto-starts the daemon as a background process if not running.
    """
    PID_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = PID_DIR / f"{_project_hash(root)}.startup.lock"
    with lock_path.open("a+") as lock_file:
        # Coordinate independent CLI processes as well as concurrent threads.
        # The lock remains held until the spawned daemon answers a status probe.
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if not is_daemon_running(root):
                live_pid = _live_daemon_pid(root)
                if live_pid is not None:
                    # A synchronous request can briefly keep the daemon event
                    # loop from answering probes. Replacing that live process
                    # would let two daemons unlink each other's socket/PID.
                    for _ in range(20):
                        if _probe_daemon(root):
                            break
                        time.sleep(0.25)
                    else:
                        raise RuntimeError(
                            f"Daemon process {live_pid} is alive but not responding; "
                            "refusing to replace it"
                        )
                    return DaemonClient(root)

                _start_daemon_background(root, config)
                for _ in range(50):  # Wait up to 5 seconds
                    time.sleep(0.1)
                    if is_daemon_running(root):
                        break
                else:
                    raise RuntimeError("Daemon did not become ready within 5 seconds")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return DaemonClient(root)


def _start_daemon_background(root: Path, config: Config):
    """Fork a daemon process in the background."""
    import subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "know.daemon", str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Popen warns when garbage-collected while still running. Retain ownership
    # in a daemon waiter thread without making the CLI wait for daemon lifetime.
    threading.Thread(
        target=process.wait,
        name=f"know-daemon-reaper-{process.pid}",
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# Entry point for background daemon process
# ---------------------------------------------------------------------------
def main():
    """Entry point when run as `python -m know.daemon <project_root>`."""
    if len(sys.argv) < 2:
        print("Usage: python -m know.daemon <project_root>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()
    config = Config.create_default(root)
    daemon = KnowDaemon(root, config)

    # Handle signals for clean shutdown
    def _signal_handler(sig, frame):
        daemon._shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    asyncio.run(daemon.serve())


if __name__ == "__main__":
    main()
