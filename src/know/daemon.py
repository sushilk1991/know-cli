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
import json
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, List, Tuple

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
    """Re-index a single file when missing or stale.

    Returns metadata with keys: updated(bool), removed(bool), file_path, reason,
    and optional chunks/call_refs counts.
    """
    from know.parsers import ParserFactory

    candidate = Path(file_path)
    abs_path = candidate if candidate.is_absolute() else (root / candidate)
    abs_path = abs_path.resolve()

    try:
        rel_path = str(abs_path.relative_to(root)).replace("\\", "/")
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

    content = abs_path.read_text(encoding="utf-8", errors="replace")
    content_hash = xxhash.xxh64(content.encode()).hexdigest()
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
        mod_info = parser.parse(abs_path, root)
    except Exception as e:
        return {
            "updated": False,
            "removed": False,
            "file_path": rel_path,
            "reason": f"parse_failed:{e.__class__.__name__}",
        }

    chunks = _build_chunks_from_module(content, mod_info)
    db.upsert_chunks(rel_path, lang, chunks)
    db.update_file_index(rel_path, content_hash, lang, len(chunks))

    call_refs = []
    try:
        if hasattr(parser, "extract_call_refs"):
            call_refs = parser.extract_call_refs(content, mod_info) or []
        # Always replace symbol refs so stale rows are removed on file edits.
        db.upsert_symbol_refs(rel_path, call_refs)
    except Exception as e:
        logger.debug(f"Call extraction failed for {rel_path}: {e}")
        db.upsert_symbol_refs(rel_path, [])

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
    """Populate the daemon DB with code chunks (standalone, no daemon needed).

    Called by the context engine on first use when the DB is empty.
    Also used by KnowDaemon._full_index_sync().

    Uses scanner.modules (ModuleInfo objects) directly to avoid re-parsing
    every file.  The scanner already parsed each file during scan(); we only
    need to read the raw content for body extraction and content hashing.

    Returns (file_count, modules_list) so callers can build import graphs.
    """
    from know.scanner import CodebaseScanner

    scanner = CodebaseScanner(config)
    scanner.scan()
    modules = scanner.modules  # List[ModuleInfo] — already parsed
    count = 0

    with db.batch():
        for mod_info in modules:
            path_str = str(mod_info.path)
            abs_path = root / path_str

            if not abs_path.exists():
                continue

            lang = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), "")
            if not lang:
                continue

            content = abs_path.read_text(encoding="utf-8", errors="replace")
            content_hash = xxhash.xxh64(content.encode()).hexdigest()
            stored_hash = db.get_file_hash(path_str)

            if stored_hash == content_hash:
                continue

            chunks = _build_chunks_from_module(content, mod_info)

            db.upsert_chunks(path_str, lang, chunks)
            db.update_file_index(path_str, content_hash, lang, len(chunks))

            # Extract call references for symbol_refs table
            try:
                from know.parsers import ParserFactory
                parser = ParserFactory.get_parser_for_file(abs_path)
                if parser and hasattr(parser, "extract_call_refs"):
                    call_refs = parser.extract_call_refs(content, mod_info)
                    if call_refs:
                        db.upsert_symbol_refs(path_str, call_refs)
            except Exception as e:
                logger.debug(f"Call extraction failed for {path_str}: {e}")

            count += 1

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


def is_daemon_running(root: Path) -> bool:
    """Check if a daemon is running for this project."""
    pf = pid_path(root)
    if not pf.exists():
        return False
    try:
        stored_pid = int(pf.read_text().strip())
        os.kill(stored_pid, 0)  # Check if process exists
        return True
    except (ValueError, OSError):
        # Stale PID file — clean up
        pf.unlink(missing_ok=True)
        socket_path(root).unlink(missing_ok=True)
        return False


class KnowDaemon:
    """Background daemon that maintains hot indexes for a project."""

    def __init__(self, project_root: Path, config: Config):
        self.root = project_root
        self.config = config
        self.db = DaemonDB(project_root)
        self._started_at = time.time()
        self._last_activity = time.time()
        self._server: Optional[asyncio.AbstractServer] = None

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
        if self._server:
            self._server.close()
        socket_path(self.root).unlink(missing_ok=True)
        pid_path(self.root).unlink(missing_ok=True)
        self.db.close()

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
        limit = params.get("limit", 20)
        results = self.db.search_chunks(query, limit)
        return {"results": results, "count": len(results)}

    async def _handle_context(self, params: dict) -> dict:
        """Build full v3 context for a query within a token budget."""
        query = params.get("query", "")
        budget = int(params.get("budget", 8000))
        include_tests = bool(params.get("include_tests", True))
        include_imports = bool(params.get("include_imports", True))
        include_patterns = params.get("include_patterns")
        exclude_patterns = params.get("exclude_patterns")
        chunk_types = params.get("chunk_types")
        session_id = params.get("session_id")
        include_markdown = bool(params.get("include_markdown", False))

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
            db=self.db,
        )

        # Inject relevant memories into context (same behavior as CLI fallback).
        try:
            from know.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(self.config)
            memory_ctx = kb.get_relevant_context(query, max_tokens=min(500, budget // 10))
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
        query = params.get("query", "")
        map_limit = int(params.get("map_limit", 20))
        context_budget = int(params.get("context_budget", 4000))
        deep_budget = int(params.get("deep_budget", 3000))
        include_tests = bool(params.get("include_tests", False))
        include_imports = bool(params.get("include_imports", True))
        include_patterns = params.get("include_patterns")
        exclude_patterns = params.get("exclude_patterns")
        chunk_types = params.get("chunk_types")
        session_id = params.get("session_id")
        explicit_deep_name = params.get("deep_name")

        map_results = self.db.search_signatures(query, map_limit)

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
        })

        selected_target = explicit_deep_name
        selected_file = None
        if not selected_target:
            picked = self._pick_deep_target(context_payload, map_results)
            if picked:
                selected_target, selected_file = picked

        deep_result: Dict[str, Any]
        if selected_target:
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

        return {
            "query": query,
            "session_id": session_id,
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
        }

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
        import uuid
        content = params.get("content", "")
        tags = json.dumps(params.get("tags", []))
        source = params.get("source", "manual")
        memory_id = str(uuid.uuid4())[:8]
        stored = self.db.store_memory(memory_id, content, tags, source)
        return {"stored": stored, "id": memory_id if stored else None}

    async def _handle_recall(self, params: dict) -> dict:
        """Recall memories by query."""
        query = params.get("query", "")
        limit = params.get("limit", 10)
        memories = self.db.recall_memories(query, limit)
        return {"memories": memories, "count": len(memories)}

    async def _handle_callers(self, params: dict) -> dict:
        """Find callers of a function."""
        function_name = params.get("function_name", "")
        limit = params.get("limit", 50)
        results = self.db.get_callers(function_name, limit)
        return {"callers": results, "count": len(results)}

    async def _handle_callees(self, params: dict) -> dict:
        """Find callees of a chunk."""
        chunk_name = params.get("chunk_name", "")
        limit = params.get("limit", 50)
        results = self.db.get_callees(chunk_name, limit)
        return {"callees": results, "count": len(results)}

    async def _handle_map(self, params: dict) -> dict:
        """Lightweight signature search — no bodies."""
        query = params.get("query", "")
        limit = params.get("limit", 20)
        chunk_type = params.get("chunk_type")
        results = self.db.search_signatures(query, limit, chunk_type)
        return {"results": results, "count": len(results)}

    async def _handle_deep(self, params: dict) -> dict:
        """Deep context: function body + callers + callees within budget."""
        name = params.get("name", "")
        budget = params.get("budget", 3000)
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
        count, modules = populate_index(self.root, self.config, self.db)

        # Build import graph so 'related' queries work
        try:
            from know.import_graph import ImportGraph
            ig = ImportGraph(self.config)
            ig.build(modules)
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
    if not is_daemon_running(root):
        _start_daemon_background(root, config)
        # Wait for socket to appear
        sock = socket_path(root)
        for _ in range(50):  # Wait up to 5 seconds
            time.sleep(0.1)
            if sock.exists():
                break

    return DaemonClient(root)


def _start_daemon_background(root: Path, config: Config):
    """Fork a daemon process in the background."""
    import subprocess
    subprocess.Popen(
        [sys.executable, "-m", "know.daemon", str(root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


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
