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
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import xxhash

from know.config import Config
from know.daemon_db import DaemonDB
from know.logger import get_logger
from know.parsers import ParserFactory, EXTENSION_TO_LANGUAGE

logger = get_logger()

IDLE_TIMEOUT = 1800  # 30 minutes
SOCKET_DIR = Path.home() / ".know" / "sockets"
PID_DIR = Path.home() / ".know" / "pids"


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

        # Initial indexing
        await self._full_index()

        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=str(sock)
        )
        # Restrict socket access to owner only
        os.chmod(str(sock), 0o600)
        logger.info(f"Daemon listening on {sock}")

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
        """Handle a single JSON-RPC connection."""
        self._last_activity = time.time()
        try:
            data = await reader.read(10 * 1024 * 1024)  # 10MB max (matches client)
            if not data:
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

            writer.write(json.dumps(response).encode())
            await writer.drain()
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
            "signatures": self._handle_signatures,
            "related": self._handle_related,
            "remember": self._handle_remember,
            "recall": self._handle_recall,
            "reindex": self._handle_reindex,
            "status": self._handle_status,
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
        """Build context for a query within a token budget."""
        query = params.get("query", "")
        budget = params.get("budget", 10000)
        results = self.db.search_chunks(query, limit=50)

        context_parts = []
        used = 0
        for chunk in results:
            tokens = chunk["token_count"]
            if used + tokens > budget:
                continue
            context_parts.append({
                "file": chunk["file_path"],
                "name": chunk["chunk_name"],
                "type": chunk["chunk_type"],
                "signature": chunk["signature"],
                "body": chunk["body"],
                "tokens": tokens,
            })
            used += tokens

        return {
            "context": context_parts,
            "tokens_used": used,
            "budget": budget,
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
        from know.scanner import CodebaseScanner

        scanner = CodebaseScanner(self.config)
        structure = scanner.get_structure()
        count = 0

        for module in structure.get("modules", []):
            path_str = module["path"] if isinstance(module, dict) else str(module.path)
            abs_path = self.root / path_str

            if not abs_path.exists():
                continue

            # Check if file changed
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            content_hash = xxhash.xxh64(content.encode()).hexdigest()
            stored_hash = self.db.get_file_hash(path_str)

            if stored_hash == content_hash:
                continue  # File unchanged

            # Detect language and parse
            lang = EXTENSION_TO_LANGUAGE.get(abs_path.suffix.lower(), "")
            if not lang:
                continue

            parser = ParserFactory.get_parser_for_file(abs_path)
            if not parser:
                continue

            try:
                mod_info = parser.parse(abs_path, self.root)
            except Exception as e:
                logger.debug(f"Parse error for {path_str}: {e}")
                continue

            # Convert to chunk dicts
            chunks = []
            for func in mod_info.functions:
                chunks.append({
                    "name": func.name,
                    "type": "method" if func.is_method else "function",
                    "start_line": func.line_number,
                    "end_line": func.line_number,
                    "signature": func.signature,
                    "body": f"{func.signature}\n{func.docstring or ''}",
                })

            for cls in mod_info.classes:
                chunks.append({
                    "name": cls.name,
                    "type": "class",
                    "start_line": cls.line_number,
                    "end_line": cls.line_number,
                    "signature": cls.name,
                    "body": f"class {cls.name}\n{cls.docstring or ''}",
                })

            if not chunks:
                # Store whole file as a module chunk
                chunks.append({
                    "name": mod_info.name,
                    "type": "module",
                    "start_line": 1,
                    "end_line": content.count("\n") + 1,
                    "signature": mod_info.name,
                    "body": content[:5000],  # Cap at 5000 chars
                })

            self.db.upsert_chunks(path_str, lang, chunks)
            self.db.update_file_index(path_str, content_hash, lang, len(chunks))
            count += 1

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
        """Send a JSON-RPC request to the daemon."""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }

        reader, writer = await asyncio.open_unix_connection(str(self.sock_path))
        try:
            writer.write(json.dumps(request).encode())
            await writer.drain()
            data = await reader.read(10 * 1024 * 1024)  # 10MB max response
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
