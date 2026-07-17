"""Index management for incremental code scanning."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional
from datetime import datetime
import logging

try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    import hashlib

from know.exceptions import IndexingError
from know.logger import get_logger

if TYPE_CHECKING:
    from know.config import Config
    from know.scanner import ModuleInfo


logger = get_logger()


def _compute_hash(content: str) -> str:
    """Compute hash of file content using xxhash if available."""
    if XXHASH_AVAILABLE:
        return xxhash.xxh64(content.encode("utf-8")).hexdigest()
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def read_source_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    """Read content and metadata from one stable version of ``path``."""
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except (FileNotFoundError, OSError):
        return None

    before_key = (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_size,
    )
    after_key = (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_size,
    )
    if before_key != after_key or len(raw) != after.st_size:
        return None

    content = raw.decode("utf-8", errors="replace")
    return {
        "content": content,
        "mtime": after.st_mtime,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "device": after.st_dev,
        "inode": after.st_ino,
        "size": after.st_size,
        "content_hash": _compute_hash(content),
    }


def source_snapshot_matches(
    before: Dict[str, Any], after: Dict[str, Any],
) -> bool:
    """Return whether two snapshots identify the same file generation."""
    return all(
        before.get(key) == after.get(key)
        for key in (
            "content_hash", "mtime_ns", "ctime_ns", "device", "inode", "size",
        )
    )


class CodebaseIndex:
    """SQLite-based index for efficient incremental scanning.
    
    Uses connection pooling for better performance with multiple operations.
    """
    
    def __init__(self, config: "Config"):
        self.config = config
        self.root = config.root
        self.db_path = config.root / ".know" / "cache" / "index.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS modules (
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    docstring TEXT,
                    data TEXT NOT NULL,  -- JSON serialized ModuleInfo
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (path) REFERENCES files(path)
                );
                
                CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
                CREATE INDEX IF NOT EXISTS idx_modules_updated ON modules(updated_at);
            """)
    
    def close(self) -> None:
        """Close database connection."""
        conn = getattr(self, "_conn", None)
        if conn:
            conn.close()
            self._conn = None

    def __del__(self):
        """Best-effort fallback for short-lived callers that omit ``close``."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _compute_file_hash(self, content: str) -> str:
        """Compute hash of file content."""
        return _compute_hash(content)

    def read_file_snapshot(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read a stable source snapshot and its matching filesystem metadata.

        The stat calls bracket the read so callers never associate parsed
        content with metadata from a different file version. A concurrent
        writer simply makes this scan retry the file on the next pass.
        """
        return read_source_snapshot(path)
    
    def get_file_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get cached metadata for a file."""
        try:
            relative_path = str(path.relative_to(self.root))
        except ValueError:
            return None
        
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT * FROM files WHERE path = ?",
                (relative_path,)
            )
            row = cursor.fetchone()
            
            if row:
                return dict(row)
        except sqlite3.Error as e:
            logger.debug(f"Database error in get_file_metadata: {e}")
        return None
    
    def is_file_changed(self, path: Path) -> bool:
        """Check if file has changed since last scan."""
        cached = self.get_file_metadata(path)
        if not cached:
            return True

        snapshot = self.read_file_snapshot(path)
        if snapshot is None:
            return True
        return (
            snapshot["mtime"] != cached["mtime"]
            or snapshot["size"] != cached["size"]
            or snapshot["content_hash"] != cached["content_hash"]
        )

    def get_cached_module_snapshot(
        self, path: Path,
    ) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
        """Return cached module plus the exact source snapshot it describes."""
        try:
            relative_path = str(path.relative_to(self.root))
        except ValueError:
            return None

        snapshot = self.read_file_snapshot(path)
        if snapshot is None:
            return None

        try:
            conn = self._get_connection()
            row = conn.execute(
                """SELECT f.mtime, f.size, f.content_hash, m.data
                   FROM files AS f
                   JOIN modules AS m ON m.path = f.path
                   WHERE f.path = ?""",
                (relative_path,),
            ).fetchone()
            if row is None:
                return None
            if (
                snapshot["mtime"] != row["mtime"]
                or snapshot["size"] != row["size"]
                or snapshot["content_hash"] != row["content_hash"]
            ):
                return None
            module_data = json.loads(row["data"])
            expected_path = relative_path.replace("\\", "/")
            if (
                not isinstance(module_data, dict)
                or str(module_data.get("path", "")).replace("\\", "/")
                != expected_path
            ):
                logger.warning(f"Invalid cached module identity for {path}")
                return None
            return module_data, snapshot
        except sqlite3.Error as e:
            logger.debug(f"Database error in get_cached_module_snapshot: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cache data for {path}: {e}")
        return None

    def get_cached_module(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get cached module data if file hasn't changed."""
        cached = self.get_cached_module_snapshot(path)
        return cached[0] if cached else None
    
    def cache_file(
        self,
        path: Path,
        language: str,
        module_data: Optional[Dict] = None,
        *,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache file metadata and parsed module."""
        try:
            relative_path = str(path.relative_to(self.root))
        except ValueError:
            logger.warning(f"Cannot cache file outside root: {path}")
            return
        
        try:
            snapshot = snapshot or self.read_file_snapshot(path)
            if snapshot is None:
                logger.debug(f"File changed while reading; not caching {path}")
                return

            # Serialize before touching metadata. If serialization fails, the
            # old file row must keep describing the old module row so the
            # changed source remains eligible for a retry.
            serialized_module = (
                json.dumps(module_data) if module_data is not None else None
            )
            conn = self._get_connection()

            # The context manager commits both rows or rolls both back. In
            # particular, a failed module write cannot make stale module data
            # look current by publishing only the new file hash.
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO files
                       (path, mtime, size, content_hash, language, parsed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (relative_path, snapshot["mtime"], snapshot["size"],
                     snapshot["content_hash"], language,
                     datetime.now().isoformat())
                )

                if module_data is None:
                    conn.execute("DELETE FROM modules WHERE path = ?", (relative_path,))
                else:
                    conn.execute(
                        """INSERT OR REPLACE INTO modules
                           (path, name, docstring, data, updated_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (relative_path, module_data.get("name", ""),
                         module_data.get("docstring"), serialized_module,
                         datetime.now().isoformat())
                    )
        except sqlite3.Error as e:
            logger.warning(f"Database error caching {path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to cache file {path}: {e}")
    
    def iter_cached_modules(self) -> Iterator[Dict[str, Any]]:
        """Iterate over cached modules (memory-efficient).
        
        Yields modules one at a time instead of loading all into memory.
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT data FROM modules")
            
            for row in cursor:
                try:
                    yield json.loads(row["data"])
                except json.JSONDecodeError:
                    continue
        except sqlite3.Error as e:
            logger.error(f"Database error iterating modules: {e}")
    
    def get_all_cached_modules(self) -> List[Dict[str, Any]]:
        """Get all cached module data."""
        return list(self.iter_cached_modules())
    
    def get_changed_files(
        self, candidate_files: List[Path],
    ) -> tuple[List[Path], List[Dict], Dict[str, Dict[str, Any]]]:
        """Split files into changed (need parsing) and unchanged (use cache).
        
        Returns:
            Tuple of (changed_files, cached_modules)
        """
        changed = []
        cached = []
        snapshots: Dict[str, Dict[str, Any]] = {}
        
        for path in candidate_files:
            cached_result = self.get_cached_module_snapshot(path)
            if cached_result:
                cached_module, snapshot = cached_result
                cached.append(cached_module)
                snapshots[str(cached_module.get("path", ""))] = snapshot
            else:
                changed.append(path)
        
        return changed, cached, snapshots
    
    def remove_stale_entries(self, existing_files: List[Path]) -> int:
        """Remove cache entries for files that no longer exist."""
        existing_paths = set()
        for p in existing_files:
            try:
                existing_paths.add(str(p.relative_to(self.root)))
            except ValueError:
                continue
        
        removed = 0
        
        try:
            conn = self._get_connection()
            
            # Get all cached paths
            cursor = conn.execute("SELECT path FROM files")
            cached_paths = {row[0] for row in cursor}
            
            # Find stale paths
            stale_paths = cached_paths - existing_paths
            
            for path in stale_paths:
                conn.execute("DELETE FROM modules WHERE path = ?", (path,))
                conn.execute("DELETE FROM files WHERE path = ?", (path,))
                removed += 1
            
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error removing stale entries: {e}")
        
        return removed
    
    def get_stats(self) -> Dict[str, int]:
        """Get index statistics."""
        try:
            conn = self._get_connection()
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            module_count = conn.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
            
            return {
                "cached_files": file_count,
                "cached_modules": module_count
            }
        except sqlite3.Error as e:
            logger.error(f"Database error getting stats: {e}")
            return {"cached_files": 0, "cached_modules": 0}
    
    def clear(self) -> None:
        """Clear all cached data."""
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM modules")
            conn.execute("DELETE FROM files")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error clearing index: {e}")
    
    def vacuum(self) -> None:
        """Optimize database file size."""
        try:
            conn = self._get_connection()
            conn.execute("VACUUM")
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error during vacuum: {e}")
