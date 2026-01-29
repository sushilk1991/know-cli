"""Index management for incremental code scanning."""

import json
import sqlite3
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

from know.exceptions import IndexError
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
        
        with sqlite3.connect(self.db_path) as conn:
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
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _compute_file_hash(self, content: str) -> str:
        """Compute hash of file content."""
        return _compute_hash(content)
    
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
        if not path.exists():
            return True
        
        cached = self.get_file_metadata(path)
        if not cached:
            return True
        
        try:
            stat = path.stat()
            if stat.st_mtime != cached["mtime"] or stat.st_size != cached["size"]:
                return True
            
            # Double-check with content hash
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_hash = self._compute_file_hash(content)
            return content_hash != cached["content_hash"]
        except Exception as e:
            logger.debug(f"Error checking file {path}: {e}")
            return True
    
    def get_cached_module(self, path: Path) -> Optional[Dict[str, Any]]:
        """Get cached module data if file hasn't changed."""
        if self.is_file_changed(path):
            return None
        
        try:
            relative_path = str(path.relative_to(self.root))
        except ValueError:
            return None
        
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT data FROM modules WHERE path = ?",
                (relative_path,)
            )
            row = cursor.fetchone()
            
            if row:
                return json.loads(row["data"])
        except sqlite3.Error as e:
            logger.debug(f"Database error in get_cached_module: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cache data for {path}: {e}")
        return None
    
    def cache_file(self, path: Path, language: str, module_data: Optional[Dict] = None) -> None:
        """Cache file metadata and parsed module."""
        try:
            relative_path = str(path.relative_to(self.root))
        except ValueError:
            logger.warning(f"Cannot cache file outside root: {path}")
            return
        
        try:
            stat = path.stat()
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_hash = self._compute_file_hash(content)
            
            conn = self._get_connection()
            
            # Update file metadata
            conn.execute(
                """INSERT OR REPLACE INTO files 
                   (path, mtime, size, content_hash, language, parsed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (relative_path, stat.st_mtime, stat.st_size, 
                 content_hash, language, datetime.now().isoformat())
            )
            
            # Update module data if provided
            if module_data:
                conn.execute(
                    """INSERT OR REPLACE INTO modules
                       (path, name, docstring, data, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (relative_path, module_data.get("name", ""),
                     module_data.get("docstring"),
                     json.dumps(module_data),
                     datetime.now().isoformat())
                )
            
            conn.commit()
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
    
    def get_changed_files(self, candidate_files: List[Path]) -> tuple[List[Path], List[Dict]]:
        """Split files into changed (need parsing) and unchanged (use cache).
        
        Returns:
            Tuple of (changed_files, cached_modules)
        """
        changed = []
        cached = []
        
        for path in candidate_files:
            cached_module = self.get_cached_module(path)
            if cached_module:
                cached.append(cached_module)
            else:
                changed.append(path)
        
        return changed, cached
    
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
