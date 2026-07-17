"""Semantic code search using real embeddings and vector similarity.

Supports both file-level (v1) and function-level (v2) embeddings.
Function-level uses AST to split Python files into individual chunks
(functions, classes, module summaries).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from know.path_filters import is_hard_excluded_path

try:
    import pathspec
except ImportError:
    pathspec = None


class EmbeddingCache:
    """Cache for code embeddings using binary storage with persistent connection support.
    
    Embeddings are scoped per project to prevent cross-project contamination.
    Each project gets its own table based on a hash of the project root path.
    """
    
    def __init__(self, cache_dir: Path = None, project_root: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "know-cli"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings-v2.db"
        self._conn = None
        
        # Project scoping: use hash of project root to isolate embeddings
        if project_root:
            project_id = hashlib.sha256(
                str(project_root.resolve()).encode()
            ).hexdigest()[:16]
        else:
            project_id = "global"
        # Validate table name is safe (only alphanumeric + underscore)
        if not project_id.isalnum():
            raise ValueError(f"Invalid project_id: {project_id}")
        self._project_id = project_id
        self._table = f"embeddings_{self._project_id}"
        
        self._init_db()
    
    def __enter__(self):
        """Enter context manager: open persistent connection."""
        self._conn = sqlite3.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager: close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get current persistent connection or create a temporary one."""
        if self._conn:
            return self._conn
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _decode_embedding(blob: bytes, dim: int) -> Optional[np.ndarray]:
        """Decode a cache row, returning ``None`` when it is corrupt."""
        try:
            expected_dim = int(dim)
            embedding = np.frombuffer(blob, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if (
            expected_dim <= 0
            or embedding.size != expected_dim
            or not np.isfinite(embedding).all()
        ):
            return None
        return embedding

    def _init_db(self):
        """Initialize SQLite cache for embeddings (project-scoped table)."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        file_hash TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        model TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_path_{self._project_id}
                    ON {self._table}(file_path)
                """)
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_model_{self._project_id}
                    ON {self._table}(model)
                """)
                # Clean old entries (30 days)
                conn.execute(
                    f"DELETE FROM {self._table} WHERE created_at < datetime('now', '-30 days')"
                )
        finally:
            if not self._conn:
                conn.close()
    
    def get(self, file_path: str, content_hash: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding if available and model matches."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"""SELECT embedding, dim FROM {self._table} 
                   WHERE file_path = ? AND content_hash = ? AND model = ?""",
                (file_path, content_hash, model)
            )
            row = cursor.fetchone()
            if row:
                return self._decode_embedding(row[0], row[1])
        finally:
            if not self._conn:
                conn.close()
        return None
    
    def delete_path(self, file_path: str, model: str) -> None:
        """Delete every cached version of one logical path/model pair."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(
                    f"DELETE FROM {self._table} WHERE file_path = ? AND model = ?",
                    (file_path, model),
                )
        finally:
            if not self._conn:
                conn.close()

    def set(self, file_path: str, content_hash: str, embedding: np.ndarray, model: str):
        """Cache an embedding as binary blob."""
        file_hash = hashlib.sha256(f"{file_path}:{content_hash}:{model}".encode()).hexdigest()
        embedding_bytes = embedding.astype(np.float32).tobytes()
        dim = embedding.shape[0]
        
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(
                    f"DELETE FROM {self._table} WHERE file_path = ? AND model = ?",
                    (file_path, model),
                )
                conn.execute(
                    f"""INSERT OR REPLACE INTO {self._table} 
                       (file_hash, file_path, content_hash, embedding, model, dim)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (file_hash, file_path, content_hash, embedding_bytes, model, dim)
                )
        finally:
            if not self._conn:
                conn.close()

    def batch_set(self, items: List[Tuple[str, str, np.ndarray, str]]):
        """Batch insert embeddings. items = [(file_path, content_hash, embedding, model), ...]"""
        if not items:
            return
            
        latest = {}
        for file_path, content_hash, embedding, model in items:
            latest[(file_path, model)] = (file_path, content_hash, embedding, model)

        data = []
        for file_path, content_hash, embedding, model in latest.values():
            file_hash = hashlib.sha256(f"{file_path}:{content_hash}:{model}".encode()).hexdigest()
            embedding_bytes = embedding.astype(np.float32).tobytes()
            dim = embedding.shape[0]
            data.append((file_hash, file_path, content_hash, embedding_bytes, model, dim))
            
        conn = self._get_conn()
        try:
            with conn:
                conn.executemany(
                    f"DELETE FROM {self._table} WHERE file_path = ? AND model = ?",
                    [(row[1], row[4]) for row in data],
                )
                conn.executemany(
                    f"""INSERT OR REPLACE INTO {self._table} 
                       (file_hash, file_path, content_hash, embedding, model, dim)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    data
                )
        finally:
            if not self._conn:
                conn.close()

    def batch_get(self, keys: List[Tuple[str, str, str]]) -> dict:
        """Batch get embeddings. keys = [(file_path, content_hash, model), ...]
        Returns dict: {(file_path, content_hash, model): embedding}
        
        Uses efficient batch query with proper parameterization.
        """
        if not keys:
            return {}
        
        # Group by model (usually all same model)
        by_model: dict = {}
        for fp, ch, md in keys:
            if md not in by_model:
                by_model[md] = []
            by_model[md].append((fp, ch))
        
        results = {}
        conn = self._get_conn()
        try:
            for model, model_keys in by_model.items():
                # Build efficient batch query
                placeholders = ','.join(['(?,?)'] * len(model_keys))
                flat_values = [v for pair in model_keys for v in pair]
                
                # Query all matching records for this model
                query = f"""
                    SELECT file_path, content_hash, embedding, dim 
                    FROM {self._table} 
                    WHERE model = ? AND (file_path, content_hash) IN ({placeholders})
                """
                cursor = conn.execute(query, (model, *flat_values))
                
                for row in cursor:
                    key = (row[0], row[1], model)
                    embedding = self._decode_embedding(row[2], row[3])
                    if embedding is not None:
                        results[key] = embedding
        finally:
            if not self._conn:
                conn.close()
        return results

    def get_all_embeddings(
        self,
        model: str,
        path_prefix: Optional[str] = None,
        expected_dim: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """Get all embeddings for a model as a matrix. Returns (file_paths, embedding_matrix)."""
        conn = self._get_conn()
        try:
            if path_prefix is None:
                cursor = conn.execute(
                    f"SELECT file_path, embedding, dim FROM {self._table} WHERE model = ?",
                    (model,),
                )
            else:
                cursor = conn.execute(
                    f"""SELECT file_path, embedding, dim FROM {self._table}
                        WHERE model = ? AND file_path LIKE ?""",
                    (model, f"{path_prefix}%"),
                )
            rows = cursor.fetchall()
        finally:
            if not self._conn:
                conn.close()
        
        if not rows:
            return [], np.array([])
        
        decoded_rows = []
        
        for row in rows:
            embedding = self._decode_embedding(row[1], row[2])
            if embedding is not None:
                decoded_rows.append((row[0], embedding))

        if not decoded_rows:
            return [], np.array([])

        # A live query's dimension is authoritative. Without one, retain the
        # dominant valid dimension for callers that only inspect the cache.
        if expected_dim is None:
            dimension_counts: dict[int, int] = {}
            for _, embedding in decoded_rows:
                dimension_counts[embedding.size] = (
                    dimension_counts.get(embedding.size, 0) + 1
                )
            expected_dim = max(dimension_counts, key=dimension_counts.get)
        compatible_rows = [
            (path, embedding)
            for path, embedding in decoded_rows
            if embedding.size == expected_dim
        ]
        if not compatible_rows:
            return [], np.array([])
        
        return (
            [path for path, _ in compatible_rows],
            np.stack([embedding for _, embedding in compatible_rows]),
        )

    def prune_paths(self, model: str, path_prefix: str, keep: set[str]) -> None:
        """Remove cached logical paths no longer present in an authoritative scan."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                f"SELECT DISTINCT file_path FROM {self._table} WHERE model = ? AND file_path LIKE ?",
                (model, f"{path_prefix}%"),
            ).fetchall()
            stale = [(path, model) for (path,) in rows if path not in keep]
            if stale:
                with conn:
                    conn.executemany(
                        f"DELETE FROM {self._table} WHERE file_path = ? AND model = ?",
                        stale,
                    )
        finally:
            if not self._conn:
                conn.close()
    
    def clear_model(self, model: str):
        """Clear all embeddings for a specific model."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(f"DELETE FROM {self._table} WHERE model = ?", (model,))
        finally:
            if not self._conn:
                conn.close()


class SemanticSearcher:
    """Semantic code search using fastembed embeddings."""
    
    # Model configuration
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, good for code, ~100MB
    MODEL_DIM = 384
    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    FILE_CACHE_PREFIX = "file:"
    CHUNK_CACHE_PREFIX = "chunk:"
    
    # Embedding model managed by know.embeddings (centralized)

    def __init__(self, model_name: Optional[str] = None, project_root: Optional[Path] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.project_root = project_root
        self.cache = EmbeddingCache(project_root=project_root)

    @staticmethod
    def _path_is_within_root(file_path: Path, root: Path) -> bool:
        """Reject lexical in-root paths whose symlink target escapes root."""
        try:
            file_path.resolve().relative_to(root.resolve())
            file_path.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def _get_embedding_model(self):
        """Get embedding model from centralized manager."""
        from know.embeddings import get_model
        model = get_model(self.model_name)
        if model is None:
            raise ImportError(
                "Embedding runtime unavailable for semantic search. "
                "Repair with: python -m pip install -U know-cli && know doctor --repair --reindex"
            )
        return model

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using centralized embedding manager."""
        model = self._get_embedding_model()
        
        # Truncate if too long (model has max tokens)
        max_chars = 8000  # Approximate for most models
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # fastembed returns a generator, get first embedding
        embeddings = list(model.embed([text]))
        return np.array(embeddings[0], dtype=np.float32)
    
    def _cosine_similarity(self, query_embedding: np.ndarray, embeddings_matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity using vectorized numpy operations."""
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(embeddings_matrix.shape[0])
        query_normalized = query_embedding / query_norm
        
        # Normalize embeddings matrix (row-wise)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_matrix / norms
        
        # Cosine similarity = dot product of normalized vectors
        return np.dot(embeddings_normalized, query_normalized)
    
    def index_file(self, file_path: Path) -> bool:
        """Index a single file. Returns True if indexed or already cached."""
        cache_key = f"{self.FILE_CACHE_PREFIX}{file_path}"
        try:
            if self.project_root is not None and not self._path_is_within_root(
                file_path, self.project_root,
            ):
                self.cache.delete_path(cache_key, self.model_name)
                return False

            # Check file size before reading
            size = file_path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                self.cache.delete_path(cache_key, self.model_name)
                return False

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                self.cache.delete_path(cache_key, self.model_name)
                return False
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Check cache
            cached = self.cache.get(cache_key, content_hash, self.model_name)
            if cached is not None:
                return True

            # Never serve a prior version if recomputing the new content fails.
            self.cache.delete_path(cache_key, self.model_name)
            
            # Get embedding
            embedding = self._get_embedding(content)
            
            # Cache it
            self.cache.set(cache_key, content_hash, embedding, self.model_name)
            return True
            
        except Exception:
            # A terminal read/stat/embed failure must not leave an older
            # version searchable as though the current file were indexed.
            try:
                self.cache.delete_path(cache_key, self.model_name)
            except Exception:
                pass
            return False
    
    def index_directory(self, root: Path, extensions: List[str] = None) -> int:
        """Index all files in a directory. Returns count of indexed files."""
        if extensions is None:
            from know.parsers import ParserFactory
            extensions = sorted(ParserFactory.supported_extensions())
        
        # Load .gitignore if present
        ignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists() and pathspec:
            try:
                with open(gitignore_path, "r") as f:
                    ignore_spec = pathspec.GitIgnoreSpec.from_lines(f)
            except Exception:
                pass

        count = 0
        discovered: set[str] = set()
        
        # Use persistent connection
        with self.cache:
            for ext in extensions:
                for file_path in root.rglob(f"*{ext}"):
                    if not self._path_is_within_root(file_path, root):
                        continue

                    # Check .gitignore
                    if ignore_spec:
                        try:
                            rel_path = file_path.relative_to(root)
                            if ignore_spec.match_file(str(rel_path)):
                                continue
                        except ValueError:
                            pass
                    
                    # Always skip runtime/build/cache paths even if .gitignore does not.
                    try:
                        filter_path = file_path.relative_to(root)
                    except ValueError:
                        filter_path = file_path
                    if is_hard_excluded_path(filter_path):
                        continue

                    if self.index_file(file_path):
                        discovered.add(f"{self.FILE_CACHE_PREFIX}{file_path}")
                        count += 1

            self.cache.prune_paths(self.model_name, self.FILE_CACHE_PREFIX, discovered)
        
        return count
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for files semantically similar to query."""
        if top_k <= 0:
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Get all cached embeddings as matrix
        file_paths, embeddings_matrix = self.cache.get_all_embeddings(
            self.model_name,
            path_prefix=self.FILE_CACHE_PREFIX,
            expected_dim=query_embedding.size,
        )
        
        if not file_paths or embeddings_matrix.size == 0:
            return []
        
        # Vectorized cosine similarity
        similarities = self._cosine_similarity(query_embedding, embeddings_matrix)
        
        # Get top-k indices
        top_indices = np.argpartition(similarities, -min(top_k, len(similarities)))[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Return results
        results = [
            (file_paths[i][len(self.FILE_CACHE_PREFIX):], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0
        ]
        return results
    
    def search_code(self, query: str, root: Path, top_k: int = 10, auto_index: bool = True) -> List[dict]:
        """Search code with context and previews."""
        # Auto-index if requested
        if auto_index:
            self.index_directory(root)
        
        # Search
        results = self.search(query, top_k)
        
        # Format results with snippets
        formatted = []
        for file_path, score in results:
            try:
                path = Path(file_path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    # Get first 10 lines as preview
                    lines = content.split("\n")[:10]
                    preview = "\n".join(lines)[:500]
                    
                    formatted.append({
                        "path": file_path,
                        "score": round(score, 3),
                        "preview": preview
                    })
            except Exception:
                pass
        
        return formatted
    
    def clear_cache(self):
        """Clear all cached embeddings for current model."""
        self.cache.clear_model(self.model_name)

    # ------------------------------------------------------------------
    # Function-level (chunk) embeddings
    # ------------------------------------------------------------------
    def index_chunks(self, root: Path, extensions: List[str] = None) -> int:
        """Index code at function/class level for Python files.

        Non-Python files fall back to file-level embedding.
        Returns number of chunks indexed.
        """
        from know.context_engine import extract_chunks_from_file

        if extensions is None:
            from know.parsers import ParserFactory
            extensions = sorted(ParserFactory.supported_extensions())

        ignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists() and pathspec:
            try:
                with open(gitignore_path, "r") as f:
                    ignore_spec = pathspec.GitIgnoreSpec.from_lines(f)
            except Exception:
                pass

        count = 0
        discovered: set[str] = set()

        with self.cache:
            for ext in extensions:
                for file_path in root.rglob(f"*{ext}"):
                    if not self._path_is_within_root(file_path, root):
                        continue

                    # Always skip runtime/build/cache paths even if .gitignore does not.
                    try:
                        filter_path = file_path.relative_to(root)
                    except ValueError:
                        filter_path = file_path
                    if is_hard_excluded_path(filter_path):
                        continue
                    if ignore_spec:
                        try:
                            rel_path = file_path.relative_to(root)
                            if ignore_spec.match_file(str(rel_path)):
                                continue
                        except ValueError:
                            pass

                    # Check file size
                    try:
                        if file_path.stat().st_size > self.MAX_FILE_SIZE:
                            continue
                    except OSError:
                        continue

                    chunks = extract_chunks_from_file(file_path, root)
                    for chunk in chunks:
                        raw_chunk_key = f"{chunk.file_path}::{chunk.name}::{chunk.line_start}"
                        chunk_key = f"{self.CHUNK_CACHE_PREFIX}{raw_chunk_key}"
                        discovered.add(chunk_key)
                        # Build text for embedding
                        embed_text = f"{chunk.name} {chunk.signature}\n{chunk.docstring}\n{chunk.body[:2000]}"
                        content_hash = hashlib.sha256(embed_text.encode()).hexdigest()[:16]

                        # Check cache
                        cached = self.cache.get(chunk_key, content_hash, self.model_name)
                        if cached is not None:
                            count += 1
                            continue

                        self.cache.delete_path(chunk_key, self.model_name)

                        try:
                            embedding = self._get_embedding(embed_text)
                            self.cache.set(chunk_key, content_hash, embedding, self.model_name)
                            count += 1
                        except Exception:
                            pass

            self.cache.prune_paths(self.model_name, self.CHUNK_CACHE_PREFIX, discovered)

        return count

    def search_chunks(self, query: str, root: Path, top_k: int = 20, auto_index: bool = True) -> List[dict]:
        """Search at function/class level with metadata.
        
        Returns list of dicts with: path, name, type, line_start, line_end, score, preview.
        """
        if top_k <= 0:
            return []

        if auto_index:
            self.index_chunks(root)

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Get all embeddings
        file_paths, embeddings_matrix = self.cache.get_all_embeddings(
            self.model_name,
            path_prefix=self.CHUNK_CACHE_PREFIX,
            expected_dim=query_embedding.size,
        )
        if not file_paths or embeddings_matrix.size == 0:
            return []

        # Cosine similarity
        similarities = self._cosine_similarity(query_embedding, embeddings_matrix)
        top_indices = np.argpartition(similarities, -min(top_k, len(similarities)))[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []
        for i in top_indices:
            if similarities[i] <= 0:
                continue
            cache_key = file_paths[i]
            chunk_key = cache_key[len(self.CHUNK_CACHE_PREFIX):]
            # Parse chunk key: "file_path::name::line_start"
            parts = chunk_key.split("::")
            if len(parts) >= 3:
                results.append({
                    "path": parts[0],
                    "name": parts[1],
                    "line_start": int(parts[2]) if parts[2].isdigit() else 1,
                    "score": round(float(similarities[i]), 3),
                    "chunk_key": chunk_key,
                })
            else:
                # File-level embedding (old style)
                results.append({
                    "path": chunk_key,
                    "name": Path(chunk_key).stem,
                    "line_start": 1,
                    "score": round(float(similarities[i]), 3),
                    "chunk_key": chunk_key,
                })

        return results
