"""Semantic code search using real embeddings and vector similarity."""

import hashlib
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import pathspec
except ImportError:
    pathspec = None


class EmbeddingCache:
    """Cache for code embeddings using binary storage with persistent connection support."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "know-cli"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings-v2.db"
        self._conn = None
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

    def _init_db(self):
        """Initialize SQLite cache for embeddings."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        file_hash TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        model TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_path ON embeddings(file_path)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)
                """)
                # Clean old entries (30 days)
                conn.execute(
                    "DELETE FROM embeddings WHERE created_at < datetime('now', '-30 days')"
                )
        finally:
            if not self._conn:
                conn.close()
    
    def get(self, file_path: str, content_hash: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding if available and model matches."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """SELECT embedding, dim FROM embeddings 
                   WHERE file_path = ? AND content_hash = ? AND model = ?""",
                (file_path, content_hash, model)
            )
            row = cursor.fetchone()
            if row:
                # Deserialize from binary blob
                embedding_bytes = row[0]
                dim = row[1]
                return np.frombuffer(embedding_bytes, dtype=np.float32).reshape(dim)
        finally:
            if not self._conn:
                conn.close()
        return None
    
    def set(self, file_path: str, content_hash: str, embedding: np.ndarray, model: str):
        """Cache an embedding as binary blob."""
        file_hash = hashlib.sha256(f"{file_path}:{content_hash}:{model}".encode()).hexdigest()
        embedding_bytes = embedding.astype(np.float32).tobytes()
        dim = embedding.shape[0]
        
        conn = self._get_conn()
        try:
            with conn:
                conn.execute(
                    """INSERT OR REPLACE INTO embeddings 
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
            
        data = []
        for file_path, content_hash, embedding, model in items:
            file_hash = hashlib.sha256(f"{file_path}:{content_hash}:{model}".encode()).hexdigest()
            embedding_bytes = embedding.astype(np.float32).tobytes()
            dim = embedding.shape[0]
            data.append((file_hash, file_path, content_hash, embedding_bytes, model, dim))
            
        conn = self._get_conn()
        try:
            with conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO embeddings 
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
        """
        if not keys:
            return {}
            
        # Simplified batch get (could be optimized with IN clause but requires constructing query)
        # For now, just iterate with the persistent connection
        results = {}
        conn = self._get_conn()
        try:
            for fp, ch, md in keys:
                cursor = conn.execute(
                    """SELECT embedding, dim FROM embeddings 
                       WHERE file_path = ? AND content_hash = ? AND model = ?""",
                    (fp, ch, md)
                )
                row = cursor.fetchone()
                if row:
                    embedding = np.frombuffer(row[0], dtype=np.float32).reshape(row[1])
                    results[(fp, ch, md)] = embedding
        finally:
            if not self._conn:
                conn.close()
        return results

    def get_all_embeddings(self, model: str) -> Tuple[List[str], np.ndarray]:
        """Get all embeddings for a model as a matrix. Returns (file_paths, embedding_matrix)."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT file_path, embedding, dim FROM embeddings WHERE model = ?",
                (model,)
            )
            rows = cursor.fetchall()
        finally:
            if not self._conn:
                conn.close()
        
        if not rows:
            return [], np.array([])
        
        file_paths = []
        embeddings = []
        
        for row in rows:
            file_paths.append(row[0])
            embedding = np.frombuffer(row[1], dtype=np.float32).reshape(row[2])
            embeddings.append(embedding)
        
        return file_paths, np.array(embeddings)
    
    def clear_model(self, model: str):
        """Clear all embeddings for a specific model."""
        conn = self._get_conn()
        try:
            with conn:
                conn.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        finally:
            if not self._conn:
                conn.close()


class SemanticSearcher:
    """Semantic code search using fastembed embeddings."""
    
    # Model configuration
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, good for code, ~100MB
    MODEL_DIM = 384
    MAX_FILE_SIZE = 1024 * 1024  # 1MB
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache = EmbeddingCache()
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self._embedding_model = TextEmbedding(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "fastembed is required for semantic search. "
                    "Install with: pip install fastembed"
                )
        return self._embedding_model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using fastembed."""
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
        try:
            # Check file size before reading
            size = file_path.stat().st_size
            if size > self.MAX_FILE_SIZE:
                return False

            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                return False
            
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Check cache
            cached = self.cache.get(str(file_path), content_hash, self.model_name)
            if cached is not None:
                return True
            
            # Get embedding
            embedding = self._get_embedding(content)
            
            # Cache it
            self.cache.set(str(file_path), content_hash, embedding, self.model_name)
            return True
            
        except Exception:
            return False
    
    def index_directory(self, root: Path, extensions: List[str] = None) -> int:
        """Index all files in a directory. Returns count of indexed files."""
        if extensions is None:
            extensions = [".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".cpp", ".c", ".h"]
        
        # Load .gitignore if present
        ignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists() and pathspec:
            try:
                with open(gitignore_path, "r") as f:
                    ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
            except Exception:
                pass

        count = 0
        
        # Use persistent connection
        with self.cache:
            for ext in extensions:
                for file_path in root.rglob(f"*{ext}"):
                    # Check .gitignore
                    if ignore_spec:
                        try:
                            rel_path = file_path.relative_to(root)
                            if ignore_spec.match_file(str(rel_path)):
                                continue
                        except ValueError:
                            pass
                    
                    # Fallback skip for common dirs if pathspec failed or didn't catch them
                    if any(part.startswith(".") or part in {"node_modules", "__pycache__", "venv", ".git", "dist", "build"} 
                           for part in file_path.parts):
                        continue
                    
                    if self.index_file(file_path):
                        count += 1
        
        return count
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for files semantically similar to query."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Get all cached embeddings as matrix
        file_paths, embeddings_matrix = self.cache.get_all_embeddings(self.model_name)
        
        if not file_paths or embeddings_matrix.size == 0:
            return []
        
        # Vectorized cosine similarity
        similarities = self._cosine_similarity(query_embedding, embeddings_matrix)
        
        # Get top-k indices
        top_indices = np.argpartition(similarities, -min(top_k, len(similarities)))[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Return results
        results = [(file_paths[i], float(similarities[i])) for i in top_indices if similarities[i] > 0]
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
