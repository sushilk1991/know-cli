"""Centralized embedding model management.

Single process-wide cache for fastembed models. All consumers
(context_engine, semantic_search, knowledge_base) use this instead
of maintaining separate caches.

fastembed is optional — this module gracefully returns None when
it's not installed.
"""

import threading
import time
from typing import Any, Dict, Optional

from know.logger import get_logger

logger = get_logger()

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_model_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def get_model(model_name: str = DEFAULT_MODEL) -> Optional[Any]:
    """Get or create a cached embedding model (thread-safe).

    Returns None if fastembed is not installed.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _lock:
        # Double-check after acquiring lock
        if model_name in _model_cache:
            return _model_cache[model_name]

        try:
            from fastembed import TextEmbedding

            start = time.time()
            model = TextEmbedding(model_name=model_name)
            elapsed = time.time() - start
            logger.debug(f"Loaded embedding model '{model_name}' in {elapsed:.2f}s")
            _model_cache[model_name] = model
            return model
        except ImportError:
            logger.debug("fastembed not installed — embeddings unavailable")
            return None


def embed_text(text: str, model_name: str = DEFAULT_MODEL, max_chars: int = 8000) -> Optional[bytes]:
    """Embed text and return raw float32 bytes, or None if unavailable."""
    model = get_model(model_name)
    if model is None:
        return None

    try:
        import numpy as np

        text = text[:max_chars]
        emb = np.array(list(model.embed([text]))[0], dtype=np.float32)
        return emb.tobytes()
    except Exception:
        return None


def embed_numpy(text: str, model_name: str = DEFAULT_MODEL, max_chars: int = 8000):
    """Embed text and return numpy array, or None if unavailable."""
    model = get_model(model_name)
    if model is None:
        return None

    try:
        import numpy as np

        text = text[:max_chars]
        return np.array(list(model.embed([text]))[0], dtype=np.float32)
    except Exception:
        return None


def is_available(model_name: str = DEFAULT_MODEL) -> bool:
    """Check if embedding model is available."""
    return get_model(model_name) is not None
