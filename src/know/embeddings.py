"""Centralized embedding model management.

Single process-wide cache for fastembed models. All consumers
(context_engine, semantic_search, knowledge_base) use this instead
of maintaining separate caches.

fastembed is optional — this module gracefully returns None when
it's not installed.
"""

import threading
import time
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from know.logger import get_logger

logger = get_logger()

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_model_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def _configure_fastembed_cache_dir() -> None:
    """Prefer a stable cache path over ephemeral temp directories."""
    cache_root = Path.home() / ".cache" / "know-cli" / "fastembed"
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(cache_root))


def _extract_fastembed_model_dir(err_text: str) -> Optional[Path]:
    """Best-effort extraction of the model cache directory from fastembed errors."""
    m = re.search(r"Load model from\s+(.+?model_optimized\.onnx)", err_text, flags=re.IGNORECASE)
    if not m:
        return None
    onnx_path = Path(m.group(1).strip())

    # Find ".../fastembed_cache/<model_dir>/..." and remove <model_dir>.
    parts = onnx_path.parts
    for idx, part in enumerate(parts):
        if "fastembed_cache" in part.lower() and idx + 1 < len(parts):
            return Path(*parts[: idx + 2])
    return onnx_path.parent


def _is_fastembed_cache_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "model_optimized.onnx" in text
        or "file doesn't exist" in text
        or "local file sizes do not match" in text
    )


def _repair_fastembed_cache(exc: Exception) -> bool:
    """Attempt one-shot repair for corrupted/incomplete fastembed model cache."""
    if not _is_fastembed_cache_error(exc):
        return False
    err_text = str(exc)
    model_dir = _extract_fastembed_model_dir(err_text)
    if not model_dir:
        return False

    try:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
            logger.warning(f"Removed corrupted fastembed model cache: {model_dir}")
            return True
    except Exception as e:
        logger.debug(f"Failed to clear fastembed cache dir '{model_dir}': {e}")
    return False


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

            _configure_fastembed_cache_dir()
            start = time.time()
            try:
                model = TextEmbedding(model_name=model_name)
            except Exception as e:
                # Recover once from corrupted cache and retry initialization.
                if _repair_fastembed_cache(e):
                    model = TextEmbedding(model_name=model_name)
                else:
                    raise
            elapsed = time.time() - start
            logger.debug(f"Loaded embedding model '{model_name}' in {elapsed:.2f}s")
            _model_cache[model_name] = model
            return model
        except ImportError:
            logger.debug("fastembed not installed — embeddings unavailable")
            return None
        except Exception as e:
            logger.warning(f"Embedding model '{model_name}' unavailable: {e}")
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



def is_available(model_name: str = DEFAULT_MODEL) -> bool:
    """Check if embedding model is available."""
    return get_model(model_name) is not None
