"""Path filtering helpers for hard excludes.

These excludes are always applied regardless of user config, to keep index
quality focused on source code rather than generated/runtime artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


_HARD_EXCLUDE_EXACT = {
    ".git",
    ".know",
    "__pycache__",
    "node_modules",
    "site-packages",
    ".next",
    ".nuxt",
    ".turbo",
    "dist",
    "build",
    "target",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
}

_HARD_EXCLUDE_PREFIXES = (
    ".venv",
    "venv",
)


def is_hard_excluded_part(part: str) -> bool:
    """Return True when a single path component should be hard-excluded."""
    token = (part or "").strip().lower()
    if not token:
        return False
    if token in _HARD_EXCLUDE_EXACT:
        return True
    return any(token.startswith(prefix) for prefix in _HARD_EXCLUDE_PREFIXES)


def is_hard_excluded_path(path_or_parts: Path | str | Iterable[str]) -> bool:
    """Return True when a path contains hard-excluded components."""
    if isinstance(path_or_parts, Path):
        parts = path_or_parts.parts
    elif isinstance(path_or_parts, str):
        parts = Path(path_or_parts).parts
    else:
        parts = tuple(path_or_parts)
    return any(is_hard_excluded_part(part) for part in parts)

