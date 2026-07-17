"""File category detection and demotion for search ranking.

Classifies files as source, test, vendor, or generated.
Applies score multipliers so non-source files rank lower.
"""

import fnmatch
import re
from pathlib import Path

_TEST_PATTERNS = [
    "test_*", "*_test.*", "tests/*", "spec/*", "__tests__/*", "*_spec.*",
    "conftest.py", "test/*",
]

_VENDOR_PATTERNS = [
    "vendor/*", "third_party/*", "node_modules/*", ".venv/*",
    "venv/*", "dist/*", "build/*", ".tox/*",
]

_GENERATED_PATTERNS = [
    "*_generated.*", "*.pb.*", "*_pb2.py", "generated/*",
    "*.auto.*", "*_gen.*",
]

DEMOTION_MULTIPLIERS = {
    "source": 1.0,
    "test": 0.3,
    "vendor": 0.1,
    "generated": 0.1,
}


def categorize_file(
    file_path: str | Path, root: str | Path | None = None,
) -> str:
    """Classify a file path as 'source', 'test', 'vendor', or 'generated'.

    Uses fnmatch patterns against path segments and full path. When an
    absolute path is supplied, callers should also supply ``root`` so build
    directories above the repository are not mistaken for repository paths.
    """
    path = Path(file_path)
    if root is not None and path.is_absolute() and path.is_relative_to(Path(root)):
        path = path.relative_to(Path(root))

    # Normalize separators
    normalized = str(path).replace("\\", "/")
    parts = normalized.split("/")

    for pattern in _VENDOR_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern) or any(
            fnmatch.fnmatch(p + "/", pattern) for p in parts
        ):
            return "vendor"

    for pattern in _GENERATED_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(parts[-1], pattern):
            return "generated"

    for pattern in _TEST_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(parts[-1], pattern):
            return "test"
        # Check if any directory segment matches
        if any(fnmatch.fnmatch(p + "/", pattern) for p in parts[:-1]):
            return "test"

    return "source"


def get_demotion(file_path: str) -> float:
    """Get the score multiplier for a file path.

    Returns 1.0 for source files, lower for test/vendor/generated.
    """
    return DEMOTION_MULTIPLIERS[categorize_file(file_path)]


def apply_category_demotion(
    chunks: list[dict], query: str = "",
    root: str | Path | None = None,
) -> list[dict]:
    """Apply category-based score demotion to search results.

    If query contains the standalone word 'test' or 'tests', skips test file
    demotion.
    Modifies the 'score' key of each chunk dict in-place.
    """
    skip_test_demotion = bool(re.search(r"\btests?\b", query, flags=re.IGNORECASE))

    for chunk in chunks:
        fp = chunk.get("file_path", "")
        category = categorize_file(fp, root=root)
        if category == "test" and skip_test_demotion:
            continue
        multiplier = DEMOTION_MULTIPLIERS.get(category, 1.0)
        if "score" in chunk:
            chunk["score"] = chunk["score"] * multiplier

    return chunks
