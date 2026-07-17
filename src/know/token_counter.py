"""Token counting for context budget management.

Uses tiktoken (cl100k_base) for accurate counting. Applies a calibration
factor for Anthropic models since Claude uses a different tokenizer than
OpenAI's cl100k_base.

Provider-aware: pass provider="anthropic" or provider="openai" for
the most accurate estimates.
"""

import re
from typing import Optional
from functools import lru_cache

import tiktoken

# Calibration factors: cl100k_base tokens * factor = provider tokens.
# Measured empirically against Anthropic's count_tokens API on code samples.
# Claude's tokenizer produces ~5-15% more tokens than cl100k_base for code.
_PROVIDER_CALIBRATION = {
    "anthropic": 1.10,
    "openai": 1.0,
    "default": 1.05,
}


@lru_cache(maxsize=4)
def _get_tokenizer(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Get cached tokenizer instance."""
    return tiktoken.get_encoding(encoding)


def count_tokens(
    text: str,
    provider: str = "anthropic",
    encoding: str = "cl100k_base",
) -> int:
    """Count tokens using tiktoken with provider-specific calibration.

    Args:
        text: The text to count tokens for.
        provider: LLM provider ("anthropic", "openai"). Applies calibration.
        encoding: tiktoken encoding name.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    tokenizer = _get_tokenizer(encoding)
    raw_count = len(tokenizer.encode(text))

    factor = _PROVIDER_CALIBRATION.get(provider, _PROVIDER_CALIBRATION["default"])
    return max(1, int(raw_count * factor))


def truncate_to_budget(
    text: str,
    budget: int,
    provider: str = "anthropic",
) -> str:
    """Truncate text to fit within a token budget.

    Args:
        text: Text to truncate.
        budget: Maximum tokens.
        provider: LLM provider for accurate counting.

    Returns:
        Truncated text that fits within budget.
    """
    if budget <= 0:
        return ""

    current = count_tokens(text, provider=provider)
    if current <= budget:
        return text

    marker = "\n... [truncated]"
    suffix = marker if count_tokens(marker, provider=provider) <= budget else ""

    # Binary search characters so the returned text, including its marker,
    # satisfies the advertised hard token limit even for tiny budgets.
    lo, hi = 0, len(text)

    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid] + suffix
        if count_tokens(candidate, provider=provider) <= budget:
            lo = mid
        else:
            hi = mid - 1

    return text[:lo] + suffix


def format_budget(used: int, total: int) -> str:
    """Format budget utilization string.

    Args:
        used: Tokens used.
        total: Total budget.

    Returns:
        Formatted string like "6,234 / 8,000 (78%)"
    """
    pct = int(used / total * 100) if total > 0 else 0
    return f"{used:,} / {total:,} ({pct}%)"
