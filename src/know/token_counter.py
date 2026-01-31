"""Token counting for context budget management.

Uses a simple word-based estimation that's fast and dependency-free.
Roughly approximates GPT/Claude tokenization (~1.3 tokens per word).
"""

import re
from typing import Optional


# Average tokens-per-word ratio for code (empirically ~1.3 for Python/JS)
CODE_TOKENS_PER_WORD = 1.3
# For natural language text
TEXT_TOKENS_PER_WORD = 1.2
# Overhead per line for whitespace/indentation tokens
TOKENS_PER_LINE_OVERHEAD = 0.5


def count_tokens(text: str, mode: str = "code") -> int:
    """Estimate token count for a string.
    
    Args:
        text: The text to count tokens for.
        mode: "code" for source code, "text" for natural language.
    
    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    
    # Split on whitespace and punctuation boundaries (how tokenizers roughly work)
    # This counts words + standalone punctuation/operators
    words = re.findall(r'\w+|[^\w\s]', text)
    word_count = len(words)
    
    line_count = text.count('\n') + 1
    
    if mode == "code":
        tokens = word_count * CODE_TOKENS_PER_WORD + line_count * TOKENS_PER_LINE_OVERHEAD
    else:
        tokens = word_count * TEXT_TOKENS_PER_WORD
    
    return max(1, int(tokens))


def truncate_to_budget(text: str, budget: int, mode: str = "code") -> str:
    """Truncate text to fit within a token budget.
    
    Args:
        text: Text to truncate.
        budget: Maximum tokens.
        mode: "code" or "text".
    
    Returns:
        Truncated text that fits within budget.
    """
    current = count_tokens(text, mode)
    if current <= budget:
        return text
    
    # Binary search for the right cutoff point
    lines = text.split('\n')
    lo, hi = 0, len(lines)
    
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = '\n'.join(lines[:mid])
        if count_tokens(candidate, mode) <= budget:
            lo = mid
        else:
            hi = mid - 1
    
    if lo == 0:
        # Even one line exceeds budget — truncate by characters
        ratio = budget / max(current, 1)
        char_limit = int(len(text) * ratio * 0.9)  # 10% safety margin
        return text[:char_limit] + "\n... [truncated]"
    
    result = '\n'.join(lines[:lo])
    if lo < len(lines):
        result += "\n... [truncated]"
    return result


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
