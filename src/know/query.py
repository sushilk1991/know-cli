"""Query understanding for intelligent FTS5 search.

Transforms raw natural-language queries (from agents or humans) into
structured search plans.  Handles stop-word removal, identifier
detection, CamelCase/snake_case splitting, and query type classification.
"""

import re
from dataclasses import dataclass, field
from typing import List, Literal, Set

# ---------------------------------------------------------------------------
# Stop words — common agent/human filler that matches everything in FTS5
# ---------------------------------------------------------------------------
_STOP_WORDS: Set[str] = {
    # English stop words
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "if", "then", "else", "so", "but", "and", "or", "not", "no",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "into", "about", "between", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over",
    "under", "again", "further", "once", "than", "too", "very",
    "just", "also", "now", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
    # Agent action words (agents say "help me fix the bug")
    "help", "fix", "find", "show", "tell", "explain", "describe",
    "get", "give", "make", "let", "try", "need", "want", "look",
    "see", "know", "think", "use", "go", "come", "take", "put",
    "add", "create", "update", "change", "modify", "remove", "delete",
    "write", "read", "run", "start", "stop", "set", "check", "test",
    # Agent filler phrases (whole words)
    "please", "thanks", "code", "file", "function", "class", "method",
    "implement", "implementation", "work", "working", "works",
    "understand", "understanding", "related", "relevant",
}

# Prefixes agents prepend: "help me find", "show me the", "how do I"
_AGENT_PREFIXES = [
    r"^help\s+me\s+",
    r"^show\s+me\s+(?:the\s+)?",
    r"^find\s+(?:the\s+)?",
    r"^how\s+(?:do\s+(?:i|we)\s+)?",
    r"^where\s+(?:is|are)\s+(?:the\s+)?",
    r"^what\s+(?:is|are)\s+(?:the\s+)?",
    r"^can\s+you\s+",
    r"^i\s+need\s+(?:to\s+)?",
    r"^i\s+want\s+(?:to\s+)?",
    r"^let'?s?\s+",
    r"^please\s+",
]

# Regex for detecting code identifiers
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_CAMEL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*$")
_DOTTED_PATH_RE = re.compile(r"^[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+$")


QueryType = Literal["identifier", "concept", "error"]


@dataclass
class SearchPlan:
    """Result of query analysis — drives search strategy."""

    original: str               # raw query from user/agent
    terms: List[str]            # meaningful search terms (stop words removed)
    identifiers: List[str]      # detected code identifiers (exact match)
    expanded_terms: List[str]   # extra terms from CamelCase/snake_case split
    query_type: QueryType       # drives budget allocation + search strategy
    all_search_terms: List[str] = field(default_factory=list)  # union for FTS

    def __post_init__(self):
        # Build combined search terms: identifiers first, then terms, then expanded
        seen: set = set()
        combined: list = []
        for t in self.identifiers + self.terms + self.expanded_terms:
            low = t.lower()
            if low not in seen and len(low) >= 2:
                seen.add(low)
                combined.append(t)
        self.all_search_terms = combined


def analyze_query(query: str) -> SearchPlan:
    """Analyze a natural-language query into a structured SearchPlan.

    >>> plan = analyze_query("fix the auth bug")
    >>> plan.terms
    ['auth', 'bug']
    >>> plan.query_type
    'concept'

    >>> plan = analyze_query("verify_session")
    >>> plan.identifiers
    ['verify_session']
    >>> plan.query_type
    'identifier'
    """
    original = query.strip()
    cleaned = original

    # Step 1: Strip agent prefixes
    for pattern in _AGENT_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Step 2: Tokenize
    raw_tokens = re.findall(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", cleaned)

    # Step 3: Detect identifiers and classify
    identifiers: List[str] = []
    regular_tokens: List[str] = []

    for token in raw_tokens:
        if _is_identifier(token):
            identifiers.append(token)
        else:
            regular_tokens.append(token)

    # Step 4: Remove stop words from regular tokens
    terms = [t for t in regular_tokens if t.lower() not in _STOP_WORDS]

    # Step 5: Expand CamelCase / snake_case identifiers
    expanded: List[str] = []
    for ident in identifiers:
        parts = _split_identifier(ident)
        for p in parts:
            if p.lower() not in _STOP_WORDS and p.lower() != ident.lower():
                expanded.append(p)

    # Step 6: Fallback — if we stripped everything, use original tokens
    if not terms and not identifiers:
        terms = [t for t in raw_tokens if len(t) >= 2]
    # If still empty, use the whole query as one term
    if not terms and not identifiers:
        terms = [original] if original else []

    # Step 7: Classify query type
    query_type = _classify_query(original, identifiers, terms)

    return SearchPlan(
        original=original,
        terms=terms,
        identifiers=identifiers,
        expanded_terms=expanded,
        query_type=query_type,
    )


def _is_identifier(token: str) -> bool:
    """Check if a token looks like a code identifier."""
    # Dotted paths: know.daemon_db.search_chunks
    if _DOTTED_PATH_RE.match(token):
        return True
    # snake_case: verify_session, build_fts_query
    if _SNAKE_CASE_RE.match(token):
        return True
    # CamelCase: AuthMiddleware, DaemonDB
    if _CAMEL_CASE_RE.match(token):
        return True
    # Contains underscore anywhere (likely code)
    if "_" in token and len(token) > 2:
        return True
    return False


def _split_identifier(ident: str) -> List[str]:
    """Split a code identifier into constituent words.

    >>> _split_identifier("AuthMiddleware")
    ['Auth', 'Middleware']
    >>> _split_identifier("verify_session")
    ['verify', 'session']
    >>> _split_identifier("know.daemon_db")
    ['know', 'daemon', 'db']
    """
    # First split on dots
    parts: List[str] = []
    for segment in ident.split("."):
        # Split on underscores
        for sub in segment.split("_"):
            if not sub:
                continue
            # Split CamelCase
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]|\d+", sub)
            if camel_parts:
                parts.extend(camel_parts)
            else:
                parts.append(sub)
    return [p for p in parts if len(p) >= 2]


def _classify_query(original: str, identifiers: List[str], terms: List[str]) -> QueryType:
    """Classify query intent to drive search strategy."""
    # If most tokens are identifiers, it's an identifier query
    if identifiers and len(identifiers) >= len(terms):
        return "identifier"

    # Error queries contain error-related words
    error_words = {"error", "exception", "traceback", "stack", "crash",
                   "fail", "failure", "broken", "wrong", "issue", "bug",
                   "raise", "raises", "throw", "throws", "panic"}
    lower_original = original.lower()
    if any(w in lower_original for w in error_words):
        return "error"

    return "concept"


def build_fts_or_query(terms: List[str], max_terms: int = 12) -> str:
    """Build an OR-based FTS5 query from analyzed terms.

    Each term is double-quoted to disable FTS5 operator interpretation.
    """
    terms = terms[:max_terms]
    if not terms:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def build_fts_and_query(terms: List[str], max_terms: int = 12) -> str:
    """Build an AND-based FTS5 query — all terms must appear.

    Returns empty string if fewer than 2 meaningful terms.
    """
    terms = terms[:max_terms]
    if len(terms) < 2:
        return ""
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)
