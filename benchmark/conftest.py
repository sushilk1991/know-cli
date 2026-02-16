"""Shared constants and helpers for know-cli benchmarks."""

import json
import os
import subprocess
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
FARFIELD_DIR = Path("/Users/sushil/Code/Github/farfield")
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
ANSWER_KEYS_DIR = BENCHMARK_DIR / "answer_keys"

RESULTS_DIR.mkdir(exist_ok=True)

# ── Queries ────────────────────────────────────────────────────────────────────
# 5 README scenarios + 3 from benchmark_task.md
SCENARIOS = [
    {"id": "websocket",     "query": "WebSocket handling and real-time communication"},
    {"id": "auth_api_keys", "query": "authentication and API key management"},
    {"id": "model_routing", "query": "model routing and LLM provider selection"},
    {"id": "error_handling","query": "error handling and retry logic"},
    {"id": "db_storage",    "query": "database storage and persistence layer"},
    {"id": "billing",       "query": "billing subscription plans and payment processing"},
    {"id": "llm_providers", "query": "LLM provider configuration and discovery"},
    {"id": "agent_exec",    "query": "agent execution pipeline and LangGraph workflow"},
]

# Session dedup queries (overlapping billing/payments domain)
SESSION_QUERIES = [
    "billing subscription plans",
    "payment processing checkout",
    "usage limits enforcement",
]

# Agent benchmark questions (from benchmark_task.md)
AGENT_QUESTIONS = [
    {
        "id": "q1",
        "question": (
            "How does the billing/subscription system work? "
            "What model represents a subscription? What are the billing plan tiers and their limits? "
            "How are sandbox limits enforced? Name the specific files, classes, and functions involved."
        ),
    },
    {
        "id": "q2",
        "question": (
            "How does the LLM provider system work? "
            "How are LLM providers (OpenAI, Anthropic, Gemini etc.) configured and selected? "
            "What is the provider_discovery module doing? How are workspace-level model settings handled? "
            "Name the specific files, classes, and functions involved."
        ),
    },
    {
        "id": "q3",
        "question": (
            "How does the agent execution pipeline work? "
            "What is the flow from a user message to an agent response? "
            "What role does LangGraph play? How are tools/MCP servers integrated? "
            "Name the specific files, classes, and functions involved."
        ),
    },
]

# ── Budgets ────────────────────────────────────────────────────────────────────
CONTEXT_BUDGET = 8000        # v0.6.0 style: single context call
V7_CONTEXT_BUDGET = 4000     # v0.7.0 tier-2 budget
V7_DEEP_BUDGET = 3000        # v0.7.0 tier-3 budget
MAP_LIMIT = 20               # v0.7.0 tier-1 map results

# Agent config
AGENT_MODEL = "claude-sonnet-4-20250514"
AGENT_MAX_TURNS = 20
AGENT_TEMPERATURE = 0


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_know(args: list[str], cwd: Path = FARFIELD_DIR) -> dict:
    """Run a know CLI command with --json and return parsed output."""
    cmd = ["know", "--json"] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), timeout=60,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "command": " ".join(cmd)}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid JSON", "raw": result.stdout[:500], "command": " ".join(cmd)}


def run_know_text(args: list[str], cwd: Path = FARFIELD_DIR) -> str:
    """Run a know CLI command and return raw text output."""
    cmd = ["know"] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), timeout=60,
    )
    return result.stdout


def count_tokens_approx(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return len(text) // 4


def run_grep_count(query: str, cwd: Path = FARFIELD_DIR) -> dict:
    """Run grep -rn for a query and count approximate tokens of matched files."""
    words = query.lower().split()
    # Use first two significant words for grep
    grep_terms = [w for w in words if len(w) > 3][:2]
    if not grep_terms:
        grep_terms = words[:2]

    matched_files = set()
    for term in grep_terms:
        result = subprocess.run(
            ["grep", "-rln", "--include=*.py", "--include=*.ts", "--include=*.tsx",
             "--include=*.js", "--include=*.jsx", "-i", term],
            capture_output=True, text=True, cwd=str(cwd), timeout=30,
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                matched_files.add(line)

    total_tokens = 0
    file_count = 0
    for fpath in sorted(matched_files)[:20]:  # Cap at 20 files like a real agent
        full = cwd / fpath
        if full.exists():
            try:
                content = full.read_text(errors="replace")
                total_tokens += count_tokens_approx(content)
                file_count += 1
            except Exception:
                pass

    return {
        "files_matched": len(matched_files),
        "files_read": file_count,
        "tokens": total_tokens,
    }


def get_know_version() -> str:
    """Get installed know-cli version."""
    try:
        import know
        return know.__version__
    except Exception:
        return "unknown"


def get_know_status(cwd: Path = FARFIELD_DIR) -> str:
    """Get know status output."""
    return run_know_text(["status"], cwd=cwd)


def save_results(filename: str, data: dict):
    """Save benchmark results to JSON."""
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def has_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
