#!/usr/bin/env python3
"""Suite 2: Session Dedup Effectiveness — measures cross-query deduplication.

Runs 3 overlapping queries with and without --session to measure token savings
from deduplication.
"""

import time
import uuid

from conftest import (
    SESSION_QUERIES, V7_CONTEXT_BUDGET,
    run_know, get_know_version, save_results,
)


def run_queries_no_session(queries: list[str], budget: int) -> list[dict]:
    """Run queries without session (independent, no dedup)."""
    results = []
    for query in queries:
        t0 = time.monotonic()
        data = run_know(["context", query, "--budget", str(budget)])
        elapsed = time.monotonic() - t0

        tokens = data.get("used_tokens", 0) if "error" not in data else 0
        chunks = len(data.get("code", [])) if "error" not in data else 0
        chunk_names = [c["name"] for c in data.get("code", [])] if "error" not in data else []

        results.append({
            "query": query,
            "tokens": tokens,
            "chunks": chunks,
            "chunk_names": chunk_names,
            "elapsed_s": round(elapsed, 3),
        })
    return results


def run_queries_with_session(queries: list[str], budget: int) -> list[dict]:
    """Run queries with a shared session (dedup across queries)."""
    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    results = []

    for i, query in enumerate(queries):
        t0 = time.monotonic()
        # First query uses 'auto' to create session, rest reuse the session ID
        session_arg = session_id if i > 0 else "auto"
        data = run_know([
            "context", query, "--budget", str(budget), "--session", session_arg,
        ])
        elapsed = time.monotonic() - t0

        tokens = data.get("used_tokens", 0) if "error" not in data else 0
        chunks = len(data.get("code", [])) if "error" not in data else 0
        chunk_names = [c["name"] for c in data.get("code", [])] if "error" not in data else []

        # Extract session ID from first response if using 'auto'
        if i == 0 and "session_id" in data:
            session_id = data["session_id"]

        results.append({
            "query": query,
            "tokens": tokens,
            "chunks": chunks,
            "chunk_names": chunk_names,
            "session_id": session_id,
            "elapsed_s": round(elapsed, 3),
        })

    return results


def run_suite():
    """Run session dedup benchmark."""
    print("=" * 60)
    print("Suite 2: Session Dedup Effectiveness")
    print("=" * 60)

    version = get_know_version()

    print("\n  Running without session...")
    no_session = run_queries_no_session(SESSION_QUERIES, V7_CONTEXT_BUDGET)
    for r in no_session:
        print(f"    [{r['query'][:30]}] {r['tokens']:,} tokens, {r['chunks']} chunks")

    print("\n  Running with session (dedup)...")
    with_session = run_queries_with_session(SESSION_QUERIES, V7_CONTEXT_BUDGET)
    for r in with_session:
        print(f"    [{r['query'][:30]}] {r['tokens']:,} tokens, {r['chunks']} chunks")

    # Compute per-query savings
    per_query = []
    for ns, ws in zip(no_session, with_session):
        savings_pct = round((1 - ws["tokens"] / ns["tokens"]) * 100, 1) if ns["tokens"] > 0 else 0
        per_query.append({
            "query": ns["query"],
            "no_session_tokens": ns["tokens"],
            "with_session_tokens": ws["tokens"],
            "savings_pct": savings_pct,
        })

    total_no = sum(r["tokens"] for r in no_session)
    total_ws = sum(r["tokens"] for r in with_session)
    total_savings = round((1 - total_ws / total_no) * 100, 1) if total_no > 0 else 0

    print(f"\n  Total without session: {total_no:,} tokens")
    print(f"  Total with session:    {total_ws:,} tokens")
    print(f"  Total savings:         {total_savings}%")

    save_results("session_dedup.json", {
        "suite": "session_dedup",
        "version": version,
        "budget": V7_CONTEXT_BUDGET,
        "queries": SESSION_QUERIES,
        "no_session": no_session,
        "with_session": with_session,
        "per_query": per_query,
        "summary": {
            "total_no_session": total_no,
            "total_with_session": total_ws,
            "total_savings_pct": total_savings,
        },
    })

    return per_query


if __name__ == "__main__":
    run_suite()
