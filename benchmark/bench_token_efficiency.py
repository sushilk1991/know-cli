#!/usr/bin/env python3
"""Suite 1: Token Efficiency Per-Query — measures raw token output from know CLI.

Compares three strategies across 8 scenarios:
  - Grep+Read: grep matched files, count tokens
  - v0.6.0: know context (budget 8000, no session)
  - v0.7.0 3-tier: know map → know context (session) → know deep
"""

import time

from conftest import (
    SCENARIOS, CONTEXT_BUDGET, V7_CONTEXT_BUDGET, V7_DEEP_BUDGET, MAP_LIMIT,
    run_know, run_grep_count, count_tokens_approx, get_know_version,
    get_know_status, save_results,
)


def bench_grep_strategy(query: str) -> dict:
    """Grep+Read: grep for query terms, read matched files."""
    t0 = time.monotonic()
    result = run_grep_count(query)
    elapsed = time.monotonic() - t0
    return {
        "strategy": "grep_read",
        "tokens": result["tokens"],
        "files_matched": result["files_matched"],
        "files_read": result["files_read"],
        "elapsed_s": round(elapsed, 3),
    }


def bench_v6_strategy(query: str) -> dict:
    """v0.6.0: single know context call, budget 8000, no session."""
    t0 = time.monotonic()
    data = run_know(["context", query, "--budget", str(CONTEXT_BUDGET)])
    elapsed = time.monotonic() - t0

    if "error" in data:
        return {"strategy": "v6_context", "error": data["error"], "elapsed_s": round(elapsed, 3)}

    return {
        "strategy": "v6_context",
        "tokens": data.get("used_tokens", 0),
        "budget": data.get("budget", CONTEXT_BUDGET),
        "chunks": len(data.get("code", [])),
        "confidence": data.get("confidence"),
        "elapsed_s": round(elapsed, 3),
    }


def bench_v7_strategy(query: str) -> dict:
    """v0.7.0 3-tier: map → context(session) → deep(top function)."""
    total_tokens = 0
    details = {}

    # Tier 1: map
    t0 = time.monotonic()
    map_data = run_know(["map", query, "--limit", str(MAP_LIMIT)])
    t_map = time.monotonic() - t0

    map_text = ""
    if "results" in map_data:
        for r in map_data["results"]:
            map_text += f"{r.get('signature', r.get('chunk_name', ''))}\n"
    map_tokens = count_tokens_approx(map_text)
    total_tokens += map_tokens
    details["map"] = {
        "tokens": map_tokens,
        "results": map_data.get("count", 0),
        "elapsed_s": round(t_map, 3),
    }

    # Tier 2: context with session
    t0 = time.monotonic()
    ctx_data = run_know([
        "context", query, "--budget", str(V7_CONTEXT_BUDGET), "--session", "auto",
    ])
    t_ctx = time.monotonic() - t0

    ctx_tokens = ctx_data.get("used_tokens", 0) if "error" not in ctx_data else 0
    total_tokens += ctx_tokens
    details["context"] = {
        "tokens": ctx_tokens,
        "chunks": len(ctx_data.get("code", [])),
        "elapsed_s": round(t_ctx, 3),
    }

    # Tier 3: deep on top function from context
    top_fn = None
    if "code" in ctx_data and ctx_data["code"]:
        top_fn = ctx_data["code"][0].get("name")

    deep_tokens = 0
    if top_fn:
        t0 = time.monotonic()
        deep_data = run_know(["deep", top_fn, "--budget", str(V7_DEEP_BUDGET)])
        t_deep = time.monotonic() - t0

        if "error" not in deep_data:
            deep_tokens = deep_data.get("used_tokens", 0)
        details["deep"] = {
            "function": top_fn,
            "tokens": deep_tokens,
            "elapsed_s": round(t_deep, 3),
            "error": deep_data.get("error"),
        }
    else:
        details["deep"] = {"function": None, "tokens": 0, "skipped": True}

    total_tokens += deep_tokens

    return {
        "strategy": "v7_3tier",
        "tokens": total_tokens,
        "details": details,
        "elapsed_s": round(sum(
            d.get("elapsed_s", 0) for d in details.values()
        ), 3),
    }


def run_suite():
    """Run all 8 scenarios across 3 strategies."""
    print("=" * 60)
    print("Suite 1: Token Efficiency Per-Query")
    print("=" * 60)

    version = get_know_version()
    status = get_know_status()
    results = []

    for scenario in SCENARIOS:
        sid = scenario["id"]
        query = scenario["query"]
        print(f"\n  [{sid}] {query}")

        grep = bench_grep_strategy(query)
        v6 = bench_v6_strategy(query)
        v7 = bench_v7_strategy(query)

        print(f"    grep: {grep['tokens']:,} tokens ({grep.get('files_read', 0)} files)")
        print(f"    v6:   {v6.get('tokens', 'ERR'):,} tokens ({v6.get('chunks', 0)} chunks)")
        print(f"    v7:   {v7.get('tokens', 'ERR'):,} tokens")

        results.append({
            "scenario_id": sid,
            "query": query,
            "grep_read": grep,
            "v6_context": v6,
            "v7_3tier": v7,
        })

    # Compute summary
    grep_total = sum(r["grep_read"]["tokens"] for r in results)
    v6_total = sum(r["v6_context"].get("tokens", 0) for r in results)
    v7_total = sum(r["v7_3tier"].get("tokens", 0) for r in results)

    summary = {
        "total_grep_tokens": grep_total,
        "total_v6_tokens": v6_total,
        "total_v7_tokens": v7_total,
        "v6_vs_grep": round(grep_total / v6_total, 1) if v6_total else None,
        "v7_vs_grep": round(grep_total / v7_total, 1) if v7_total else None,
        "v7_vs_v6": round((1 - v7_total / v6_total) * 100, 1) if v6_total else None,
    }

    print(f"\n  Summary:")
    print(f"    Grep total:  {grep_total:,} tokens")
    print(f"    v0.6 total:  {v6_total:,} tokens ({summary['v6_vs_grep']}x reduction vs grep)")
    print(f"    v0.7 total:  {v7_total:,} tokens ({summary['v7_vs_grep']}x reduction vs grep)")
    print(f"    v0.7 vs v0.6: {summary['v7_vs_v6']}% fewer tokens")

    save_results("token_efficiency.json", {
        "suite": "token_efficiency",
        "version": version,
        "status": status,
        "scenarios": results,
        "summary": summary,
    })

    return results


if __name__ == "__main__":
    run_suite()
