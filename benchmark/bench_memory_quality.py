"""Memory quality benchmark harness for know-cli.

Evaluates structured-memory recall quality and latency on a repository.
Outputs JSON to benchmark/results/memory_quality.json by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import sys

# Ensure src imports when running from repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from know.knowledge_base import KnowledgeBase


CASES: List[Tuple[str, str, str]] = [
    (
        "decision: use single daemon architecture for index+search+memory",
        "single daemon architecture",
        "decision",
    ),
    (
        "decision: related command should trigger stale-file auto-refresh on miss",
        "refresh stale files on related miss",
        "decision",
    ),
    (
        "decision: deep command fallback should return body with call_graph flag",
        "deep fallback and call graph flag",
        "decision",
    ),
    (
        "decision: prefer RRF fusion over single-lane ranking",
        "how ranking fusion works",
        "decision",
    ),
    (
        "fact: supported extensions include tsx ts js jsx py go rs swift",
        "supported file extensions",
        "fact",
    ),
    (
        "decision: context packing should put highest utility chunks at prompt edges",
        "lost in middle mitigation",
        "decision",
    ),
    (
        "decision: mcp remember/recall should persist across sessions",
        "mcp memory persistence",
        "decision",
    ),
    (
        "fact: embeddings gracefully degrade to fts when model unavailable",
        "what happens without embedding model",
        "fact",
    ),
    (
        "decision: include import/call graph neighborhoods before rerank",
        "graph first retrieval",
        "decision",
    ),
    (
        "decision: benchmark harness should compare grep versus know-cli",
        "benchmark harness purpose",
        "decision",
    ),
]


def run(repo: Path) -> Dict[str, Any]:
    config = SimpleNamespace(root=repo)
    kb = KnowledgeBase(config)
    run_id = int(time.time() * 1000)

    created_ids: List[int] = []
    inserted: List[Tuple[str, str, str]] = []
    try:
        for idx, (text, query, memory_type) in enumerate(CASES, 1):
            tagged_text = f"{text} [bench:{run_id}:{idx}]"
            mid = kb.remember(
                text=tagged_text,
                source="bench-memory",
                tags="bench",
                memory_type=memory_type,
                trust_level="local_verified",
            )
            created_ids.append(mid)
            inserted.append((tagged_text, query, memory_type))

        top1 = 0
        top3 = 0
        top5 = 0
        latencies: List[float] = []
        rows: List[Dict[str, Any]] = []

        for text, query, memory_type in inserted:
            t0 = time.perf_counter()
            results = kb.recall(query, top_k=5, memory_type=memory_type)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            top = [r.text for r in results]
            is_top1 = bool(top) and top[0] == text
            in_top3 = text in top[:3]
            in_top5 = text in top[:5]
            top1 += int(is_top1)
            top3 += int(in_top3)
            top5 += int(in_top5)
            rows.append(
                {
                    "query": query,
                    "memory_type": memory_type,
                    "top1": is_top1,
                    "top3": in_top3,
                    "top5": in_top5,
                    "latency_ms": round(dt_ms, 2),
                    "top_result": top[0] if top else "",
                }
            )

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)
        p95 = sorted_lat[p95_idx] if sorted_lat else 0.0

        return {
            "repo": str(repo),
            "cases": len(CASES),
            "top1_hits": top1,
            "top3_hits": top3,
            "top5_hits": top5,
            "top1_rate": round((top1 / len(CASES)) * 100, 1),
            "top3_rate": round((top3 / len(CASES)) * 100, 1),
            "top5_rate": round((top5 / len(CASES)) * 100, 1),
            "latency_ms_avg": round(mean(latencies), 2) if latencies else 0.0,
            "latency_ms_p50": round(p50, 2),
            "latency_ms_p95": round(p95, 2),
            "rows": rows,
        }
    finally:
        for mid in created_ids:
            kb.forget(mid)


def run_suite(repo: Path | None = None, output: Path | None = None) -> Dict[str, Any]:
    repo = (repo or ROOT).resolve()
    out_path = (output or (ROOT / "benchmark" / "results" / "memory_quality.json")).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = run(repo)
    out_path.write_text(json.dumps(result, indent=2))
    print("Suite: Memory Quality")
    print(
        f"  Top-1: {result['top1_hits']}/{result['cases']} ({result['top1_rate']}%) | "
        f"Top-5: {result['top5_hits']}/{result['cases']} ({result['top5_rate']}%) | "
        f"Avg latency: {result['latency_ms_avg']}ms"
    )
    print(f"  Saved: {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark memory quality for know-cli")
    parser.add_argument(
        "--repo",
        default=str(ROOT),
        help="Repository path to benchmark (default: know-cli repo root)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmark" / "results" / "memory_quality.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_suite(repo=repo, output=out_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
