#!/usr/bin/env python3
"""Workflow relevance benchmark for know-cli.

Measures whether `know workflow` returns expected files for known repo-specific
queries, alongside latency and token usage. This is intentionally deterministic:
it does not use an LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from know.config import Config, load_config
from know.daemon import populate_index
from know.daemon_db import DaemonDB
from know.token_counter import count_tokens


CASES: List[Dict[str, Any]] = [
    {
        "id": "config",
        "query": "configuration loading config yaml environment variables",
        "expected_files": ["src/know/config.py"],
    },
    {
        "id": "workflow_cli",
        "query": "workflow command mode latency read only compact JSON",
        "expected_files": ["src/know/cli/agent.py"],
    },
    {
        "id": "daemon_workflow",
        "query": "daemon workflow session read_only memory capture",
        "expected_files": ["src/know/daemon.py"],
    },
    {
        "id": "stats",
        "query": "stats retrieval workflow quality metrics",
        "expected_files": ["src/know/stats.py"],
    },
    {
        "id": "git_hooks",
        "query": "git hooks post commit pre commit behavior",
        "expected_files": ["src/know/git_hooks.py"],
    },
    {
        "id": "hooks_cli",
        "query": "hooks suggest claude codex settings",
        "expected_files": ["src/know/cli/hooks.py"],
    },
    {
        "id": "search_diagnostics",
        "query": "semantic search fallback diagnostics editable install",
        "expected_files": ["src/know/cli/search.py"],
    },
    {
        "id": "memory_capture",
        "query": "memory capture workflow decision dedup",
        "expected_files": ["src/know/memory_capture.py"],
    },
    {
        "id": "context_engine",
        "query": "context engine deep call graph callers callees",
        "expected_files": ["src/know/context_engine.py"],
    },
    {
        "id": "benchmark_harness",
        "query": "benchmark dual repo quality gate payload tokens",
        "expected_files": ["benchmark/bench_dual_repo_parallel.py"],
    },
]


def _normalize_file(value: Any) -> Optional[str]:
    if not value:
        return None
    return str(value).replace("\\", "/")


def _add_file(files: set[str], value: Any) -> None:
    normalized = _normalize_file(value)
    if normalized:
        files.add(normalized)


def _iter_items(value: Any) -> Iterable[Dict[str, Any]]:
    return value if isinstance(value, list) else []


def collect_workflow_files(payload: Dict[str, Any]) -> List[str]:
    """Collect all file paths surfaced by compact or full workflow JSON."""
    files: set[str] = set()

    targets = payload.get("targets") or {}
    _add_file(files, targets.get("selected_file_path"))
    _add_file(files, targets.get("selected_file"))
    for candidate in _iter_items(targets.get("candidates")):
        _add_file(files, candidate.get("file_path"))
        _add_file(files, candidate.get("file"))

    context = payload.get("context") or {}
    for file_path in context.get("source_files") or []:
        _add_file(files, file_path)
    for snippet in _iter_items(context.get("snippets") or context.get("code")):
        _add_file(files, snippet.get("file_path"))
        _add_file(files, snippet.get("file"))

    deep = payload.get("deep") or {}
    target = deep.get("target") or {}
    _add_file(files, target.get("file_path"))
    _add_file(files, target.get("file"))
    for key in ("caller_examples", "callee_examples", "callers", "callees"):
        for item in _iter_items(deep.get(key)):
            _add_file(files, item.get("file_path"))
            _add_file(files, item.get("file"))

    return sorted(files)


def _percent(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100, 1) if denominator else 0.0


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
    return ordered[idx]


def refresh_signature_index(repo: Path, config_path: Optional[Path]) -> Dict[str, Any]:
    """Refresh the daemon signature DB used by map/context/workflow."""
    if config_path is not None:
        config = Config.load(config_path)
    else:
        cwd = Path.cwd()
        try:
            os.chdir(repo)
            config = load_config()
        finally:
            os.chdir(cwd)

    db = DaemonDB(config.root)
    count, modules = populate_index(config.root, config, db)
    return {
        "files_indexed_or_updated": count,
        "modules_scanned": len(modules),
    }


def run_workflow(
    repo: Path,
    query: str,
    *,
    mode: str,
    max_latency_ms: int,
    config_path: Optional[Path],
) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "know.cli"]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    cmd.extend([
        "--json",
        "workflow",
        query,
        "--json-compact",
        "--read-only",
        "--mode",
        mode,
        "--max-latency-ms",
        str(max_latency_ms),
    ])

    env = os.environ.copy()
    env["KNOW_NO_DAEMON"] = "1"

    started = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=max(30, int(max_latency_ms / 1000) + 20),
        env=env,
    )
    elapsed_s = time.monotonic() - started
    payload_tokens = count_tokens(result.stdout, provider="anthropic") if result.stdout else 0

    if result.returncode != 0:
        return {
            "ok": False,
            "elapsed_s": round(elapsed_s, 3),
            "payload_tokens": payload_tokens,
            "error": result.stderr.strip() or result.stdout[:500],
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "elapsed_s": round(elapsed_s, 3),
            "payload_tokens": payload_tokens,
            "error": "invalid_json",
            "raw": result.stdout[:500],
        }

    metrics = payload.get("metrics") or {}
    targets = payload.get("targets") or {}
    deep = payload.get("deep") or {}
    return {
        "ok": True,
        "elapsed_s": round(elapsed_s, 3),
        "payload_tokens": payload_tokens,
        "retrieval_tokens": int(metrics.get("total_tokens") or 0),
        "degraded_by_latency": bool(metrics.get("degraded_by_latency")),
        "deep_error": deep.get("error"),
        "selected_file": _normalize_file(
            targets.get("selected_file_path") or targets.get("selected_file"),
        ),
        "selected_symbol": targets.get("selected_symbol"),
        "files": collect_workflow_files(payload),
    }


def run_suite(
    repo: Path,
    *,
    mode: str,
    max_latency_ms: int,
    config_path: Optional[Path],
    refresh_index: bool,
    cases: Iterable[Dict[str, Any]] = CASES,
) -> Dict[str, Any]:
    index_refresh = None
    if refresh_index:
        index_refresh = refresh_signature_index(repo, config_path)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        workflow = run_workflow(
            repo,
            case["query"],
            mode=mode,
            max_latency_ms=max_latency_ms,
            config_path=config_path,
        )
        expected = [_normalize_file(path) for path in case["expected_files"]]
        expected = [path for path in expected if path]
        files = set(workflow.get("files") or [])
        selected_file = workflow.get("selected_file")
        relevance_passed = workflow.get("ok") and any(path in files for path in expected)
        selected_passed = workflow.get("ok") and selected_file in expected
        rows.append({
            "id": case["id"],
            "query": case["query"],
            "expected_files": expected,
            "relevance_passed": bool(relevance_passed),
            "selected_passed": bool(selected_passed),
            **workflow,
        })

    total = len(rows)
    relevance_hits = sum(1 for row in rows if row["relevance_passed"])
    selected_hits = sum(1 for row in rows if row["selected_passed"])
    failures = [row for row in rows if not row["relevance_passed"]]
    latencies = [float(row["elapsed_s"]) for row in rows if row.get("ok")]
    payload_tokens = sum(int(row.get("payload_tokens") or 0) for row in rows)
    retrieval_tokens = sum(int(row.get("retrieval_tokens") or 0) for row in rows)

    return {
        "suite": "workflow_relevance",
        "repo": str(repo),
        "mode": mode,
        "max_latency_ms": max_latency_ms,
        "index_refresh": index_refresh,
        "summary": {
            "cases": total,
            "relevance_hits": relevance_hits,
            "relevance_rate_pct": _percent(relevance_hits, total),
            "selected_hits": selected_hits,
            "selected_rate_pct": _percent(selected_hits, total),
            "failures": len(failures),
            "degraded_by_latency": sum(1 for row in rows if row.get("degraded_by_latency")),
            "deep_errors": sum(1 for row in rows if row.get("deep_error")),
            "payload_tokens": payload_tokens,
            "retrieval_tokens": retrieval_tokens,
            "payload_to_retrieval_ratio": (
                round(payload_tokens / retrieval_tokens, 2) if retrieval_tokens else None
            ),
            "latency_s_avg": round(mean(latencies), 3) if latencies else 0.0,
            "latency_s_p50": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
            "latency_s_p95": round(_p95(latencies), 3),
        },
        "failures": [
            {
                "id": row["id"],
                "query": row["query"],
                "expected_files": row["expected_files"],
                "selected_file": row.get("selected_file"),
                "files": row.get("files") or [],
                "error": row.get("error"),
            }
            for row in failures
        ],
        "rows": rows,
    }


def run_and_save(
    repo: Optional[Path] = None,
    output: Optional[Path] = None,
    *,
    mode: str = "implement",
    max_latency_ms: int = 2000,
    config_path: Optional[Path] = None,
    refresh_index: bool = False,
) -> Dict[str, Any]:
    repo = (repo or ROOT).resolve()
    out_path = (output or (ROOT / "benchmark" / "results" / "workflow_relevance.json")).resolve()
    result = run_suite(
        repo,
        mode=mode,
        max_latency_ms=max_latency_ms,
        config_path=config_path,
        refresh_index=refresh_index,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = result["summary"]
    print("Suite: Workflow Relevance")
    print(
        f"  Relevance: {summary['relevance_hits']}/{summary['cases']} "
        f"({summary['relevance_rate_pct']}%) | "
        f"Selected: {summary['selected_hits']}/{summary['cases']} "
        f"({summary['selected_rate_pct']}%)"
    )
    print(
        f"  Latency p50/p95: {summary['latency_s_p50']}s/"
        f"{summary['latency_s_p95']}s | Payload/retrieval: "
        f"{summary['payload_to_retrieval_ratio']}x"
    )
    print(f"  Saved: {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark know workflow relevance")
    parser.add_argument("--repo", default=str(ROOT), help="Repository path to benchmark")
    parser.add_argument(
        "--output",
        default=str(ROOT / "benchmark" / "results" / "workflow_relevance.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional know config path. Relative paths are resolved from --repo.",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=["explore", "implement", "thorough"],
        default="implement",
    )
    parser.add_argument("--max-latency-ms", type=int, default=2000)
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Refresh the signature DB before running. This writes .know/daemon.db.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    config_path = None
    if args.config:
        raw_config = Path(args.config).expanduser()
        config_path = raw_config if raw_config.is_absolute() else repo / raw_config
        config_path = config_path.resolve()

    result = run_and_save(
        repo,
        Path(args.output).expanduser().resolve(),
        mode=args.workflow_mode,
        max_latency_ms=args.max_latency_ms,
        config_path=config_path,
        refresh_index=args.refresh_index,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
