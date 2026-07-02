#!/usr/bin/env python3
"""Parallel dual-repo benchmark: grep+read baseline vs know workflow.

Runs two "agents" in parallel per query:
  - grep_read: keyword grep + full file reads
  - know_single_workflow: know workflow (single CLI call)

Outputs:
  - benchmark/results/dual_repo_parallel.json
  - benchmark/results/DUAL_REPO_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from know.token_counter import count_tokens


RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_REPOS = [
    Path("/Users/sushil/Code/Github/know-cli"),
    Path("/Users/sushil/Code/Github/farfield"),
]

COMMON_QUERIES = [
    "configuration loading and environment variable handling",
    "database schema and persistence layer",
    "module dependency graph and call graph tracking",
    "error handling and retry logic",
    "workflow command availability and command registration",
    "memory recall and decision capture flow",
    "background daemon indexing and cache refresh behavior",
    "api route validation and schema handling",
    "session deduplication and token budget reuse",
    "retry/backoff behavior in external service integrations",
]

CODE_GLOBS = [
    "*.py",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.go",
    "*.rs",
    "*.swift",
]

KNOW_CMD = [sys.executable, "-m", "know.cli"]

MODE_DEFAULTS = {
    "explore": {"map_limit": 30, "context_budget": 3500, "deep_budget": 0, "max_latency_ms": 2500},
    "implement": {"map_limit": 20, "context_budget": 4000, "deep_budget": 3000, "max_latency_ms": 6000},
    "thorough": {"map_limit": 30, "context_budget": 6000, "deep_budget": 4500, "max_latency_ms": 15000},
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_WORKFLOW_MODE = os.environ.get("KNOW_BENCH_WORKFLOW_MODE", "thorough")
if DEFAULT_WORKFLOW_MODE not in MODE_DEFAULTS:
    DEFAULT_WORKFLOW_MODE = "thorough"
DEFAULT_WORKFLOW_DEFAULTS = MODE_DEFAULTS[DEFAULT_WORKFLOW_MODE]
DEFAULT_MAX_LATENCY_MS = _env_int(
    "KNOW_BENCH_MAX_LATENCY_MS",
    DEFAULT_WORKFLOW_DEFAULTS["max_latency_ms"],
)
DEFAULT_MAP_LIMIT = _env_int("KNOW_BENCH_MAP_LIMIT", DEFAULT_WORKFLOW_DEFAULTS["map_limit"])
DEFAULT_CONTEXT_BUDGET = _env_int(
    "KNOW_BENCH_CONTEXT_BUDGET",
    DEFAULT_WORKFLOW_DEFAULTS["context_budget"],
)
DEFAULT_DEEP_BUDGET = _env_int("KNOW_BENCH_DEEP_BUDGET", DEFAULT_WORKFLOW_DEFAULTS["deep_budget"])

STOP_WORDS = {
    "and", "or", "the", "how", "what", "where", "when", "why", "with", "from",
    "into", "onto", "for", "this", "that", "between", "across", "about", "using",
}


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def run_cmd(cmd: List[str], cwd: Path, timeout: int = 90) -> CommandResult:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CommandResult(result.returncode, result.stdout, result.stderr)


def percentile(values: List[float], p: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def stop_project_daemon(repo: Path) -> None:
    """Best-effort: stop know daemon for repo to measure cold-start latency."""
    try:
        from know.daemon import pid_path

        pf = pid_path(repo)
        if not pf.exists():
            return
        pid = int(pf.read_text().strip())
        os.kill(pid, 15)
        for _ in range(80):
            if not pf.exists():
                return
            time.sleep(0.05)
    except Exception:
        return


def tokenize_query(query: str, max_terms: int = 4) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]+", query.lower())
    terms: List[str] = []
    for token in raw:
        if token in STOP_WORDS:
            continue
        if len(token) <= 2:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms or raw[:max_terms] or [query]


def _parse_know_json(result: CommandResult) -> Dict[str, Any]:
    if result.returncode != 0:
        return {"error": result.stderr.strip() or "command failed"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": result.stdout[:500]}


def run_know_agent(
    repo: Path,
    query: str,
    *,
    mode: str = DEFAULT_WORKFLOW_MODE,
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
    map_limit: int = DEFAULT_MAP_LIMIT,
    context_budget: int = DEFAULT_CONTEXT_BUDGET,
    deep_budget: int = DEFAULT_DEEP_BUDGET,
    read_only: bool = True,
) -> Dict[str, Any]:
    start = time.monotonic()
    workflow_t0 = time.monotonic()
    cmd = [
        *KNOW_CMD, "--json", "workflow", query,
        "--json-compact",
        "--mode", mode,
        "--max-latency-ms", str(max_latency_ms),
        "--map-limit", str(map_limit),
        "--context-budget", str(context_budget),
        "--deep-budget", str(deep_budget),
        "--session", "auto",
    ]
    if read_only:
        cmd.append("--read-only")
    workflow_raw = run_cmd(cmd, cwd=repo)
    workflow_elapsed = time.monotonic() - workflow_t0
    workflow_data = _parse_know_json(workflow_raw)

    map_data = workflow_data.get("map", {}) if isinstance(workflow_data, dict) else {}
    ctx_data = workflow_data.get("context", {}) if isinstance(workflow_data, dict) else {}
    deep_data = workflow_data.get("deep", {}) if isinstance(workflow_data, dict) else {}
    metrics = workflow_data.get("metrics", {}) if isinstance(workflow_data, dict) else {}

    map_tokens = int(map_data.get("tokens", metrics.get("map_tokens", 0)) or 0)
    ctx_tokens = int(ctx_data.get("used_tokens", metrics.get("context_tokens", 0)) or 0)
    deep_tokens = int(
        deep_data.get(
            "budget_used",
            deep_data.get("used_tokens", metrics.get("deep_tokens", 0)),
        ) or 0
    )
    retrieval_tokens = int(
        workflow_data.get(
            "total_tokens",
            metrics.get("total_tokens", map_tokens + ctx_tokens + deep_tokens),
        ) or 0
    )
    if not map_tokens and retrieval_tokens:
        map_tokens = max(0, retrieval_tokens - ctx_tokens - deep_tokens)
    payload_bytes = len((workflow_raw.stdout or "").encode("utf-8"))
    payload_tokens = count_tokens(workflow_raw.stdout or "", provider="anthropic")
    elapsed = time.monotonic() - start

    tool_calls = 1
    if map_data.get("error") or ctx_data.get("error") or deep_data.get("error"):
        # Keep metric honest if workflow failed and caller retries manually in practice.
        tool_calls = 1

    deep_error = deep_data.get("error") if isinstance(deep_data, dict) else None
    target = deep_data.get("target") if isinstance(deep_data, dict) else {}
    if not isinstance(target, dict):
        target = {}

    def _edge_count(value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    fallback_triggered = bool(
        workflow_data.get("error")
        or map_data.get("error")
        or ctx_data.get("error")
        or deep_error
    )

    return {
        "strategy": "know_single_workflow",
        "query": query,
        "tool_calls": tool_calls,
        "elapsed_s": round(elapsed, 3),
        "tokens": payload_tokens,
        "retrieval_tokens": retrieval_tokens,
        "payload_bytes": payload_bytes,
        "payload_tokens": payload_tokens,
        "fallback_triggered": fallback_triggered,
        "workflow_mode": metrics.get("mode", workflow_data.get("workflow_mode", mode)),
        "latency_budget_ms": metrics.get(
            "latency_budget_ms",
            workflow_data.get("latency_budget_ms", max_latency_ms),
        ),
        "read_only": read_only,
        "map_tokens": map_tokens,
        "context_tokens": ctx_tokens,
        "deep_tokens": deep_tokens,
        "steps": {
            "workflow_s": round(workflow_elapsed, 3),
        },
        "call_graph": {
            "available": deep_data.get("call_graph_available") if isinstance(deep_data, dict) else None,
            "reason": deep_data.get("call_graph_reason") if isinstance(deep_data, dict) else None,
            "callers": _edge_count(deep_data.get("callers")) if isinstance(deep_data, dict) else 0,
            "callees": _edge_count(deep_data.get("callees")) if isinstance(deep_data, dict) else 0,
            "target": target.get("name"),
        },
        "errors": {
            "workflow": workflow_data.get("error") if isinstance(workflow_data, dict) else "unknown",
            "map": map_data.get("error") if isinstance(map_data, dict) else None,
            "context": ctx_data.get("error") if isinstance(ctx_data, dict) else None,
            "deep": deep_error,
        },
    }


def _read_file_tokens(path: Path) -> int:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    return count_tokens(content, provider="anthropic")


def run_grep_agent(repo: Path, query: str, max_files: int = 12) -> Dict[str, Any]:
    start = time.monotonic()
    terms = tokenize_query(query)

    match_counts: Dict[str, int] = {}
    grep_calls = 0

    for term in terms:
        cmd = ["rg", "--line-number", "--ignore-case", "--no-heading"]
        for glob in CODE_GLOBS:
            cmd.extend(["-g", glob])
        cmd.extend([term, "."])
        result = run_cmd(cmd, cwd=repo, timeout=60)
        grep_calls += 1
        if result.returncode not in (0, 1):
            continue
        for line in result.stdout.splitlines():
            path = line.split(":", 1)[0].strip()
            if path:
                match_counts[path] = match_counts.get(path, 0) + 1

    ranked_paths = sorted(match_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_files = [repo / item[0] for item in ranked_paths[:max_files]]

    read_tokens = 0
    files_read = 0
    for fpath in top_files:
        if fpath.exists() and fpath.is_file():
            read_tokens += _read_file_tokens(fpath)
            files_read += 1

    elapsed = time.monotonic() - start
    return {
        "strategy": "grep_read",
        "query": query,
        "tool_calls": grep_calls + files_read,
        "elapsed_s": round(elapsed, 3),
        "tokens": read_tokens,
        "search_terms": terms,
        "matched_files": len(match_counts),
        "files_read": files_read,
        "max_files": max_files,
    }


def benchmark_repo(
    repo: Path,
    queries: List[str],
    *,
    mode: str = DEFAULT_WORKFLOW_MODE,
    max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
    map_limit: int = DEFAULT_MAP_LIMIT,
    context_budget: int = DEFAULT_CONTEXT_BUDGET,
    deep_budget: int = DEFAULT_DEEP_BUDGET,
    read_only: bool = True,
) -> Dict[str, Any]:
    if not repo.exists():
        return {
            "repo": str(repo),
            "error": "repo_not_found",
            "queries": [],
            "summary": {},
        }

    # Cold-start probe (daemon stopped) to quantify first-hit penalty.
    cold_probe = None
    try:
        stop_project_daemon(repo)
        cold_probe = run_know_agent(
            repo,
            queries[0],
            mode=mode,
            max_latency_ms=max_latency_ms,
            map_limit=map_limit,
            context_budget=context_budget,
            deep_budget=deep_budget,
            read_only=read_only,
        )
    except Exception:
        cold_probe = None

    # Warm know index/cache so measured query latency is steady-state.
    _ = run_cmd([*KNOW_CMD, "--json", "status"], cwd=repo, timeout=120)

    know_version = None
    status_result = run_cmd([*KNOW_CMD, "--json", "status"], cwd=repo, timeout=120)
    status_data = _parse_know_json(status_result)
    if isinstance(status_data, dict):
        know_version = status_data.get("version")

    # Warm-up one workflow call so measured queries are steady-state.
    try:
        run_know_agent(
            repo,
            "__warmup__ indexing readiness",
            mode=mode,
            max_latency_ms=max_latency_ms,
            map_limit=map_limit,
            context_budget=context_budget,
            deep_budget=deep_budget,
            read_only=read_only,
        )
    except Exception:
        pass

    query_rows = []
    for query in queries:
        with ThreadPoolExecutor(max_workers=2) as pool:
            know_future = pool.submit(
                run_know_agent,
                repo,
                query,
                mode=mode,
                max_latency_ms=max_latency_ms,
                map_limit=map_limit,
                context_budget=context_budget,
                deep_budget=deep_budget,
                read_only=read_only,
            )
            grep_future = pool.submit(run_grep_agent, repo, query)
            know_row = know_future.result()
            grep_row = grep_future.result()

        know_error = bool((know_row.get("errors") or {}).get("workflow"))
        token_savings = None if know_error else grep_row["tokens"] - know_row["tokens"]
        token_savings_pct = (
            round((token_savings / grep_row["tokens"]) * 100, 1)
            if token_savings is not None and grep_row["tokens"]
            else None
        )

        query_rows.append({
            "query": query,
            "know": know_row,
            "grep": grep_row,
            "delta": {
                "token_savings": token_savings,
                "token_savings_pct": token_savings_pct,
                "latency_ratio_know_over_grep": round(
                    know_row["elapsed_s"] / grep_row["elapsed_s"], 2,
                ) if grep_row["elapsed_s"] else None,
                "tool_call_reduction_pct": round(
                    ((grep_row["tool_calls"] - know_row["tool_calls"]) / grep_row["tool_calls"]) * 100,
                    1,
                ) if grep_row["tool_calls"] else None,
            },
        })

    know_tokens = sum(q["know"]["tokens"] for q in query_rows)
    grep_tokens = sum(q["grep"]["tokens"] for q in query_rows)
    know_time = sum(q["know"]["elapsed_s"] for q in query_rows)
    grep_time = sum(q["grep"]["elapsed_s"] for q in query_rows)
    know_calls = sum(q["know"]["tool_calls"] for q in query_rows)
    grep_calls = sum(q["grep"]["tool_calls"] for q in query_rows)
    know_payload_bytes = sum(q["know"].get("payload_bytes", 0) for q in query_rows)
    fallback_count = sum(1 for q in query_rows if q["know"].get("fallback_triggered"))
    error_count = sum(1 for q in query_rows if (q["know"].get("errors") or {}).get("workflow"))
    successful_rows = [
        q for q in query_rows
        if not (q["know"].get("errors") or {}).get("workflow")
    ]
    quality_rows = [
        q for q in successful_rows
        if not q["know"].get("fallback_triggered")
    ]
    deep_rows = [q["know"]["call_graph"] for q in query_rows]
    deep_available = [d for d in deep_rows if d.get("available") is True]
    non_empty_deep = [
        d for d in deep_rows if (d.get("callers", 0) + d.get("callees", 0)) > 0
    ]
    know_warm_latencies = [q["know"]["elapsed_s"] for q in query_rows]

    summary = {
        "workflow_mode": mode,
        "latency_budget_ms": max_latency_ms,
        "read_only": read_only,
        "know_tokens_total": know_tokens,
        "grep_tokens_total": grep_tokens,
        "know_payload_bytes_total": know_payload_bytes,
        "know_time_total_s": round(know_time, 3),
        "grep_time_total_s": round(grep_time, 3),
        "know_warm_p50_s": round(percentile(know_warm_latencies, 0.5), 3),
        "know_warm_p95_s": round(percentile(know_warm_latencies, 0.95), 3),
        "know_cold_start_s": round(float(cold_probe["elapsed_s"]), 3) if cold_probe else None,
        "know_tool_calls_total": know_calls,
        "grep_tool_calls_total": grep_calls,
        "fallback_count": fallback_count,
        "error_count": error_count,
        "quality_gate_passed": error_count == 0 and fallback_count == 0,
        "success_rate_pct": round(
            ((len(query_rows) - error_count) / len(query_rows)) * 100,
            1,
        ) if query_rows else 0.0,
        "fallback_rate_pct": round((fallback_count / len(query_rows)) * 100, 1) if query_rows else 0.0,
        "token_reduction_pct": round(
            ((grep_tokens - know_tokens) / grep_tokens) * 100, 1
        ) if grep_tokens and error_count == 0 else None,
        "quality_adjusted_token_reduction_pct": round(
            (
                (
                    sum(q["grep"]["tokens"] for q in quality_rows)
                    - sum(q["know"]["tokens"] for q in quality_rows)
                )
                / sum(q["grep"]["tokens"] for q in quality_rows)
            )
            * 100,
            1,
        ) if quality_rows and sum(q["grep"]["tokens"] for q in quality_rows) else None,
        "latency_ratio_know_over_grep": round(
            know_time / grep_time, 2
        ) if grep_time else None,
        "tool_call_reduction_pct": round(
            ((grep_calls - know_calls) / grep_calls) * 100, 1
        ) if grep_calls else None,
        "lookup_call_reduction_pct": round(
            ((grep_calls - know_calls) / grep_calls) * 100, 1
        ) if grep_calls else None,
        "deep_call_graph_available_rate": round(
            (len(deep_available) / len(deep_rows)) * 100, 1
        ) if deep_rows else None,
        "deep_non_empty_edges_rate": round(
            (len(non_empty_deep) / len(deep_rows)) * 100, 1
        ) if deep_rows else None,
    }

    return {
        "repo": str(repo),
        "know_version": know_version,
        "queries": query_rows,
        "summary": summary,
    }


def render_markdown(data: Dict[str, Any]) -> str:
    lines: List[str] = []

    def _fmt_pct(value: Any) -> str:
        return "N/A" if value is None else f"{value}%"

    lines.append("# Dual-Repo Parallel Benchmark")
    lines.append("")
    lines.append(
        f"- Generated at: {data['generated_at']}"
    )
    lines.append(
        "- Strategies: `grep+read` baseline vs `know workflow` (single CLI call)"
    )
    lines.append("- Language globs: `py, ts, tsx, js, jsx, go, rs, swift`")
    lines.append("")

    for repo_row in data["repos"]:
        repo = repo_row["repo"]
        summary = repo_row.get("summary", {})
        lines.append(f"## {repo}")
        lines.append("")
        if repo_row.get("error"):
            lines.append(f"Error: `{repo_row['error']}`")
            lines.append("")
            continue

        lines.append(
            f"- know version: `{repo_row.get('know_version')}`"
        )
        gate = "PASS" if summary.get("quality_gate_passed") else "FAIL"
        lines.append(
            f"- Workflow: `{summary.get('workflow_mode')}` | "
            f"Latency budget: `{summary.get('latency_budget_ms')}ms` | "
            f"Read-only: `{summary.get('read_only')}` | "
            f"Quality gate: `{gate}`"
        )
        lines.append(
            f"- Raw token reduction: `{_fmt_pct(summary.get('token_reduction_pct'))}` | "
            f"Quality-adjusted: `{_fmt_pct(summary.get('quality_adjusted_token_reduction_pct'))}` | "
            f"Latency ratio (know/grep): `{summary.get('latency_ratio_know_over_grep')}x` | "
            f"Lookup-call reduction: `{summary.get('lookup_call_reduction_pct')}%`"
        )
        lines.append(
            f"- Warm p50/p95: `{summary.get('know_warm_p50_s')}s / {summary.get('know_warm_p95_s')}s` | "
            f"Cold start: `{summary.get('know_cold_start_s')}s` | "
            f"Fallback rate: `{summary.get('fallback_rate_pct')}%`"
        )
        lines.append(
            f"- Success rate: `{_fmt_pct(summary.get('success_rate_pct'))}` | "
            f"Errors: `{summary.get('error_count')}`"
        )
        lines.append(
            f"- Deep call-graph available: `{summary.get('deep_call_graph_available_rate')}%` "
            f"| non-empty edges: `{summary.get('deep_non_empty_edges_rate')}%`"
        )
        lines.append("")
        lines.append(
            "| Query | Grep Tokens | know Payload Tokens | Retrieval Tokens | Token Savings | "
            "Grep Time (s) | know Time (s) | know Payload (bytes) | Fallback |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in repo_row["queries"]:
            delta = row["delta"]
            lines.append(
                f"| {row['query']} | {row['grep']['tokens']:,} | {row['know']['tokens']:,} | "
                f"{row['know'].get('retrieval_tokens', 0):,} | {_fmt_pct(delta['token_savings_pct'])} | "
                f"{row['grep']['elapsed_s']} | {row['know']['elapsed_s']} | "
                f"{row['know'].get('payload_bytes', 0):,} | "
                f"{'yes' if row['know'].get('fallback_triggered') else 'no'} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dual-repo parallel benchmark")
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="Repo path to benchmark (repeatable). Defaults to know-cli + farfield.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to benchmark (repeatable). Defaults to 10 shared architecture queries.",
    )
    parser.add_argument(
        "--results-dir",
        dest="results_dir",
        default=str(RESULTS_DIR),
        help="Directory for benchmark artifacts (default: benchmark/results)",
    )
    parser.add_argument(
        "--workflow-mode",
        choices=sorted(MODE_DEFAULTS),
        default=DEFAULT_WORKFLOW_MODE,
        help="know workflow mode to benchmark (default: thorough)",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=int,
        default=DEFAULT_MAX_LATENCY_MS,
        help="know workflow latency budget in ms",
    )
    parser.add_argument(
        "--map-limit",
        type=int,
        default=DEFAULT_MAP_LIMIT,
        help="know workflow map result limit",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        help="know workflow context token budget",
    )
    parser.add_argument(
        "--deep-budget",
        type=int,
        default=DEFAULT_DEEP_BUDGET,
        help="know workflow deep token budget",
    )
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="Allow benchmark workflow calls to write sessions, stats, and memories.",
    )
    args = parser.parse_args()

    repos = [Path(p).expanduser().resolve() for p in (args.repos or DEFAULT_REPOS)]
    queries = args.queries or COMMON_QUERIES
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_json = results_dir / "dual_repo_parallel.json"
    results_md = results_dir / "DUAL_REPO_BENCHMARK.md"

    results_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    repo_rows = [
        benchmark_repo(
            repo,
            queries,
            mode=args.workflow_mode,
            max_latency_ms=args.max_latency_ms,
            map_limit=args.map_limit,
            context_budget=args.context_budget,
            deep_budget=args.deep_budget,
            read_only=not args.allow_side_effects,
        )
        for repo in repos
    ]
    elapsed = round(time.monotonic() - started, 3)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "workflow_mode": args.workflow_mode,
        "latency_budget_ms": args.max_latency_ms,
        "read_only": not args.allow_side_effects,
        "queries": queries,
        "repos": repo_rows,
    }

    results_json.write_text(json.dumps(out, indent=2))
    results_md.write_text(render_markdown(out))

    print(f"Saved JSON: {results_json}")
    print(f"Saved report: {results_md}")
    print(f"Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
