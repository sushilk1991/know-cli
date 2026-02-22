#!/usr/bin/env python3
"""Parallel dual-repo benchmark: grep+read baseline vs know 3-tier workflow.

Runs two "agents" in parallel per query:
  - grep_read: keyword grep + full file reads
  - know_3tier: know map -> know context -> know deep

Outputs:
  - benchmark/results/dual_repo_parallel.json
  - benchmark/results/DUAL_REPO_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from know.token_counter import count_tokens


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_JSON = RESULTS_DIR / "dual_repo_parallel.json"
RESULTS_MD = RESULTS_DIR / "DUAL_REPO_BENCHMARK.md"

DEFAULT_REPOS = [
    Path("/Users/sushil/Code/Github/know-cli"),
    Path("/Users/sushil/Code/Github/farfield"),
]

COMMON_QUERIES = [
    "configuration loading and environment variable handling",
    "database schema and persistence layer",
    "module dependency graph and call graph tracking",
    "error handling and retry logic",
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


def _extract_map_tokens(map_data: Dict[str, Any]) -> int:
    lines: List[str] = []
    for row in map_data.get("results", []):
        signature = row.get("signature") or row.get("chunk_name") or ""
        doc = row.get("docstring") or ""
        if signature:
            lines.append(signature)
        if doc:
            lines.append(doc)
    return count_tokens("\n".join(lines), provider="anthropic") if lines else 0


def run_know_agent(repo: Path, query: str) -> Dict[str, Any]:
    start = time.monotonic()

    map_t0 = time.monotonic()
    map_raw = run_cmd(["know", "--json", "map", query, "--limit", "20"], cwd=repo)
    map_elapsed = time.monotonic() - map_t0
    map_data = _parse_know_json(map_raw)
    map_tokens = _extract_map_tokens(map_data) if "error" not in map_data else 0

    ctx_t0 = time.monotonic()
    ctx_raw = run_cmd(
        ["know", "--json", "context", query, "--budget", "4000", "--session", "auto"],
        cwd=repo,
    )
    ctx_elapsed = time.monotonic() - ctx_t0
    ctx_data = _parse_know_json(ctx_raw)
    ctx_tokens = 0
    if "error" not in ctx_data:
        ctx_tokens = int(ctx_data.get("used_tokens", 0) or 0)

    top_name = None
    top_file = None
    code = ctx_data.get("code", []) if isinstance(ctx_data, dict) else []
    if code:
        preferred = None
        for chunk in code:
            ctype = (chunk.get("type") or "").lower()
            if ctype in {"function", "method"}:
                preferred = chunk
                break
        if preferred is None:
            preferred = code[0]
        top_name = preferred.get("name")
        top_file = preferred.get("file")
    if not top_name:
        rows = map_data.get("results", []) if isinstance(map_data, dict) else []
        if rows:
            top_name = rows[0].get("chunk_name") or rows[0].get("signature")
            top_file = rows[0].get("file_path")

    deep_elapsed = 0.0
    deep_tokens = 0
    deep_data: Dict[str, Any] = {}
    tool_calls = 2
    if top_name:
        tool_calls += 1
        deep_t0 = time.monotonic()
        deep_raw = run_cmd(
            ["know", "--json", "deep", str(top_name), "--budget", "3000"],
            cwd=repo,
        )
        deep_elapsed = time.monotonic() - deep_t0
        deep_data = _parse_know_json(deep_raw)
        if deep_data.get("error") == "ambiguous":
            candidates = deep_data.get("candidates", []) or []
            selected = None
            if top_file:
                for c in candidates:
                    if c.get("file_path") == top_file:
                        selected = c
                        break
            if selected is None and candidates:
                selected = candidates[0]
            if selected:
                disambiguated = f"{selected.get('file_path')}:{selected.get('chunk_name')}"
                tool_calls += 1
                deep_retry_t0 = time.monotonic()
                deep_retry_raw = run_cmd(
                    ["know", "--json", "deep", disambiguated, "--budget", "3000"],
                    cwd=repo,
                )
                deep_elapsed += time.monotonic() - deep_retry_t0
                deep_data = _parse_know_json(deep_retry_raw)
        if "error" not in deep_data:
            deep_tokens = int(
                deep_data.get("budget_used", deep_data.get("used_tokens", 0)) or 0
            )

    elapsed = time.monotonic() - start
    total_tokens = map_tokens + ctx_tokens + deep_tokens

    return {
        "strategy": "know_3tier",
        "query": query,
        "tool_calls": tool_calls,
        "elapsed_s": round(elapsed, 3),
        "tokens": total_tokens,
        "map_tokens": map_tokens,
        "context_tokens": ctx_tokens,
        "deep_tokens": deep_tokens,
        "steps": {
            "map_s": round(map_elapsed, 3),
            "context_s": round(ctx_elapsed, 3),
            "deep_s": round(deep_elapsed, 3),
        },
        "call_graph": {
            "available": deep_data.get("call_graph_available") if deep_data else None,
            "reason": deep_data.get("call_graph_reason") if deep_data else None,
            "callers": len(deep_data.get("callers", [])) if deep_data else 0,
            "callees": len(deep_data.get("callees", [])) if deep_data else 0,
            "target": (deep_data.get("target") or {}).get("name") if deep_data else None,
        },
        "errors": {
            "map": map_data.get("error") if isinstance(map_data, dict) else "unknown",
            "context": ctx_data.get("error") if isinstance(ctx_data, dict) else "unknown",
            "deep": deep_data.get("error") if isinstance(deep_data, dict) else None,
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


def benchmark_repo(repo: Path, queries: List[str]) -> Dict[str, Any]:
    if not repo.exists():
        return {
            "repo": str(repo),
            "error": "repo_not_found",
            "queries": [],
            "summary": {},
        }

    # Warm know index/cache so measured query latency is steady-state.
    _ = run_cmd(["know", "--json", "status"], cwd=repo, timeout=120)

    know_version = None
    status_result = run_cmd(["know", "--json", "status"], cwd=repo, timeout=120)
    status_data = _parse_know_json(status_result)
    if isinstance(status_data, dict):
        know_version = status_data.get("version")

    query_rows = []
    for query in queries:
        with ThreadPoolExecutor(max_workers=2) as pool:
            know_future = pool.submit(run_know_agent, repo, query)
            grep_future = pool.submit(run_grep_agent, repo, query)
            know_row = know_future.result()
            grep_row = grep_future.result()

        query_rows.append({
            "query": query,
            "know": know_row,
            "grep": grep_row,
            "delta": {
                "token_savings": grep_row["tokens"] - know_row["tokens"],
                "token_savings_pct": round(
                    ((grep_row["tokens"] - know_row["tokens"]) / grep_row["tokens"]) * 100,
                    1,
                ) if grep_row["tokens"] else None,
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
    deep_rows = [q["know"]["call_graph"] for q in query_rows]
    deep_available = [d for d in deep_rows if d.get("available") is True]
    non_empty_deep = [
        d for d in deep_rows if (d.get("callers", 0) + d.get("callees", 0)) > 0
    ]

    summary = {
        "know_tokens_total": know_tokens,
        "grep_tokens_total": grep_tokens,
        "know_time_total_s": round(know_time, 3),
        "grep_time_total_s": round(grep_time, 3),
        "know_tool_calls_total": know_calls,
        "grep_tool_calls_total": grep_calls,
        "token_reduction_pct": round(
            ((grep_tokens - know_tokens) / grep_tokens) * 100, 1
        ) if grep_tokens else None,
        "latency_ratio_know_over_grep": round(
            know_time / grep_time, 2
        ) if grep_time else None,
        "tool_call_reduction_pct": round(
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
    lines.append("# Dual-Repo Parallel Benchmark")
    lines.append("")
    lines.append(
        f"- Generated at: {data['generated_at']}"
    )
    lines.append(
        "- Strategies: `grep+read` baseline vs `know map -> context -> deep`"
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
        lines.append(
            f"- Token reduction: `{summary.get('token_reduction_pct')}%` | "
            f"Latency ratio (know/grep): `{summary.get('latency_ratio_know_over_grep')}x` | "
            f"Tool-call reduction: `{summary.get('tool_call_reduction_pct')}%`"
        )
        lines.append(
            f"- Deep call-graph available: `{summary.get('deep_call_graph_available_rate')}%` "
            f"| non-empty edges: `{summary.get('deep_non_empty_edges_rate')}%`"
        )
        lines.append("")
        lines.append("| Query | Grep Tokens | know Tokens | Token Savings | Grep Time (s) | know Time (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in repo_row["queries"]:
            delta = row["delta"]
            lines.append(
                f"| {row['query']} | {row['grep']['tokens']:,} | {row['know']['tokens']:,} | "
                f"{delta['token_savings_pct']}% | {row['grep']['elapsed_s']} | {row['know']['elapsed_s']} |"
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
        help="Query to benchmark (repeatable). Defaults to 4 shared architecture queries.",
    )
    args = parser.parse_args()

    repos = [Path(p).expanduser().resolve() for p in (args.repos or DEFAULT_REPOS)]
    queries = args.queries or COMMON_QUERIES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    repo_rows = [benchmark_repo(repo, queries) for repo in repos]
    elapsed = round(time.monotonic() - started, 3)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "queries": queries,
        "repos": repo_rows,
    }

    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    RESULTS_MD.write_text(render_markdown(out))

    print(f"Saved JSON: {RESULTS_JSON}")
    print(f"Saved report: {RESULTS_MD}")
    print(f"Total elapsed: {elapsed}s")


if __name__ == "__main__":
    main()
