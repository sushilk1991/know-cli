"""Agent commands: next-file, signatures, related, generate-context."""

import json
import os
import sys
from typing import Optional

import click
from click.core import ParameterSource

from know.cli import console, logger


def _get_daemon_client(config):
    """Try to get a DaemonClient, return None if daemon unavailable.

    In CI/CD environments (KNOW_NO_DAEMON=1), always returns None.
    """
    if os.environ.get("KNOW_NO_DAEMON"):
        return None
    try:
        from know.daemon import ensure_daemon
        return ensure_daemon(config.root, config)
    except Exception as e:
        logger.debug(f"Daemon unavailable, falling back to direct DB: {e}")
        return None


def _get_db_fallback(config):
    """Get a direct DaemonDB connection for when daemon is unavailable."""
    from know.daemon_db import DaemonDB
    return DaemonDB(config.root)


def _query_domain_intent(query: str) -> str:
    """Infer whether query is frontend/backend/mixed intent."""
    q = (query or "").lower()
    frontend_terms = {
        "react", "tsx", "jsx", "sidebar", "component", "page", "route",
        "frontend", "client", "ui", "css", "tailwind", "nextjs", "next",
    }
    backend_terms = {
        "api", "backend", "server", "database", "db", "sql", "endpoint",
        "middleware", "auth", "worker", "queue", "python", "fastapi",
    }
    f = sum(1 for t in frontend_terms if t in q)
    b = sum(1 for t in backend_terms if t in q)
    if f > b and f >= 1:
        return "frontend"
    if b > f and b >= 1:
        return "backend"
    return "mixed"


def _file_intent_boost(file_path: str, intent: str) -> float:
    """Score boost/penalty for query intent vs file path."""
    fp = file_path.lower()
    ext = os.path.splitext(fp)[1]
    frontend_ext = {".ts", ".tsx", ".js", ".jsx", ".css", ".scss"}
    backend_ext = {".py", ".go", ".rs", ".java", ".sql"}
    frontend_dirs = ("web/", "frontend/", "client/", "components/", "pages/", "app/")
    backend_dirs = ("backend/", "server/", "api/", "services/", "db/", "models/")

    if intent == "frontend":
        boost = 0.0
        if ext in frontend_ext:
            boost += 2.0
        if any(d in fp for d in frontend_dirs):
            boost += 2.0
        if ext in backend_ext:
            boost -= 1.0
        return boost
    if intent == "backend":
        boost = 0.0
        if ext in backend_ext:
            boost += 2.0
        if any(d in fp for d in backend_dirs):
            boost += 2.0
        if ext in frontend_ext:
            boost -= 1.0
        return boost
    return 0.0


def _pick_deep_target_from_context_payload(context_payload: dict, map_results: list) -> tuple[Optional[str], Optional[str]]:
    """Pick best deep target from context code first, map fallback second."""
    for preferred_type in ("function", "method", "class"):
        for chunk in context_payload.get("code", []) or []:
            if (chunk.get("type") or "").lower() == preferred_type and chunk.get("name"):
                return str(chunk["name"]), chunk.get("file")

    for preferred_type in ("function", "method", "class", "constant", "module"):
        for row in map_results:
            if (row.get("chunk_type") or "").lower() == preferred_type:
                name = row.get("chunk_name") or row.get("signature")
                if name:
                    return str(name), row.get("file_path")

    return None, None


def _stdout_is_tty() -> bool:
    """Return whether stdout is a TTY."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _build_workflow_compact_payload(result: dict) -> dict:
    """Build a compact, human-readable JSON payload for workflow output."""
    map_payload = result.get("map") or {}
    context_payload = result.get("context") or {}
    deep_payload = result.get("deep") or {}

    map_rows = map_payload.get("results") or []
    candidates = []
    for row in map_rows[:5]:
        candidates.append(
            {
                "file": row.get("file_path"),
                "symbol": row.get("chunk_name") or row.get("signature"),
                "type": row.get("chunk_type"),
                "line": row.get("start_line"),
                "score": row.get("score"),
            }
        )

    snippets = []
    for chunk in (context_payload.get("code") or [])[:5]:
        body = str(chunk.get("body") or "")
        body_lines = [ln.rstrip() for ln in body.splitlines()[:6]]
        snippets.append(
            {
                "file": chunk.get("file"),
                "symbol": chunk.get("name"),
                "type": chunk.get("type"),
                "lines": chunk.get("lines"),
                "snippet": "\n".join(body_lines),
            }
        )

    deep_target = deep_payload.get("target") or {}
    selected_symbol = result.get("selected_deep_target") or deep_target.get("name")
    selected_file = deep_target.get("file")
    if not selected_file and candidates:
        selected_file = candidates[0].get("file")

    reason = "top-ranked context target"
    if deep_payload.get("error"):
        reason = f"deep_error:{deep_payload.get('error')}"
    next_step = "proceed_with_targeted_edit"
    if deep_payload.get("error") == "no_target":
        next_step = "run: know map \"<query>\" --json --limit 30"
    elif deep_payload.get("error") == "skipped_by_mode":
        next_step = "run: know --json workflow \"<query>\" --mode implement --session auto"
    elif deep_payload.get("error") == "skipped_latency_budget":
        next_step = "rerun with: --max-latency-ms 12000 or --mode thorough"
    elif deep_payload.get("error") == "ambiguous":
        next_step = "run: know deep \"file.py:symbol\" --json --budget 3000"
    elif deep_payload.get("error"):
        next_step = "run: know related <file_path> --json"

    return {
        "query": result.get("query"),
        "session_id": result.get("session_id"),
        "indexing_status": context_payload.get("indexing_status", "unknown"),
        "targets": {
            "selected_symbol": selected_symbol,
            "selected_file": selected_file,
            "reason": reason,
            "next_step": next_step,
            "candidates": candidates,
        },
        "context": {
            "snippets": snippets,
            "warnings": context_payload.get("warnings") or [],
        },
        "deep": {
            "target": {
                "name": deep_target.get("name"),
                "file": deep_target.get("file"),
                "line_start": deep_target.get("line_start"),
                "line_end": deep_target.get("line_end"),
            },
            "callers": len(deep_payload.get("callers") or []),
            "callees": len(deep_payload.get("callees") or []),
            "call_graph_available": deep_payload.get("call_graph_available"),
            "call_graph_reason": deep_payload.get("call_graph_reason"),
            "error": deep_payload.get("error"),
        },
        "metrics": {
            "profile": "compact",
            "mode": result.get("workflow_mode", "implement"),
            "latency_budget_ms": result.get("latency_budget_ms"),
            "latency_ms": result.get("latency_ms", {}),
            "degraded_by_latency": result.get("degraded_by_latency", False),
            "map_results": map_payload.get("count", 0),
            "context_tokens": context_payload.get("used_tokens", 0),
            "deep_tokens": deep_payload.get(
                "budget_used", deep_payload.get("used_tokens", 0),
            ),
            "total_tokens": result.get("total_tokens"),
        },
    }


@click.command("next-file")
@click.argument("query")
@click.option("--exclude", "-x", multiple=True, help="Files to exclude")
@click.option("--budget", "-b", type=int, default=10000, help="Token budget")
@click.pass_context
def next_file(ctx: click.Context, query: str, exclude: tuple, budget: int) -> None:
    """Return the single most relevant file path for a query."""
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run 'know init' or use in a git repo.[/red]")
        return

    client = _get_daemon_client(config)
    if client:
        try:
            result = client.call_sync("search", {"query": query, "limit": 50})
            results = result.get("results", [])
        except Exception as e:
            logger.debug(f"Daemon search failed, falling back: {e}")
            client = None

    if not client:
        db = _get_db_fallback(config)
        results = db.search_chunks(query, limit=50)
        db.close()

    # Rerank file candidates by inferred domain intent + best chunk score
    intent = _query_domain_intent(query)
    file_best = {}
    for chunk in results:
        fp = chunk.get("file_path", "")
        if not fp:
            continue
        raw = float(chunk.get("score", chunk.get("rank", 0.0)) or 0.0)
        boosted = raw + _file_intent_boost(fp, intent)
        prev = file_best.get(fp)
        if prev is None or boosted > prev:
            file_best[fp] = boosted

    ranked_files = [fp for fp, _ in sorted(file_best.items(), key=lambda kv: kv[1], reverse=True)]

    is_json = bool(ctx.obj.get("json"))
    is_quiet = bool(ctx.obj.get("quiet"))

    # Filter excluded files and find best match
    seen_files = set(exclude)
    for fp in ranked_files:
        if fp not in seen_files:
            if is_json:
                click.echo(_json.dumps({
                    "file": fp,
                    "relevance": file_best.get(fp, 0.0),
                    "intent": intent,
                }))
            elif is_quiet:
                click.echo(fp)
            else:
                console.print(fp)
            return

    console.print("[dim]No more relevant files found.[/dim]")


@click.command("signatures")
@click.argument("file_path", required=False, default=None)
@click.pass_context
def signatures(ctx: click.Context, file_path: Optional[str]) -> None:
    """Get function/class signatures for a file or entire project."""
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized.[/red]")
        return

    client = _get_daemon_client(config)
    if client:
        try:
            result = client.call_sync("signatures", {"file": file_path})
            sigs = result.get("signatures", [])
        except Exception as e:
            logger.debug(f"Daemon signatures failed, falling back: {e}")
            client = None

    if not client:
        db = _get_db_fallback(config)
        sigs = db.get_signatures(file_path)
        db.close()

    is_json = bool(ctx.obj.get("json"))
    if is_json:
        click.echo(_json.dumps(sigs))
    else:
        for s in sigs:
            console.print(f"[cyan]{s['file_path']}[/cyan]:{s['start_line']} "
                          f"[bold]{s['chunk_type']}[/bold] {s['signature']}")


@click.command("related")
@click.argument("file_path")
@click.pass_context
def related(ctx: click.Context, file_path: str) -> None:
    """Show import dependencies and dependents for a file."""
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized.[/red]")
        return

    # Phase 2 freshness: re-index target file on demand if stale.
    db_refresh = None
    try:
        from know.daemon import refresh_file_if_stale
        db_refresh = _get_db_fallback(config)
        refresh_file_if_stale(
            config.root, config, db_refresh, file_path, remove_missing=True,
        )
    except Exception as e:
        logger.debug(f"Related stale-file refresh skipped: {e}")
    finally:
        try:
            db_refresh.close()
        except Exception:
            pass

    client = _get_daemon_client(config)
    imports = []
    imported_by = []
    if client:
        try:
            # Daemon's related is Python-centric; use it as best-effort signal.
            module = file_path.replace("/", ".").replace("\\", ".")
            module = module.removesuffix(".py").removesuffix(".ts").removesuffix(".tsx")
            module = module.removesuffix(".js").removesuffix(".jsx")
            result = client.call_sync("related", {"module": module})
            imports = result.get("imports", [])
            imported_by = result.get("imported_by", [])
        except Exception as e:
            logger.debug(f"Daemon related failed, falling back: {e}")

    # Always run language-agnostic resolver and merge, because daemon graph is
    # currently Python-only and often misses TS/TSX relationships.
    try:
        from know.import_graph import ImportGraph
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        scanner.get_structure()  # Populates scanner.modules
        modules_with_imports = [
            {
                "path": str(m.path),
                "name": m.name,
                "imports": list(getattr(m, "imports", []) or []),
            }
            for m in scanner.modules
        ]
        lang_imports, lang_imported_by = ImportGraph.related_files_from_modules(
            file_path, modules_with_imports,
        )
        if lang_imports:
            imports = sorted(set(imports).union(lang_imports))
        if lang_imported_by:
            imported_by = sorted(set(imported_by).union(lang_imported_by))
    except Exception as e:
        logger.debug(f"Language-agnostic related lookup failed: {e}")

    is_json = bool(ctx.obj.get("json"))
    if is_json:
        click.echo(_json.dumps({"imports": imports, "imported_by": imported_by}))
    else:
        if imports:
            console.print("[bold]Imports (dependencies):[/bold]")
            for m in sorted(imports):
                console.print(f"  → {m}")
        else:
            console.print("[dim]No imports found.[/dim]")

        if imported_by:
            console.print("[bold]Imported by (dependents):[/bold]")
            for m in sorted(imported_by):
                console.print(f"  ← {m}")
        else:
            console.print("[dim]No dependents found.[/dim]")


@click.command("callers")
@click.argument("function_name")
@click.pass_context
def callers(ctx: click.Context, function_name: str) -> None:
    """Find all chunks that call a given function."""
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized.[/red]")
        return

    client = _get_daemon_client(config)
    results = None
    if client:
        try:
            result = client.call_sync("callers", {"function_name": function_name})
            results = result.get("callers", [])
        except Exception as e:
            logger.debug(f"Daemon callers failed, falling back: {e}")
            client = None

    if results is None:
        db = _get_db_fallback(config)
        results = db.get_callers(function_name)
        db.close()

    is_json = bool(ctx.obj.get("json"))
    if is_json:
        click.echo(_json.dumps({"callers": results, "count": len(results)}))
    else:
        if results:
            console.print(f"[bold]Callers of [cyan]{function_name}[/cyan]:[/bold]")
            for r in results:
                console.print(f"  {r['file_path']} → {r['containing_chunk']}:{r['line_number']}")
        else:
            console.print(f"[dim]No callers found for '{function_name}'.[/dim]")


@click.command("callees")
@click.argument("chunk_name")
@click.pass_context
def callees(ctx: click.Context, chunk_name: str) -> None:
    """Find all functions called by a given chunk."""
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized.[/red]")
        return

    client = _get_daemon_client(config)
    results = None
    if client:
        try:
            result = client.call_sync("callees", {"chunk_name": chunk_name})
            results = result.get("callees", [])
        except Exception as e:
            logger.debug(f"Daemon callees failed, falling back: {e}")
            client = None

    if results is None:
        db = _get_db_fallback(config)
        results = db.get_callees(chunk_name)
        db.close()

    is_json = bool(ctx.obj.get("json"))
    if is_json:
        click.echo(_json.dumps({"callees": results, "count": len(results)}))
    else:
        if results:
            console.print(f"[bold]Functions called by [cyan]{chunk_name}[/cyan]:[/bold]")
            for r in results:
                console.print(f"  → {r['ref_name']} ({r['ref_type']}) at {r['file_path']}:{r['line_number']}")
        else:
            console.print(f"[dim]No callees found for '{chunk_name}'.[/dim]")


@click.command("generate-context")
@click.option("--budget", "-b", type=int, default=8000, help="Token budget for CONTEXT.md")
@click.pass_context
def generate_context(ctx: click.Context, budget: int) -> None:
    """Generate .know/CONTEXT.md for AI agents to read on session start."""
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized.[/red]")
        return

    from know.token_counter import count_tokens

    client = _get_daemon_client(config)
    if client:
        try:
            status_result = client.call_sync("status")
            stats = status_result.get("stats", {})
            sig_result = client.call_sync("signatures", {})
            sigs = sig_result.get("signatures", [])
            recall_result = client.call_sync("recall", {"query": "project architecture patterns", "limit": 20})
            memories = recall_result.get("memories", [])
        except Exception as e:
            logger.debug(f"Daemon generate-context failed, falling back: {e}")
            client = None

    if not client:
        db = _get_db_fallback(config)
        stats = db.get_stats()
        sigs = db.get_signatures()
        memories = db.recall_memories("project architecture patterns", limit=20)
        db.close()

    lines = [
        f"# {config.root.name}",
        "",
        f"**Files:** {stats.get('files', 0)} | **Functions/Classes:** {stats.get('chunks', 0)} | "
        f"**Total tokens:** {stats.get('total_tokens', 0):,}",
        "",
        "## Key Signatures",
        "",
    ]

    # Add top signatures
    for s in sigs[:50]:
        sig_line = f"- `{s['file_path']}:{s['start_line']}` — {s['signature']}"
        lines.append(sig_line)
        if count_tokens("\n".join(lines)) > budget * 0.8:
            break

    # Add memories
    if memories:
        lines.append("")
        lines.append("## Remembered Context")
        lines.append("")
        for m in memories:
            text = (m.get("text") or m.get("content") or "").strip()
            if not text:
                continue
            lines.append(f"- {text[:200]}")
            if count_tokens("\n".join(lines)) > budget * 0.95:
                break

    content = "\n".join(lines) + "\n"
    output_path = config.root / ".know" / "CONTEXT.md"
    output_path.write_text(content)

    tokens = count_tokens(content)
    console.print(f"[green]Generated[/green] {output_path} ({tokens:,} tokens)")


@click.command("map")
@click.argument("query")
@click.option("--limit", "-k", type=int, default=20, help="Max results (default 20)")
@click.option("--type", "chunk_type", type=click.Choice(["function", "class", "module", "method"]),
              default=None, help="Filter by chunk type")
@click.option("--session", "session_id", default=None, help="Session ID (accepted for workflow compatibility)")
@click.pass_context
def map_cmd(
    ctx: click.Context,
    query: str,
    limit: int,
    chunk_type: Optional[str],
    session_id: Optional[str],
) -> None:
    """Lightweight signature search — orient before reading.

    Returns function/class signatures matching a query with no bodies.
    Use this to discover what exists before using `know context` or `know deep`.

    Example: know map "billing subscription"
    """
    import json as _json
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run: know init[/red]")
        return

    client = _get_daemon_client(config)
    results = None
    if client:
        try:
            result = client.call_sync("map", {
                "query": query,
                "limit": limit,
                "chunk_type": chunk_type,
                "session_id": session_id,
            })
            results = result.get("results", [])
        except Exception as e:
            logger.debug(f"Daemon map failed, falling back: {e}")
            client = None

    if results is None:
        db = _get_db_fallback(config)
        results = db.search_signatures(query, limit, chunk_type)
        db.close()

    is_json = ctx.obj.get("json")
    if is_json:
        click.echo(_json.dumps({
            "query": query,
            "session_id": session_id,
            "results": results,
            "count": len(results),
            "truncated": len(results) >= limit,
        }))
    else:
        if results:
            console.print(f"[bold]Map results for [cyan]{query}[/cyan]:[/bold] ({len(results)} matches)\n")
            for r in results:
                sig = r.get("signature", r["chunk_name"])
                doc = r.get("docstring", "")
                score = r.get("score", 0)
                line = f"  [green]{r['file_path']}[/green]:{r['start_line']}  {sig}"
                if doc:
                    line += f"  [dim]— {doc}[/dim]"
                console.print(line)
        else:
            console.print(f"[dim]No matches for '{query}'.[/dim]")


@click.command("workflow")
@click.argument("query")
@click.option("--map-limit", type=int, default=20, help="Max map results (default 20)")
@click.option("--context-budget", type=int, default=4000, help="Context budget (default 4000)")
@click.option("--deep-budget", type=int, default=3000, help="Deep budget (default 3000)")
@click.option("--session", "session_id", default="auto", help="Session ID for dedup ('auto' generates)")
@click.option("--include-tests", is_flag=True, help="Include test files")
@click.option(
    "--mode",
    type=click.Choice(["explore", "implement", "thorough"]),
    default="implement",
    show_default=True,
    help="Workflow mode: explore(fast), implement(balanced), thorough(deep).",
)
@click.option(
    "--max-latency-ms",
    type=int,
    default=None,
    help="End-to-end latency budget; workflow degrades gracefully when exceeded.",
)
@click.option("--json-compact", is_flag=True, help="Compact JSON profile (human-readable)")
@click.option("--json-full", is_flag=True, help="Full JSON profile (backward-compatible)")
@click.pass_context
def workflow(
    ctx: click.Context,
    query: str,
    map_limit: int,
    context_budget: int,
    deep_budget: int,
    session_id: Optional[str],
    include_tests: bool,
    mode: str,
    max_latency_ms: Optional[int],
    json_compact: bool,
    json_full: bool,
) -> None:
    """Single-call daemon workflow: map -> context -> deep."""
    import uuid
    import time

    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run: know init[/red]")
        return

    if json_compact and json_full:
        raise click.UsageError("--json-compact and --json-full are mutually exclusive")

    is_json = bool(ctx.obj.get("json"))
    if (json_compact or json_full) and not is_json:
        raise click.UsageError("JSON profile flags require global --json")

    mode_defaults = {
        "explore": {"map_limit": 30, "context_budget": 3500, "deep_budget": 0, "max_latency_ms": 2500},
        "implement": {"map_limit": 20, "context_budget": 4000, "deep_budget": 3000, "max_latency_ms": 6000},
        "thorough": {"map_limit": 30, "context_budget": 6000, "deep_budget": 4500, "max_latency_ms": 15000},
    }[mode]
    # Respect explicit CLI values; only override defaults by mode when user did not set them.
    if ctx.get_parameter_source("map_limit") == ParameterSource.DEFAULT:
        map_limit = mode_defaults["map_limit"]
    if ctx.get_parameter_source("context_budget") == ParameterSource.DEFAULT:
        context_budget = mode_defaults["context_budget"]
    if ctx.get_parameter_source("deep_budget") == ParameterSource.DEFAULT:
        deep_budget = mode_defaults["deep_budget"]
    if ctx.get_parameter_source("max_latency_ms") == ParameterSource.DEFAULT:
        max_latency_ms = mode_defaults["max_latency_ms"]
    if max_latency_ms is not None and int(max_latency_ms) <= 0:
        max_latency_ms = None

    resolved_session_id = session_id
    if session_id in ("auto", "new"):
        resolved_session_id = uuid.uuid4().hex[:8]
    if resolved_session_id:
        try:
            from know.runtime_context import set_active_session_id
            set_active_session_id(config, resolved_session_id)
        except Exception as e:
            logger.debug(f"Failed to persist active session id: {e}")

    result = None
    workflow_started = time.monotonic()
    workflow_from_daemon = False
    client = _get_daemon_client(config)
    if client:
        try:
            result = client.call_sync("workflow", {
                "query": query,
                "map_limit": map_limit,
                "context_budget": context_budget,
                "deep_budget": deep_budget,
                "session_id": resolved_session_id,
                "include_tests": include_tests,
                "mode": mode,
                "max_latency_ms": max_latency_ms,
            })
            # Guard mixed-version upgrades: stale daemon may not support
            # workflow mode/SLA fields. Fall back to local workflow when missing.
            if isinstance(result, dict) and "workflow_mode" in result and "latency_ms" in result:
                workflow_from_daemon = True
            else:
                logger.debug("Daemon workflow response missing mode/SLA fields; using local workflow fallback")
                result = None
                workflow_from_daemon = False
        except Exception as e:
            logger.debug(f"Daemon workflow failed, falling back: {e}")
            client = None

    if result is None:
        from know.context_engine import ContextEngine

        engine = ContextEngine(config)
        db = _get_db_fallback(config)
        try:
            deadline = (
                workflow_started + (int(max_latency_ms) / 1000.0)
                if max_latency_ms is not None and int(max_latency_ms) > 0
                else None
            )

            def _remaining_ms() -> int:
                if deadline is None:
                    return 10**9
                return max(0, int((deadline - time.monotonic()) * 1000))

            map_t0 = time.monotonic()
            map_results = db.search_signatures(query, map_limit)
            map_elapsed_ms = int((time.monotonic() - map_t0) * 1000)

            context_t0 = time.monotonic()
            if _remaining_ms() <= 120:
                context_payload = {
                    "query": query,
                    "budget": context_budget,
                    "used_tokens": 0,
                    "budget_utilization": "0 / 0 (0%)",
                    "indexing_status": "complete",
                    "confidence": 0,
                    "warnings": [
                        f"workflow latency budget exhausted before context (max_latency_ms={max_latency_ms})",
                    ],
                    "code": [],
                    "dependencies": [],
                    "tests": [],
                    "summaries": [],
                    "overview": "",
                    "source_files": [],
                    "error": "skipped_latency_budget",
                }
            else:
                semantic_budget_ms = (
                    max(200, int(_remaining_ms() * 0.75))
                    if deadline is not None
                    else None
                )
                context_result = engine.build_context(
                    query,
                    budget=context_budget,
                    include_tests=include_tests,
                    include_imports=True,
                    session_id=resolved_session_id,
                    retrieval_profile=(
                        "fast"
                        if mode == "explore"
                        else ("thorough" if mode == "thorough" else "balanced")
                    ),
                    semantic_max_ms=semantic_budget_ms,
                    db=db,
                )

                # Memory injection parity with context command.
                try:
                    from know.knowledge_base import KnowledgeBase
                    kb = KnowledgeBase(config)
                    memory_ctx = kb.get_relevant_context(
                        query, max_tokens=min(500, context_budget // 10),
                    )
                    if memory_ctx:
                        context_result["memories_context"] = memory_ctx
                except Exception as e:
                    logger.debug(f"Workflow memory injection failed: {e}")

                context_payload = json.loads(engine.format_agent_json(context_result))
            context_elapsed_ms = int((time.monotonic() - context_t0) * 1000)

            target_name, target_file = _pick_deep_target_from_context_payload(
                context_payload, map_results,
            )

            deep_t0 = time.monotonic()
            if mode == "explore" or deep_budget <= 0:
                deep_result = {"error": "skipped_by_mode", "reason": f"mode_{mode}"}
            elif _remaining_ms() <= 150:
                deep_result = {"error": "skipped_latency_budget", "reason": f"max_latency_ms={max_latency_ms}"}
            elif target_name:
                deep_query = f"{target_file}:{target_name}" if target_file else target_name
                deep_result = engine.build_deep_context(
                    deep_query,
                    budget=deep_budget,
                    include_tests=include_tests,
                    session_id=resolved_session_id,
                    db=db,
                )
                if deep_result.get("error") == "ambiguous":
                    deep_result = engine.build_deep_context(
                        target_name,
                        budget=deep_budget,
                        include_tests=include_tests,
                        session_id=resolved_session_id,
                        db=db,
                    )
            else:
                deep_result = {"error": "no_target", "reason": "no_context_or_map_target"}
            deep_elapsed_ms = int((time.monotonic() - deep_t0) * 1000)
            total_elapsed_ms = int((time.monotonic() - workflow_started) * 1000)
            from know.token_counter import count_tokens

            map_text = "\n".join(
                filter(
                    None,
                    [
                        f"{r.get('signature', '')}\n{r.get('docstring', '')}".strip()
                        for r in map_results
                    ],
                )
            )
            map_tokens = count_tokens(map_text) if map_text else 0
            total_tokens = (
                map_tokens
                + int(context_payload.get("used_tokens", 0) or 0)
                + int(deep_result.get("budget_used", deep_result.get("used_tokens", 0)) or 0)
            )

            result = {
                "query": query,
                "session_id": resolved_session_id,
                "workflow_mode": mode,
                "latency_budget_ms": max_latency_ms,
                "budgets": {
                    "map_limit": map_limit,
                    "context_budget": context_budget,
                    "deep_budget": deep_budget,
                },
                "selected_deep_target": target_name,
                "map": {
                    "results": map_results,
                    "count": len(map_results),
                    "truncated": len(map_results) >= map_limit,
                    "tokens": map_tokens,
                },
                "context": context_payload,
                "deep": deep_result,
                "total_tokens": total_tokens,
                "latency_ms": {
                    "map": map_elapsed_ms,
                    "context": context_elapsed_ms,
                    "deep": deep_elapsed_ms,
                    "total": total_elapsed_ms,
                },
                "degraded_by_latency": bool(
                    context_payload.get("error") == "skipped_latency_budget"
                    or deep_result.get("error") == "skipped_latency_budget"
                ),
            }
        finally:
            db.close()

    # Auto-capture key workflow decision for cross-session recall.
    # Daemon workflow already captures this; avoid duplicate work/memory rows.
    if not workflow_from_daemon:
        try:
            from know.memory_capture import capture_workflow_decision

            capture_workflow_decision(
                config,
                query,
                result,
                session_id=resolved_session_id,
                source="auto-workflow",
                agent="know-cli",
            )
        except Exception as e:
            logger.debug(f"Workflow decision capture failed: {e}")

    if is_json:
        if json_full:
            profile = "full"
        elif json_compact:
            profile = "compact"
        else:
            profile = "compact" if _stdout_is_tty() else "full"

        if profile == "compact":
            click.echo(json.dumps(_build_workflow_compact_payload(result)))
        else:
            click.echo(json.dumps(result))
        return

    map_count = (result.get("map") or {}).get("count", 0)
    ctx_payload = result.get("context") or {}
    deep_result = result.get("deep") or {}
    latency = result.get("latency_ms") or {}
    console.print(f"[bold]Workflow:[/bold] [cyan]{query}[/cyan]")
    console.print(
        f"[dim]mode={result.get('workflow_mode', 'implement')} | "
        f"map={map_count} results | context={ctx_payload.get('used_tokens', 0)} tokens | "
        f"deep={deep_result.get('budget_used', deep_result.get('used_tokens', 0))} tokens | "
        f"latency={latency.get('total', 0)}ms[/dim]"
    )

    map_rows = (result.get("map") or {}).get("results", [])[:8]
    if map_rows:
        console.print("\n[bold]Map (top):[/bold]")
        for row in map_rows:
            sig = row.get("signature") or row.get("chunk_name", "")
            console.print(f"  [green]{row.get('file_path', '')}[/green]:{row.get('start_line', 0)}  {sig}")

    target = deep_result.get("target", {})
    if target:
        console.print(
            f"\n[bold]Deep target:[/bold] [cyan]{target.get('name', '')}[/cyan] "
            f"[dim]({target.get('file', '')}:{target.get('line_start', 0)})[/dim]"
        )
        callers = len(deep_result.get("callers", []) or [])
        callees = len(deep_result.get("callees", []) or [])
        console.print(f"[dim]callers={callers}, callees={callees}[/dim]")
    elif deep_result.get("error"):
        console.print(f"\n[yellow]Deep skipped:[/yellow] {deep_result.get('error')}")


@click.command("warm")
@click.pass_context
def warm(ctx: click.Context) -> None:
    """Start daemon and report warmup/index readiness."""
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run: know init[/red]")
        return

    client = _get_daemon_client(config)
    daemon_running = bool(client)
    stats = {}
    if client:
        try:
            status_result = client.call_sync("status")
            stats = status_result.get("stats", {}) or {}
        except Exception as e:
            logger.debug(f"Daemon status check failed: {e}")

    if not stats:
        try:
            db = _get_db_fallback(config)
            stats = db.get_stats()
        finally:
            try:
                db.close()
            except Exception:
                pass

    files = int(stats.get("files", 0) or 0)
    indexing_status = "complete" if files > 0 else "warming"
    payload = {
        "daemon_running": daemon_running,
        "indexing_status": indexing_status,
        "stats": stats,
        "project_root": str(config.root),
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(payload))
        return

    if daemon_running:
        console.print("[green]Daemon is running[/green]")
    else:
        console.print("[yellow]Daemon unavailable; using direct DB status[/yellow]")
    console.print(f"Indexing status: [cyan]{indexing_status}[/cyan] (files={files})")


@click.command("deep")
@click.argument("name")
@click.option("--budget", "-b", type=int, default=3000, help="Token budget (default 3000)")
@click.option("--session", "session_id", default=None, help="Session ID for dedup")
@click.option("--include-tests", is_flag=True, help="Include test files in results")
@click.pass_context
def deep(ctx: click.Context, name: str, budget: int, session_id: Optional[str],
         include_tests: bool) -> None:
    """Deep context: function body + callers + callees.

    Resolve a function by name and return its body along with
    the functions it calls and the functions that call it.

    Name formats: function_name, Class.method, file.py:function_name

    Example: know deep "check_cloud_access" --budget 3000
    """
    import json as _json
    import sys
    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run: know init[/red]")
        return

    client = _get_daemon_client(config)
    result = None
    if client:
        try:
            result = client.call_sync("deep", {
                "name": name, "budget": budget,
                "include_tests": include_tests,
                "session_id": session_id,
            })
        except Exception as e:
            logger.debug(f"Daemon deep failed, falling back: {e}")
            client = None

    if result is None:
        from know.context_engine import ContextEngine
        engine = ContextEngine(config)
        result = engine.build_deep_context(
            name, budget=budget, include_tests=include_tests,
            session_id=session_id,
        )

    is_json = ctx.obj.get("json")

    if "error" in result:
        if is_json:
            click.echo(_json.dumps(result))
        else:
            err = result["error"]
            if err == "ambiguous":
                console.print(f"[yellow]Ambiguous name '{name}'. Candidates:[/yellow]")
                for c in result.get("candidates", []):
                    console.print(f"  {c['file_path']}:{c['start_line']} — {c['chunk_name']} ({c['chunk_type']})")
            elif err == "not_found":
                console.print(f"[red]Function '{name}' not found.[/red]")
                nearest = result.get("nearest", [])
                if nearest:
                    console.print("[dim]Did you mean:[/dim]")
                    for n in nearest:
                        console.print(f"  {n}")
            else:
                console.print(f"[red]Error: {err}[/red]")
        sys.exit(2 if result["error"] == "not_found" else 1)

    if is_json:
        click.echo(_json.dumps(result))
    else:
        target = result.get("target", {})
        console.print(f"[bold cyan]{target.get('name', name)}[/bold cyan] "
                       f"[dim]({target.get('file', '')}:{target.get('line_start', '')})[/dim]")
        console.print(target.get("body", ""))
        callees = result.get("callees", [])
        if callees:
            console.print(f"\n[bold]Calls ({len(callees)}):[/bold]")
            for c in callees:
                console.print(f"  [green]{c['name']}[/green] — {c['file']}:{c.get('call_site_line', '')}")
        callers = result.get("callers", [])
        if callers:
            console.print(f"\n[bold]Called by ({len(callers)}):[/bold]")
            for c in callers:
                console.print(f"  [green]{c['name']}[/green] — {c['file']}:{c.get('call_site_line', '')}")
        if not result.get("call_graph_available", True):
            reason = result.get("call_graph_reason", "unavailable")
            console.print(f"\n[dim]Call graph unavailable: {reason}[/dim]")
        overflow = result.get("overflow_signatures", [])
        if overflow:
            console.print(f"\n[dim]+{len(overflow)} more (budget exhausted)[/dim]")
