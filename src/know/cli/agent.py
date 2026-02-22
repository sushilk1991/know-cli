"""Agent commands: next-file, signatures, related, generate-context."""

import json
import os
from typing import Optional

import click

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

    # Filter excluded files and find best match
    seen_files = set(exclude)
    for fp in ranked_files:
        if fp not in seen_files:
            output = ctx.obj.get("output_format", "rich")
            if output == "json":
                console.print(_json.dumps({
                    "file": fp,
                    "relevance": file_best.get(fp, 0.0),
                    "intent": intent,
                }))
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

    output = ctx.obj.get("output_format", "rich")
    if output == "json":
        console.print(_json.dumps(sigs))
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

    output = ctx.obj.get("output_format", "rich")
    if output == "json":
        console.print(_json.dumps({"imports": imports, "imported_by": imported_by}))
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

    output = ctx.obj.get("output_format", "rich")
    if output == "json":
        console.print(_json.dumps({"callers": results, "count": len(results)}))
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

    output = ctx.obj.get("output_format", "rich")
    if output == "json":
        console.print(_json.dumps({"callees": results, "count": len(results)}))
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
            lines.append(f"- {m['content'][:200]}")
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
@click.pass_context
def map_cmd(ctx: click.Context, query: str, limit: int, chunk_type: Optional[str]) -> None:
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
                "query": query, "limit": limit, "chunk_type": chunk_type,
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
@click.pass_context
def workflow(
    ctx: click.Context,
    query: str,
    map_limit: int,
    context_budget: int,
    deep_budget: int,
    session_id: Optional[str],
    include_tests: bool,
) -> None:
    """Single-call daemon workflow: map -> context -> deep."""
    import uuid

    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Not initialized. Run: know init[/red]")
        return

    resolved_session_id = session_id
    if session_id in ("auto", "new"):
        resolved_session_id = uuid.uuid4().hex[:8]

    result = None
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
            })
        except Exception as e:
            logger.debug(f"Daemon workflow failed, falling back: {e}")
            client = None

    if result is None:
        from know.context_engine import ContextEngine

        engine = ContextEngine(config)
        db = _get_db_fallback(config)
        try:
            map_results = db.search_signatures(query, map_limit)
            context_result = engine.build_context(
                query,
                budget=context_budget,
                include_tests=include_tests,
                include_imports=True,
                session_id=resolved_session_id,
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
            target_name, target_file = _pick_deep_target_from_context_payload(
                context_payload, map_results,
            )

            if target_name:
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

            result = {
                "query": query,
                "session_id": resolved_session_id,
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
                },
                "context": context_payload,
                "deep": deep_result,
            }
        finally:
            db.close()

    is_json = ctx.obj.get("json")
    if is_json:
        click.echo(json.dumps(result))
        return

    map_count = (result.get("map") or {}).get("count", 0)
    ctx_payload = result.get("context") or {}
    deep_result = result.get("deep") or {}
    console.print(f"[bold]Workflow:[/bold] [cyan]{query}[/cyan]")
    console.print(
        f"[dim]map={map_count} results | context={ctx_payload.get('used_tokens', 0)} tokens | "
        f"deep={deep_result.get('budget_used', deep_result.get('used_tokens', 0))} tokens[/dim]"
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
