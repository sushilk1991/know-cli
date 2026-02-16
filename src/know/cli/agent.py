"""Agent commands: next-file, signatures, related, generate-context."""

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

    # Filter excluded files and find best match
    seen_files = set(exclude)
    for chunk in results:
        fp = chunk["file_path"]
        if fp not in seen_files:
            output = ctx.obj.get("output_format", "rich")
            if output == "json":
                console.print(_json.dumps({"file": fp, "relevance": chunk.get("rank", 0)}))
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

    # Convert file path to module name
    module = file_path.replace("/", ".").replace(".py", "").replace(".ts", "")

    client = _get_daemon_client(config)
    if client:
        try:
            result = client.call_sync("related", {"module": module})
            imports = result.get("imports", [])
            imported_by = result.get("imported_by", [])
        except Exception as e:
            logger.debug(f"Daemon related failed, falling back: {e}")
            client = None

    if not client:
        # Build import graph if not already populated
        from know.import_graph import ImportGraph
        ig = ImportGraph(config)
        from know.scanner import CodebaseScanner
        scanner = CodebaseScanner(config)
        structure = scanner.get_structure()
        ig.build(structure.get("modules", []))
        imports = ig.imports_of(module)
        imported_by = ig.imported_by(module)

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
        overflow = result.get("overflow_signatures", [])
        if overflow:
            console.print(f"\n[dim]+{len(overflow)} more (budget exhausted)[/dim]")
