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
