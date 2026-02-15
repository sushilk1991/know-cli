"""Search commands: search, context, graph, reindex."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel

from know.cli import console, logger
from know.scanner import CodebaseScanner


@click.command()
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=10,
    help="Number of results to show"
)
@click.option(
    "--index",
    is_flag=True,
    help="Index the codebase before searching"
)
@click.option(
    "--chunk",
    is_flag=True,
    help="Search at function/class level (chunk embeddings)"
)
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, index: bool, chunk: bool) -> None:
    """Search code semantically using embeddings."""
    config = ctx.obj["config"]

    from know.semantic_search import SemanticSearcher

    searcher = SemanticSearcher(project_root=config.root)

    if index:
        if not ctx.obj.get("quiet"):
            console.print(f"[dim]Indexing {config.root}...[/dim]")
        if chunk:
            count = searcher.index_chunks(config.root)
        else:
            count = searcher.index_directory(config.root)
        if not ctx.obj.get("quiet"):
            console.print(f"[green]✓[/green] Indexed {count} {'chunks' if chunk else 'files'}")

    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Searching for: {query}[/dim]")

    import time as _time
    t0 = _time.monotonic()

    if chunk:
        results = searcher.search_chunks(query, config.root, top_k, auto_index=not index)
    else:
        results = searcher.search_code(query, config.root, top_k, auto_index=not index)

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_search(query, len(results), duration_ms)
    except Exception as e:
        logger.debug(f"Stats tracking (search) failed: {e}")

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"results": results}))
    elif ctx.obj.get("quiet"):
        for r in results:
            click.echo(f"{r['score']:.3f} {r.get('path', r.get('name', ''))}")
    else:
        if not results:
            console.print("[yellow]No results found[/yellow]")
            sys.exit(2)

        console.print(f"\n[bold]Top {len(results)} results:[/bold]\n")
        for i, r in enumerate(results, 1):
            score_color = "green" if r['score'] > 0.7 else "yellow" if r['score'] > 0.4 else "dim"
            label = r.get("path", r.get("name", ""))
            if chunk and r.get("name"):
                label = f"{r['path']}:{r['name']}" if r.get("path") else r["name"]
            console.print(f"{i}. [{score_color}]{r['score']:.3f}[/{score_color}] {label}")
            if r.get('preview'):
                preview = r['preview'][:200].replace('\n', ' ')
                console.print(f"   [dim]{preview}...[/dim]")
            console.print()


@click.command()
@click.argument("query", required=False, default=None)
@click.option(
    "--budget",
    "-b",
    type=int,
    default=8000,
    help="Token budget (default 8000)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "agent"]),
    default="markdown",
    help="Output format (markdown or agent JSON)",
)
@click.option(
    "--no-tests",
    is_flag=True,
    help="Skip test file inclusion",
)
@click.option(
    "--no-imports",
    is_flag=True,
    help="Skip import expansion",
)
@click.pass_context
def context(
    ctx: click.Context,
    query: Optional[str],
    budget: int,
    output_format: str,
    no_tests: bool,
    no_imports: bool,
) -> None:
    """Build LLM-optimized context for a query.

    Supports STDIN: echo "query" | know context --budget 4000

    Example: know context "help me fix the auth bug" --budget 8000
    """
    config = ctx.obj["config"]

    # STDIN support: read query from pipe if not provided as argument
    if query is None:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        if not query:
            click.echo("Error: query is required (pass as argument or via STDIN)", err=True)
            sys.exit(1)

    if not ctx.obj.get("quiet") and not ctx.obj.get("json") and output_format != "agent":
        console.print(f'[dim]Building context for: "{query}" (budget {budget} tokens)[/dim]')

    import time as _time
    t0 = _time.monotonic()

    from know.context_engine import ContextEngine

    engine = ContextEngine(config)
    result = engine.build_context(
        query,
        budget=budget,
        include_tests=not no_tests,
        include_imports=not no_imports,
    )

    # Inject relevant memories into context
    try:
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        memory_ctx = kb.get_relevant_context(query, max_tokens=min(500, budget // 10))
        if memory_ctx:
            result["memories_context"] = memory_ctx
    except Exception as e:
        logger.debug(f"Memory injection into context failed: {e}")

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_context(
            query, budget, result["used_tokens"], duration_ms,
        )
    except Exception as e:
        logger.debug(f"Stats tracking (context) failed: {e}")

    if ctx.obj.get("json") or output_format == "agent":
        click.echo(engine.format_agent_json(result))
    elif ctx.obj.get("quiet"):
        click.echo(engine.format_markdown(result))
    else:
        md = engine.format_markdown(result)
        from rich.markup import escape
        console.print(Panel(
            escape(md),
            title=f"🧠 Context ({result['budget_display']})",
            border_style="blue",
        ))


@click.command()
@click.argument("file_path")
@click.pass_context
def graph(ctx: click.Context, file_path: str) -> None:
    """Show import graph for a file.

    Example: know graph src/know/ai.py
    """
    config = ctx.obj["config"]

    from know.import_graph import ImportGraph

    # Ensure graph is built
    scanner = CodebaseScanner(config)
    structure = scanner.get_structure()
    ig = ImportGraph(config)
    ig.build(structure["modules"])

    # Resolve the module name from the file path
    rel = str(Path(file_path).with_suffix("")).replace("/", ".").replace("\\", ".")

    output = ig.format_graph(rel)

    if ctx.obj.get("json"):
        import json
        data = {
            "module": rel,
            "imports": ig.imports_of(rel),
            "imported_by": ig.imported_by(rel),
        }
        click.echo(json.dumps(data, indent=2))
    elif ctx.obj.get("quiet"):
        click.echo(output)
    else:
        console.print(Panel(
            output,
            title=f"📊 Import Graph: {file_path}",
            border_style="green",
        ))


@click.command()
@click.option("--chunks", is_flag=True, help="Index at function/class level (default)")
@click.option("--files", "file_level", is_flag=True, help="Index at file level (legacy)")
@click.pass_context
def reindex(ctx: click.Context, chunks: bool, file_level: bool) -> None:
    """Rebuild search embeddings from scratch.

    By default indexes at function/class level for precise search.
    """
    config = ctx.obj["config"]

    from know.semantic_search import SemanticSearcher

    if not ctx.obj.get("quiet"):
        console.print("[dim]Clearing existing embeddings...[/dim]")

    searcher = SemanticSearcher(project_root=config.root)
    searcher.clear_cache()

    if file_level and not chunks:
        if not ctx.obj.get("quiet"):
            console.print("[dim]Indexing at file level...[/dim]")
        count = searcher.index_directory(config.root)
    else:
        if not ctx.obj.get("quiet"):
            console.print("[dim]Indexing at function/class level...[/dim]")
        count = searcher.index_chunks(config.root)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"indexed": count}))
    elif ctx.obj.get("quiet"):
        click.echo(count)
    else:
        console.print(f"[green]✓[/green] Indexed [bold]{count}[/bold] chunks")
