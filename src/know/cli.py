"""CLI entry point for know."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from know.config import Config, load_config
from know.scanner import CodebaseScanner
from know.generator import DocGenerator
from know.watcher import FileWatcher
from know.ai import AISummarizer
from know.git_hooks import GitHookManager
from know.logger import setup_logging, get_logger
from know.exceptions import KnowError

console = Console()


def get_output_format(json_output: bool, quiet: bool) -> str:
    """Determine output format based on flags."""
    if json_output:
        return "json"
    if quiet:
        return "quiet"
    return "rich"


@click.group()
@click.version_option(version=__import__("know").__version__, prog_name="know")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to config file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Only show errors")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--time", "show_time", is_flag=True, help="Show execution time")
@click.option(
    "--log-file",
    type=click.Path(),
    help="Write logs to file for debugging"
)
@click.pass_context
def cli(
    ctx: click.Context, 
    config: Optional[str], 
    verbose: bool, 
    quiet: bool,
    json_output: bool,
    show_time: bool,
    log_file: Optional[str]
) -> None:
    """know — Context Intelligence for AI Coding Agents."""
    ctx.ensure_object(dict)
    
    # Validate flag combinations
    if verbose and quiet:
        click.echo("Error: --verbose and --quiet are mutually exclusive", err=True)
        sys.exit(1)
    
    # Setup logging
    log_path = Path(log_file) if log_file else None
    setup_logging(
        verbose=verbose,
        quiet=quiet or json_output,  # JSON implies quiet
        log_file=log_path
    )
    
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["json"] = json_output
    ctx.obj["show_time"] = show_time
    
    # Record start time for --time flag
    if show_time:
        import time as _time
        global _timing_start, _timing_enabled
        _timing_start = _time.monotonic()
        _timing_enabled = True
    
    # Load configuration
    try:
        if config:
            ctx.obj["config"] = load_config(Path(config))
        else:
            ctx.obj["config"] = load_config()
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)
    
    if verbose and not quiet and not json_output:
        console.print("[dim]Verbose mode enabled[/dim]")


@cli.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Path to codebase root",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing configuration",
)
@click.pass_context
def init(ctx: click.Context, path: str, force: bool) -> None:
    """Initialize know in your project."""
    root = Path(path).resolve()
    config_path = root / ".know" / "config.yaml"
    
    if config_path.exists() and not force:
        if ctx.obj.get("quiet"):
            click.echo("Already initialized", err=True)
        else:
            console.print("[yellow]⚠ know is already initialized. Use --force to overwrite.[/yellow]")
        return
    
    if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
        console.print(Panel.fit(
            "[bold blue]Initializing know...[/bold blue]",
            title="🧠 know",
            border_style="blue"
        ))
    
    # Create .know directory
    (root / ".know").mkdir(exist_ok=True)
    (root / ".know" / "cache").mkdir(exist_ok=True)
    
    # Create default config
    config = Config.create_default(root)
    config.save(config_path)
    
    if ctx.obj.get("quiet"):
        click.echo(config_path)
    elif ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "status": "initialized",
            "config_path": str(config_path)
        }))
    else:
        console.print(f"[green]✓[/green] Created config at [cyan]{config_path}[/cyan]")
    
    # Scan codebase
    if not ctx.obj.get("quiet"):
        console.print("\n[dim]Scanning codebase...[/dim]")
    
    scanner = CodebaseScanner(config)
    
    if ctx.obj.get("verbose") or ctx.obj.get("json"):
        stats = scanner.scan()
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning files...", total=None)
            stats = scanner.scan()
            progress.update(task, completed=True)
    
    if ctx.obj.get("quiet"):
        pass  # No output
    elif ctx.obj.get("json"):
        import json
        click.echo(json.dumps(stats))
    else:
        console.print(f"[green]✓[/green] Found [bold]{stats['files']}[/bold] files")
        console.print(f"  - Functions: [bold]{stats['functions']}[/bold]")
        console.print(f"  - Classes: [bold]{stats['classes']}[/bold]")
        console.print(f"  - Modules: [bold]{stats['modules']}[/bold]")
        
        # Generate initial docs
        console.print("\n[dim]Generating initial documentation...[/dim]")
    
    generator = DocGenerator(config)
    generator.generate_all()
    
    if ctx.obj.get("quiet"):
        pass
    elif ctx.obj.get("json"):
        pass  # Already output stats
    else:
        console.print(f"[green]✓[/green] Generated documentation in [cyan]{config.output.directory}[/cyan]")
        console.print("\n[bold green]know is ready![/bold green]")
        console.print("\nNext steps:")
        console.print("  [dim]•[/dim] Run [bold]know watch[/bold] to auto-update on changes")
        console.print("  [dim]•[/dim] Run [bold]know explain <component>[/bold] for AI explanations")
        console.print("  [dim]•[/dim] Run [bold]know hooks install[/bold] for git integration")


@cli.command()
@click.option(
    "--component",
    "-c",
    required=True,
    help="Component or module to explain",
)
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Generate detailed explanation",
)
@click.pass_context
def explain(ctx: click.Context, component: str, detailed: bool) -> None:
    """Explain a specific component using AI."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Analyzing [bold]{component}[/bold]...[/dim]")
    
    scanner = CodebaseScanner(config)
    ai = AISummarizer(config)
    
    # Find component
    matches = scanner.find_component(component)
    if not matches:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": f"Component {component} not found"}))
        else:
            console.print(f"[red]✗[/red] Component [bold]{component}[/bold] not found")
        sys.exit(1)
    
    # Generate explanation
    try:
        explanation = ai.explain_component(matches[0], detailed=detailed)
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)
    
    # Auto-store explanation as memory
    try:
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        # Store a summary — first 300 chars — as an auto-memory
        summary = explanation[:300].strip()
        if summary:
            kb.remember(
                f"[{component}] {summary}",
                source="auto-explain",
                tags=component,
            )
    except Exception:
        pass

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "component": component,
            "explanation": explanation
        }))
    elif ctx.obj.get("quiet"):
        click.echo(explanation)
    else:
        console.print(Panel(
            explanation,
            title=f"📚 {component}",
            border_style="blue"
        ))


@cli.command()
@click.option(
    "--type",
    "-t",
    "diagram_type",
    type=click.Choice(["architecture", "components", "deps", "all"]),
    default="all",
    help="Type of diagram to generate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.pass_context
def diagram(ctx: click.Context, diagram_type: str, output: Optional[str]) -> None:
    """Generate architecture diagrams."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Generating [bold]{diagram_type}[/bold] diagrams...[/dim]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    
    results = []
    
    if diagram_type in ("architecture", "all"):
        path = generator.generate_c4_diagram(structure, output)
        results.append({"type": "c4", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] C4 Architecture: [cyan]{path}[/cyan]")
    
    if diagram_type in ("components", "all"):
        path = generator.generate_component_diagram(structure, output)
        results.append({"type": "component", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Component Diagram: [cyan]{path}[/cyan]")
    
    if diagram_type in ("deps", "all"):
        path = generator.generate_dependency_graph(structure, output)
        results.append({"type": "dependency", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Dependency Graph: [cyan]{path}[/cyan]")
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"diagrams": results}))


@cli.command()
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["openapi", "postman", "markdown"]),
    default="openapi",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.pass_context
def api(ctx: click.Context, output_format: str, output: Optional[str]) -> None:
    """Generate API documentation."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Generating API docs ([bold]{output_format}[/bold])...[/dim]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    routes = scanner.extract_api_routes()
    
    if not routes:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"warning": "No API routes found", "routes": []}))
        else:
            console.print("[yellow]⚠ No API routes found[/yellow]")
        return
    
    if output_format == "openapi":
        path = generator.generate_openapi(routes, output)
    elif output_format == "postman":
        path = generator.generate_postman(routes, output)
    else:
        path = generator.generate_api_markdown(routes, output)
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "format": output_format,
            "path": str(path),
            "routes_count": len(routes)
        }))
    elif ctx.obj.get("quiet"):
        click.echo(path)
    else:
        console.print(f"[green]✓[/green] Generated: [cyan]{path}[/cyan]")


@cli.command()
@click.option(
    "--for",
    "-f",
    "audience",
    required=True,
    help="Target audience (e.g., 'new devs', 'backend team')",
)
@click.option(
    "--format",
    type=click.Choice(["markdown", "pdf", "html"]),
    default="markdown",
    help="Output format",
)
@click.pass_context
def onboard(ctx: click.Context, audience: str, format: str) -> None:
    """Create onboarding guide for new team members."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Creating onboarding guide for [bold]{audience}[/bold]...[/dim]")
    
    scanner = CodebaseScanner(config)
    ai = AISummarizer(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    guide = ai.generate_onboarding_guide(structure, audience)
    
    path = generator.save_onboarding(guide, audience, format)
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "audience": audience,
            "format": format,
            "path": str(path)
        }))
    elif ctx.obj.get("quiet"):
        click.echo(path)
    else:
        console.print(f"[green]✓[/green] Onboarding guide: [cyan]{path}[/cyan]")


@cli.command()
@click.option(
    "--for-llm",
    is_flag=True,
    help="Optimize for LLM consumption (like GitIngest)",
)
@click.option(
    "--compact",
    is_flag=True,
    help="Generate compact summary",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
@click.pass_context
def digest(ctx: click.Context, for_llm: bool, compact: bool, output: Optional[str]) -> None:
    """Generate AI-optimized codebase summary."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print("[dim]Generating codebase digest...[/dim]")
    
    scanner = CodebaseScanner(config)
    ai = AISummarizer(config)
    
    structure = scanner.get_structure()
    
    if for_llm:
        digest_content = ai.generate_llm_digest(structure, compact=compact)
    else:
        digest_content = ai.generate_summary(structure, compact=compact)
    
    # Determine output path
    if output:
        out_path = Path(output)
    else:
        suffix = "-llm" if for_llm else ""
        suffix += "-compact" if compact else ""
        out_path = Path(config.output.directory) / f"digest{suffix}.md"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(digest_content)

    # Auto-store digest insights as memories
    try:
        from know.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(config)
        # Extract first 3 non-empty lines as key insights
        digest_lines = [l.strip() for l in digest_content.splitlines() if l.strip() and not l.startswith("#")]
        for insight in digest_lines[:3]:
            if len(insight) > 20:
                kb.remember(insight[:500], source="auto-digest", tags="digest")
    except Exception:
        pass

    lines = len(digest_content.splitlines())
    chars = len(digest_content)
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "path": str(out_path),
            "lines": lines,
            "characters": chars
        }))
    elif ctx.obj.get("quiet"):
        click.echo(out_path)
    else:
        console.print(f"[green]✓[/green] Digest saved: [cyan]{out_path}[/cyan]")
        console.print(f"  [dim]{lines} lines, {chars} characters[/dim]")


@cli.command()
@click.option(
    "--all",
    "update_all",
    is_flag=True,
    default=True,
    help="Update all documentation",
)
@click.option(
    "--only",
    type=click.Choice(["system", "diagrams", "api", "onboarding"]),
    help="Update only specific docs",
)
@click.pass_context
def update(ctx: click.Context, update_all: bool, only: Optional[str]) -> None:
    """Manually trigger documentation update."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print("[bold blue]Updating documentation...[/bold blue]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    
    results = []
    
    if only == "system" or update_all:
        path = generator.generate_system_doc(structure)
        results.append({"type": "system", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] System doc: [cyan]{path}[/cyan]")
    
    if only == "diagrams" or update_all:
        path = generator.generate_c4_diagram(structure)
        results.append({"type": "diagram", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Architecture: [cyan]{path}[/cyan]")
    
    if only == "api" or update_all:
        routes = scanner.extract_api_routes()
        if routes:
            path = generator.generate_openapi(routes)
            results.append({"type": "api", "path": str(path), "routes": len(routes)})
            if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
                console.print(f"[green]✓[/green] API docs: [cyan]{path}[/cyan]")
    
    if only == "onboarding" or update_all:
        path = generator.generate_onboarding(structure)
        results.append({"type": "onboarding", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Onboarding: [cyan]{path}[/cyan]")
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"updated": results}))
    elif not ctx.obj.get("quiet"):
        console.print("\n[bold green]Documentation updated![/bold green]")


@cli.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch for file changes and auto-update docs."""
    config = ctx.obj["config"]
    
    if not ctx.obj.get("quiet"):
        console.print(Panel.fit(
            "[bold blue]👁 Watching for changes...[/bold blue]\n"
            "[dim]Press Ctrl+C to stop[/dim]",
            border_style="blue"
        ))
    
    watcher = FileWatcher(config)
    
    try:
        watcher.run()
    except KeyboardInterrupt:
        if not ctx.obj.get("quiet"):
            console.print("\n[yellow]👋 Stopped watching[/yellow]")


@cli.group()
def hooks() -> None:
    """Manage git hooks."""
    pass


@hooks.command()
@click.pass_context
def install(ctx: click.Context) -> None:
    """Install git hooks for auto-updating docs."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)
    
    try:
        manager.install_post_commit()
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"status": "installed"}))
        elif ctx.obj.get("quiet"):
            pass
        else:
            console.print("[green]✓[/green] Git hooks installed")
            console.print("  [dim]Docs will update automatically after each commit[/dim]")
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)


@hooks.command()
@click.pass_context
def uninstall(ctx: click.Context) -> None:
    """Remove git hooks."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)
    
    try:
        manager.uninstall_post_commit()
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"status": "uninstalled"}))
        elif ctx.obj.get("quiet"):
            pass
        else:
            console.print("[green]✓[/green] Git hooks removed")
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)


@cli.command()
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
    except Exception:
        pass

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


@cli.command()
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
    except Exception:
        pass

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_context(
            query, budget, result["used_tokens"], duration_ms,
        )
    except Exception:
        pass

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


@cli.command()
@click.argument("file_path")
@click.pass_context
def graph(ctx: click.Context, file_path: str) -> None:
    """Show import graph for a file.
    
    Example: know graph src/know/ai.py
    """
    config = ctx.obj["config"]

    from know.import_graph import ImportGraph
    from know.scanner import CodebaseScanner

    # Ensure graph is built
    scanner = CodebaseScanner(config)
    structure = scanner.get_structure()
    ig = ImportGraph(config)
    ig.build(structure["modules"])

    # Resolve the module name from the file path
    stem = Path(file_path).stem
    # Try full module name first
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


@cli.command()
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


@cli.command()
@click.option(
    "--since",
    "-s",
    default="1 week ago",
    help="Time period to analyze (e.g., '1 week ago', '3 days ago')"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path"
)
@click.pass_context
def diff(ctx: click.Context, since: str, output: Optional[str]) -> None:
    """Show architectural changes since a given time."""
    config = ctx.obj["config"]
    
    from know.diff import ArchitectureDiff
    
    differ = ArchitectureDiff(config.root)
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Analyzing changes since: {since}[/dim]")
    
    try:
        diff_content = differ.generate_diff(since)
        
        if output:
            output_path = Path(output)
            output_path.write_text(diff_content)
            if not ctx.obj.get("quiet"):
                console.print(f"[green]✓[/green] Diff saved to {output_path}")
        else:
            # Save to docs by default
            output_path = config.root / "docs" / "architecture-diff.md"
            output_path.write_text(diff_content)
            if not ctx.obj.get("quiet"):
                console.print(f"[green]✓[/green] Diff saved to docs/architecture-diff.md")
        
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"output": str(output_path)}))
        elif not ctx.obj.get("quiet"):
            # Show summary
            console.print("\n" + diff_content.split("---")[0])  # Show only summary section
            
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error generating diff: {e}")
        sys.exit(1)


# ===================================================================
# Week 3: Knowledge base commands
# ===================================================================

@cli.command()
@click.argument("text")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--source", "-s", default="manual", help="Memory source (manual, auto-explain, auto-digest)")
@click.pass_context
def remember(ctx: click.Context, text: str, tags: str, source: str) -> None:
    """Store a memory for cross-session recall.

    Example: know remember "The auth system uses JWT with Redis session store"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    mem_id = kb.remember(text, source=source, tags=tags)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_remember(text, source)
    except Exception:
        pass

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": mem_id, "text": text, "source": source}))
    elif ctx.obj.get("quiet"):
        click.echo(str(mem_id))
    else:
        console.print(f"[green]✓[/green] Remembered (id={mem_id}): {text[:80]}")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=5, help="Max results")
@click.pass_context
def recall(ctx: click.Context, query: str, top_k: int) -> None:
    """Recall memories semantically similar to a query.

    Example: know recall "how does auth work?"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    import time as _time
    t0 = _time.monotonic()

    kb = KnowledgeBase(config)
    memories = kb.recall(query, top_k=top_k)
    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_recall(query, len(memories), duration_ms)
    except Exception:
        pass

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"query": query, "results": [m.to_dict() for m in memories]}))
    elif ctx.obj.get("quiet"):
        for m in memories:
            click.echo(f"{m.id}\t{m.text}")
    else:
        if not memories:
            console.print("[yellow]No matching memories found[/yellow]")
            sys.exit(2)
        console.print(f"\n[bold]Recalled {len(memories)} memories:[/bold]\n")
        for m in memories:
            console.print(f"  [cyan]#{m.id}[/cyan] [{m.source}] {m.text}")
            console.print(f"       [dim]{m.created_at}[/dim]")
        console.print()


@cli.command()
@click.argument("memory_id", type=int)
@click.pass_context
def forget(ctx: click.Context, memory_id: int) -> None:
    """Delete a memory by ID.

    Example: know forget 3
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    deleted = kb.forget(memory_id)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": memory_id, "deleted": deleted}))
    elif ctx.obj.get("quiet"):
        click.echo("1" if deleted else "0")
    else:
        if deleted:
            console.print(f"[green]✓[/green] Forgot memory #{memory_id}")
        else:
            console.print(f"[red]✗[/red] Memory #{memory_id} not found")
            sys.exit(1)


@cli.group()
def memories() -> None:
    """Manage stored memories (list, export, import)."""
    pass


@memories.command(name="list")
@click.option("--source", "-s", default=None, help="Filter by source")
@click.pass_context
def memories_list(ctx: click.Context, source: Optional[str]) -> None:
    """List all stored memories."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    mems = kb.list_all(source=source)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps([m.to_dict() for m in mems]))
    elif ctx.obj.get("quiet"):
        for m in mems:
            click.echo(f"{m.id}\t{m.source}\t{m.text}")
    else:
        if not mems:
            console.print("[yellow]No memories stored yet[/yellow]")
            sys.exit(2)
        console.print(f"\n[bold]📝 {len(mems)} memories:[/bold]\n")
        for m in mems:
            console.print(f"  [cyan]#{m.id}[/cyan] [{m.source}] {m.text}")
            if m.tags:
                console.print(f"       tags: {m.tags}")
            console.print(f"       [dim]{m.created_at}[/dim]")
        console.print()


@memories.command(name="export")
@click.pass_context
def memories_export(ctx: click.Context) -> None:
    """Export all memories as JSON (to stdout)."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    click.echo(kb.export_json())


@memories.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def memories_import(ctx: click.Context, file: str) -> None:
    """Import memories from a JSON file."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    data = Path(file).read_text()
    kb = KnowledgeBase(config)
    count = kb.import_json(data)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"imported": count}))
    elif ctx.obj.get("quiet"):
        click.echo(str(count))
    else:
        console.print(f"[green]✓[/green] Imported {count} memories")


# ===================================================================
# Week 3: Stats & Status commands
# ===================================================================

@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show usage statistics and ROI."""
    config = ctx.obj["config"]

    from know.stats import StatsTracker
    from know.knowledge_base import KnowledgeBase

    tracker = StatsTracker(config)
    summary = tracker.get_summary()

    # Codebase info
    scanner = CodebaseScanner(config)
    try:
        structure = scanner.get_structure()
        py_files = structure.get("file_count", len(structure.get("modules", [])))
        functions = structure.get("function_count", 0)
    except Exception:
        py_files = 0
        functions = 0

    # Memory info
    try:
        kb = KnowledgeBase(config)
        total_mem = kb.count()
        manual_mem = kb.count(source="manual")
        auto_mem = total_mem - manual_mem
    except Exception:
        total_mem = manual_mem = auto_mem = 0

    if ctx.obj.get("json"):
        import json
        data = {**summary, "memories_total": total_mem,
                "memories_manual": manual_mem, "memories_auto": auto_mem,
                "project_files": py_files, "project_functions": functions}
        click.echo(json.dumps(data, indent=2))
        return

    console.print("\n[bold]📊 know-cli Statistics[/bold]")
    console.print("─────────────────────")
    console.print(f"  Project: {config.project.name or config.root.name} ({py_files} files, {functions} functions)")
    console.print()

    console.print(f"  [bold]Knowledge Base:[/bold]")
    console.print(f"    {total_mem} memories ({manual_mem} manual, {auto_mem} auto)")
    console.print()

    console.print(f"  [bold]Context Engine:[/bold]")
    console.print(f"    Queries served: {summary['context_queries']}")
    if summary["context_queries"] > 0:
        console.print(f"    Avg budget utilization: {summary['context_budget_util']}%")
        console.print(f"    Avg response time: {summary['context_avg_ms']}ms")
    console.print()

    console.print(f"  [bold]Search:[/bold]")
    console.print(f"    Queries: {summary['search_queries']}")
    if summary["search_queries"] > 0:
        console.print(f"    Avg response time: {summary['search_avg_ms']}ms")
    console.print()

    console.print(f"  [bold]Memory Ops:[/bold]")
    console.print(f"    Remember calls: {summary['remember_count']}")
    console.print(f"    Recall calls: {summary['recall_count']}")
    console.print()


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Quick project health check."""
    config = ctx.obj["config"]

    from know import __version__

    # Codebase info
    scanner = CodebaseScanner(config)
    try:
        structure = scanner.get_structure()
        modules = structure.get("modules", [])
        n_files = structure.get("file_count", len(modules))
        n_functions = structure.get("function_count", 0)
    except Exception:
        n_files = n_functions = 0

    # Index info
    index_chunks = 0
    index_age = "unknown"
    cache_dir = config.root / ".know" / "cache"
    if cache_dir.exists():
        for db_file in cache_dir.glob("*.db"):
            try:
                import time as _time
                mtime = db_file.stat().st_mtime
                age_s = _time.time() - mtime
                if age_s < 3600:
                    index_age = f"{int(age_s / 60)}m ago"
                elif age_s < 86400:
                    index_age = f"{int(age_s / 3600)}h ago"
                else:
                    index_age = f"{int(age_s / 86400)}d ago"
            except Exception:
                pass

    # Memories
    mem_count = 0
    try:
        from know.knowledge_base import KnowledgeBase
        mem_count = KnowledgeBase(config).count()
    except Exception:
        pass

    # Cache size
    cache_size = "0 B"
    total_bytes = 0
    know_dir = config.root / ".know"
    if know_dir.exists():
        for f in know_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
        if total_bytes > 1024 * 1024:
            cache_size = f"{total_bytes / 1024 / 1024:.1f} MB"
        elif total_bytes > 1024:
            cache_size = f"{total_bytes / 1024:.1f} KB"
        else:
            cache_size = f"{total_bytes} B"

    # Config check
    config_ok = (config.root / ".know" / "config.yaml").exists()

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "version": __version__,
            "project": str(config.root),
            "files": n_files,
            "functions": n_functions,
            "index_age": index_age,
            "memories": mem_count,
            "cache_size": cache_size,
            "config_ok": config_ok,
        }, indent=2))
        return

    console.print(f"\n[green]✓[/green] [bold]know-cli v{__version__}[/bold]")
    console.print(f"  Project: {config.root}")
    console.print(f"  Files: {n_files} Python")
    console.print(f"  Functions: {n_functions}")
    console.print(f"  Indexed: {index_age}")
    console.print(f"  Memories: {mem_count}")
    console.print(f"  Cache: {cache_size}")
    if config_ok:
        console.print("  Config: .know/config.yaml [green]✓[/green]")
    else:
        console.print("  Config: [red]not initialized[/red]")
    console.print()


# ===================================================================
# Week 4: MCP Server commands
# ===================================================================

@cli.group()
def mcp() -> None:
    """MCP (Model Context Protocol) server for AI agents."""
    pass


@mcp.command(name="serve")
@click.option("--sse", is_flag=True, help="Use SSE transport instead of stdio")
@click.option("--port", type=int, default=3000, help="Port for SSE transport (default 3000)")
@click.pass_context
def mcp_serve(ctx: click.Context, sse: bool, port: int) -> None:
    """Start the MCP server.

    Default: stdio transport (for Claude Desktop).
    Use --sse for web clients.

    \b
    Examples:
      know mcp serve                    # stdio transport
      know mcp serve --sse --port 3000  # SSE transport
    """
    try:
        from know.mcp_server import run_server
    except ImportError:
        click.echo(
            "Error: The 'mcp' package is required.\n\n"
            "  pip install mcp\n\n"
            "Or install know-cli with:\n\n"
            "  pip install know-cli[mcp]\n",
            err=True,
        )
        sys.exit(1)

    config = ctx.obj["config"]
    run_server(sse=sse, port=port, project_root=config.root)


@mcp.command(name="config")
@click.pass_context
def mcp_config(ctx: click.Context) -> None:
    """Print Claude Desktop configuration snippet.

    Copy the output into your Claude Desktop config file.
    """
    try:
        from know.mcp_server import print_config
    except ImportError:
        click.echo(
            "Error: The 'mcp' package is required.\n\n"
            "  pip install mcp\n",
            err=True,
        )
        sys.exit(1)

    config = ctx.obj["config"]
    print_config(project_root=config.root)


def get_shell_config_path(shell: str) -> str:
    """Get the config file path for a shell."""
    home = Path.home()
    if shell == "bash":
        return str(home / ".bashrc")
    elif shell == "zsh":
        return str(home / ".zshrc")
    elif shell == "fish":
        return str(home / ".config" / "fish" / "config.fish")
    return str(home / ".profile")


# Global timing state — set by --time flag, read at exit
_timing_start: Optional[float] = None
_timing_enabled: bool = False


def _print_timing():
    """Print execution time if --time flag was set."""
    global _timing_start, _timing_enabled
    if _timing_enabled and _timing_start is not None:
        import time as _time
        elapsed = _time.monotonic() - _timing_start
        if elapsed < 1:
            console.print(f"[dim]⏱ {elapsed * 1000:.0f}ms[/dim]", highlight=False)
        else:
            console.print(f"[dim]⏱ {elapsed:.2f}s[/dim]", highlight=False)


def main() -> None:
    """Entry point."""
    import atexit
    atexit.register(_print_timing)

    try:
        cli()
    except KnowError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        from rich.markup import escape
        console.print(f"[red]Unexpected error:[/red] {escape(str(e))}")
        if os.environ.get("KNOW_DEBUG"):
            import traceback
            console.print(escape(traceback.format_exc()))
        sys.exit(1)


if __name__ == "__main__":
    main()
