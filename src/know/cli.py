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
    log_file: Optional[str]
) -> None:
    """Living documentation generator for codebases."""
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
@click.option(
    "--daemon",
    "-d",
    is_flag=True,
    help="Run as daemon process",
)
@click.pass_context
def watch(ctx: click.Context, daemon: bool) -> None:
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
        if daemon:
            watcher.run_daemon()
        else:
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
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell type (auto-detected if not specified)",
)
def completion(shell: Optional[str]) -> None:
    """Generate shell completion script."""
    import click_completion
    
    if not shell:
        # Auto-detect
        shell = os.path.basename(os.environ.get("SHELL", ""))
        if shell not in ["bash", "zsh", "fish"]:
            console.print("[red]Could not detect shell. Please specify --shell.[/red]")
            console.print("\nExample:")
            console.print("  [bold]know completion --shell bash[/bold]")
            sys.exit(1)
    
    script = click_completion.get_code(shell, prog_name="know")
    
    console.print(f"[dim]# Add this to your shell configuration file:[/dim]")
    console.print(f"[dim]# ({get_shell_config_path(shell)})[/dim]")
    console.print()
    console.print(script)
    console.print()
    console.print(f"[dim]# Or run this command to add it automatically:[/dim]")
    console.print(f"[bold]know completion --shell {shell} >> {get_shell_config_path(shell)}[/bold]")


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
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, index: bool) -> None:
    """Search code semantically using embeddings."""
    config = ctx.obj["config"]
    
    from know.semantic_search import SemanticSearcher
    
    searcher = SemanticSearcher()
    
    if index:
        if not ctx.obj.get("quiet"):
            console.print(f"[dim]Indexing {config.root}...[/dim]")
        count = searcher.index_directory(config.root)
        if not ctx.obj.get("quiet"):
            console.print(f"[green]✓[/green] Indexed {count} files")
    
    if not ctx.obj.get("quiet"):
        console.print(f"[dim]Searching for: {query}[/dim]")
    
    # Avoid re-indexing if we just did it manually
    results = searcher.search_code(query, config.root, top_k, auto_index=not index)
    
    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"results": results}))
    elif ctx.obj.get("quiet"):
        for r in results:
            click.echo(f"{r['score']:.3f} {r['path']}")
    else:
        if not results:
            console.print("[yellow]No results found[/yellow]")
            return
        
        console.print(f"\n[bold]Top {len(results)} results:[/bold]\n")
        for i, r in enumerate(results, 1):
            score_color = "green" if r['score'] > 0.7 else "yellow" if r['score'] > 0.4 else "dim"
            console.print(f"{i}. [{score_color}]{r['score']:.3f}[/{score_color}] {r['path']}")
            if r['preview']:
                preview = r['preview'][:200].replace('\n', ' ')
                console.print(f"   [dim]{preview}...[/dim]")
            console.print()


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


def main() -> None:
    """Entry point."""
    try:
        cli()
    except KnowError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        if os.environ.get("KNOW_DEBUG"):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
