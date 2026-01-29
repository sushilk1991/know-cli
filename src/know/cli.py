"""CLI entry point for know."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from know.config import Config, load_config
from know.scanner import CodebaseScanner
from know.generator import DocGenerator
from know.watcher import FileWatcher
from know.ai import AISummarizer
from know.git_hooks import GitHookManager

console = Console()


@click.group()
@click.version_option(version=__import__("know").__version__, prog_name="know")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to config file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool) -> None:
    """Living documentation generator for codebases."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    # Load configuration
    if config:
        ctx.obj["config"] = load_config(Path(config))
    else:
        ctx.obj["config"] = load_config()
    
    if verbose:
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
        console.print("[yellow]⚠ know is already initialized. Use --force to overwrite.[/yellow]")
        return
    
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
    
    console.print(f"[green]✓[/green] Created config at [cyan]{config_path}[/cyan]")
    
    # Scan codebase
    console.print("\n[dim]Scanning codebase...[/dim]")
    scanner = CodebaseScanner(config)
    stats = scanner.scan()
    
    console.print(f"[green]✓[/green] Found [bold]{stats['files']}[/bold] files")
    console.print(f"  - Functions: [bold]{stats['functions']}[/bold]")
    console.print(f"  - Classes: [bold]{stats['classes']}[/bold]")
    console.print(f"  - Modules: [bold]{stats['modules']}[/bold]")
    
    # Generate initial docs
    console.print("\n[dim]Generating initial documentation...[/dim]")
    generator = DocGenerator(config)
    generator.generate_all()
    
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
    
    console.print(f"[dim]Analyzing [bold]{component}[/bold]...[/dim]")
    
    scanner = CodebaseScanner(config)
    ai = AISummarizer(config)
    
    # Find component
    matches = scanner.find_component(component)
    if not matches:
        console.print(f"[red]✗[/red] Component [bold]{component}[/bold] not found")
        return
    
    # Generate explanation
    explanation = ai.explain_component(matches[0], detailed=detailed)
    
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
    
    console.print(f"[dim]Generating [bold]{diagram_type}[/bold] diagrams...[/dim]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    
    if diagram_type in ("architecture", "all"):
        path = generator.generate_c4_diagram(structure, output)
        console.print(f"[green]✓[/green] C4 Architecture: [cyan]{path}[/cyan]")
    
    if diagram_type in ("components", "all"):
        path = generator.generate_component_diagram(structure, output)
        console.print(f"[green]✓[/green] Component Diagram: [cyan]{path}[/cyan]")
    
    if diagram_type in ("deps", "all"):
        path = generator.generate_dependency_graph(structure, output)
        console.print(f"[green]✓[/green] Dependency Graph: [cyan]{path}[/cyan]")


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
    
    console.print(f"[dim]Generating API docs ([bold]{output_format}[/bold])...[/dim]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    routes = scanner.extract_api_routes()
    
    if not routes:
        console.print("[yellow]⚠ No API routes found[/yellow]")
        return
    
    if output_format == "openapi":
        path = generator.generate_openapi(routes, output)
    elif output_format == "postman":
        path = generator.generate_postman(routes, output)
    else:
        path = generator.generate_api_markdown(routes, output)
    
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
    
    console.print(f"[dim]Creating onboarding guide for [bold]{audience}[/bold]...[/dim]")
    
    scanner = CodebaseScanner(config)
    ai = AISummarizer(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    guide = ai.generate_onboarding_guide(structure, audience)
    
    path = generator.save_onboarding(guide, audience, format)
    
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
    
    console.print(f"[green]✓[/green] Digest saved: [cyan]{out_path}[/cyan]")
    
    # Show stats
    lines = len(digest_content.splitlines())
    chars = len(digest_content)
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
    type=click.Choice(["readme", "diagrams", "api", "onboarding"]),
    help="Update only specific docs",
)
@click.pass_context
def update(ctx: click.Context, update_all: bool, only: Optional[str]) -> None:
    """Manually trigger documentation update."""
    config = ctx.obj["config"]
    
    console.print("[bold blue]Updating documentation...[/bold blue]")
    
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)
    
    structure = scanner.get_structure()
    
    if only == "readme" or update_all:
        path = generator.generate_readme(structure)
        console.print(f"[green]✓[/green] README: [cyan]{path}[/cyan]")
    
    if only == "diagrams" or update_all:
        path = generator.generate_c4_diagram(structure)
        console.print(f"[green]✓[/green] Architecture: [cyan]{path}[/cyan]")
    
    if only == "api" or update_all:
        routes = scanner.extract_api_routes()
        if routes:
            path = generator.generate_openapi(routes)
            console.print(f"[green]✓[/green] API docs: [cyan]{path}[/cyan]")
    
    if only == "onboarding" or update_all:
        path = generator.generate_onboarding(structure)
        console.print(f"[green]✓[/green] Onboarding: [cyan]{path}[/cyan]")
    
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
    
    manager.install_post_commit()
    console.print("[green]✓[/green] Git hooks installed")
    console.print("  [dim]Docs will update automatically after each commit[/dim]")


@hooks.command()
@click.pass_context
def uninstall(ctx: click.Context) -> None:
    """Remove git hooks."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)
    
    manager.uninstall_post_commit()
    console.print("[green]✓[/green] Git hooks removed")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
