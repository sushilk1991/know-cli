"""Core commands: init, explain, diagram, api, onboard, digest, update, watch."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from know.cli import console, logger
from know.config import Config
from know.scanner import CodebaseScanner
from know.generator import DocGenerator
from know.ai import AISummarizer
from know.watcher import FileWatcher


@click.command()
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


@click.command()
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
        summary = explanation[:300].strip()
        if summary:
            kb.remember(
                f"[{component}] {summary}",
                source="auto-explain",
                tags=component,
            )
    except Exception as e:
        logger.debug(f"Auto-store explanation memory failed: {e}")

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


@click.command()
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


@click.command()
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


@click.command()
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


@click.command()
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
        digest_lines = [l.strip() for l in digest_content.splitlines() if l.strip() and not l.startswith("#")]
        for insight in digest_lines[:3]:
            if len(insight) > 20:
                kb.remember(insight[:500], source="auto-digest", tags="digest")
    except Exception as e:
        logger.debug(f"Auto-store digest memory failed: {e}")

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


@click.command()
@click.option(
    "--all",
    "update_all",
    is_flag=True,
    default=False,
    help="Update all documentation (default when --only is omitted)",
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

    if update_all and only is not None:
        raise click.UsageError("Cannot combine --all with --only. Choose one.")

    run_all = update_all or only is None

    if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
        console.print("[bold blue]Updating documentation...[/bold blue]")

    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)

    structure = scanner.get_structure()

    results = []

    if only == "system" or run_all:
        path = generator.generate_system_doc(structure)
        results.append({"type": "system", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] System doc: [cyan]{path}[/cyan]")

    if only == "diagrams" or run_all:
        path = generator.generate_c4_diagram(structure)
        results.append({"type": "diagram", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Architecture: [cyan]{path}[/cyan]")

    if only == "api" or run_all:
        routes = scanner.extract_api_routes()
        if routes:
            path = generator.generate_openapi(routes)
            results.append({"type": "api", "path": str(path), "routes": len(routes)})
            if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
                console.print(f"[green]✓[/green] API docs: [cyan]{path}[/cyan]")

    if only == "onboarding" or run_all:
        path = generator.generate_onboarding(structure)
        results.append({"type": "onboarding", "path": str(path)})
        if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
            console.print(f"[green]✓[/green] Onboarding: [cyan]{path}[/cyan]")

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"updated": results}))
    elif not ctx.obj.get("quiet"):
        console.print("\n[bold green]Documentation updated![/bold green]")


@click.command()
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
