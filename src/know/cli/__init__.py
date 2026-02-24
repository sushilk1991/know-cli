"""CLI entry point for know."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from know.config import Config, load_config
from know.logger import setup_logging, get_logger
from know.exceptions import KnowError

console = Console()
logger = get_logger()


class KnowCLIGroup(click.Group):
    """Top-level CLI group with curated default help output."""

    SIMPLE_COMMANDS = (
        "init",
        "workflow",
        "ask",
        "recall",
        "decide",
        "done",
        "docs",
        "status",
        "commands",
    )

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = []
        for subcommand in self.SIMPLE_COMMANDS:
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            rows.append((subcommand, cmd.get_short_help_str()))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)
            formatter.write_paragraph()

        legacy = sorted(
            name for name in self.commands.keys() if name not in self.SIMPLE_COMMANDS
        )
        if legacy:
            formatter.write_text(
                "Legacy/advanced commands (still supported): "
                + ", ".join(legacy)
            )
            formatter.write_paragraph()

        formatter.write_text(
            "Run `know commands --all` to view advanced and legacy commands."
        )


def get_output_format(json_output: bool, quiet: bool) -> str:
    """Determine output format based on flags."""
    if json_output:
        return "json"
    if quiet:
        return "quiet"
    return "rich"


@click.group(cls=KnowCLIGroup)
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


# -----------------------------------------------------------------------
# Register all sub-module commands
# -----------------------------------------------------------------------
from know.cli.core import init, explain, diagram, api, onboard, digest, update, watch
from know.cli.search import search, context, graph, reindex
from know.cli.knowledge import remember, recall, forget, memories, decide
from know.cli.stats import stats, status
from know.cli.hooks import hooks
from know.cli.mcp import mcp
from know.cli.agent import (
    next_file, signatures, related, generate_context, callers, callees, map_cmd, workflow, deep,
)
from know.cli.diff import diff
from know.cli.doctor import doctor


@click.command("docs")
@click.option(
    "--only",
    type=click.Choice(["all", "digest", "api", "diagram"]),
    default="all",
    help="Generate only a subset of docs",
)
@click.pass_context
def docs(ctx: click.Context, only: str) -> None:
    """Generate core docs in one command (digest + API + architecture)."""
    import json as _json
    from know.generator import DocGenerator
    from know.scanner import CodebaseScanner

    config = ctx.obj["config"]
    scanner = CodebaseScanner(config)
    generator = DocGenerator(config)

    structure = scanner.get_structure()
    results = {}

    def _render_local_system_doc() -> str:
        lines = [
            f"# {config.project.name}",
            "",
            config.project.description or "Project documentation generated by know.",
            "",
            "## Project Structure",
            "",
        ]
        for module in structure.get("modules", [])[:15]:
            name = module.get("name", "unknown")
            path = module.get("path", "")
            lines.append(f"- `{name}` ({path})" if path else f"- `{name}`")

        lines.extend(
            [
                "",
                "## Statistics",
                "",
                f"- Files: {structure.get('file_count', 0)}",
                f"- Modules: {structure.get('module_count', 0)}",
                f"- Functions: {structure.get('function_count', 0)}",
                f"- Classes: {structure.get('class_count', 0)}",
                "",
                "*Generated by know docs*",
            ]
        )
        return "\n".join(lines) + "\n"

    def _render_local_digest() -> str:
        lines = [
            "# Codebase Digest",
            "",
            f"- Files: {structure.get('file_count', 0)}",
            f"- Modules: {structure.get('module_count', 0)}",
            f"- Functions: {structure.get('function_count', 0)}",
            f"- Classes: {structure.get('class_count', 0)}",
            "",
            "## Top Modules",
            "",
        ]
        for module in structure.get("modules", [])[:25]:
            name = module.get("name", "unknown")
            fcount = module.get("function_count", 0)
            ccount = module.get("class_count", 0)
            lines.append(f"- `{name}` ({fcount} functions, {ccount} classes)")

        lines.extend(["", "*Generated by know docs*"])
        return "\n".join(lines) + "\n"

    # System architecture summary doc
    if only == "all":
        system_path = (config.root / config.output.directory / "arc.md")
        system_path.parent.mkdir(parents=True, exist_ok=True)
        system_path.write_text(_render_local_system_doc())
        results["system"] = str(system_path)

    # LLM-oriented digest
    if only in ("all", "digest"):
        digest_content = _render_local_digest()
        digest_path = (config.root / config.output.directory / "digest-llm-compact.md")
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.write_text(digest_content)
        results["digest"] = str(digest_path)

    # API docs (if routes found)
    if only in ("all", "api"):
        routes = scanner.extract_api_routes()
        if routes:
            api_path = generator.generate_openapi(routes)
            results["api"] = str(api_path)
            results["api_routes"] = len(routes)
        else:
            results["api"] = None
            results["api_routes"] = 0

    # Architecture diagram
    if only in ("all", "diagram"):
        diagram_path = generator.generate_c4_diagram(structure)
        results["diagram"] = str(diagram_path)

    if ctx.obj.get("json"):
        click.echo(_json.dumps(results))
    elif ctx.obj.get("quiet"):
        for key in ("system", "digest", "api", "diagram"):
            value = results.get(key)
            if value:
                click.echo(value)
    else:
        console.print("[green]✓[/green] Documentation refresh complete")
        if results.get("system"):
            console.print(f"  system: [cyan]{results['system']}[/cyan]")
        if results.get("digest"):
            console.print(f"  digest: [cyan]{results['digest']}[/cyan]")
        if results.get("diagram"):
            console.print(f"  diagram: [cyan]{results['diagram']}[/cyan]")
        if results.get("api"):
            console.print(f"  api: [cyan]{results['api']}[/cyan] ({results.get('api_routes', 0)} routes)")
        elif only in ("all", "api"):
            console.print("  [yellow]api:[/yellow] no routes detected")


@click.command("commands")
@click.option("--all", "show_all", is_flag=True, help="Show all commands (advanced + legacy)")
@click.pass_context
def commands_cmd(ctx: click.Context, show_all: bool) -> None:
    """List available commands."""
    import json as _json

    if show_all:
        names = sorted(cli.commands.keys())
    else:
        names = [name for name in KnowCLIGroup.SIMPLE_COMMANDS if name in cli.commands]

    if ctx.obj.get("json"):
        click.echo(_json.dumps({"commands": names, "all": show_all}))
    else:
        for name in names:
            cmd = cli.commands.get(name)
            short = cmd.get_short_help_str() if cmd else ""
            click.echo(f"{name:10} {short}")


@click.command("ask")
@click.argument("query")
@click.option(
    "--budget",
    "-b",
    type=int,
    default=4000,
    help="Context budget for retrieval (default 4000)",
)
@click.option(
    "--session",
    "session_id",
    default="auto",
    help="Session ID for dedup ('auto' generates)",
)
@click.option("--include-tests", is_flag=True, help="Include test files in results")
@click.pass_context
def ask(
    ctx: click.Context,
    query: str,
    budget: int,
    session_id: str,
    include_tests: bool,
) -> None:
    """Simple one-command retrieval for coding tasks.

    Alias wrapper around `know workflow` with sensible defaults.
    """
    deep_budget = max(1200, int(budget * 0.75))
    ctx.invoke(
        workflow,
        query=query,
        map_limit=20,
        context_budget=budget,
        deep_budget=deep_budget,
        session_id=session_id,
        include_tests=include_tests,
    )


@click.command("done")
@click.argument("memory_id", type=int)
@click.option(
    "--status",
    default="resolved",
    type=click.Choice(["resolved", "superseded", "rejected", "active"]),
    help="Resolution status for the memory",
)
@click.pass_context
def done(ctx: click.Context, memory_id: int, status: str) -> None:
    """Shortcut for resolving a memory.

    Alias for: know memories resolve <memory_id> --status <status>
    """
    resolve_cmd = memories.commands.get("resolve")
    if resolve_cmd is None:
        click.echo("Error: memory resolve command unavailable", err=True)
        sys.exit(1)
    ctx.invoke(resolve_cmd, memory_id=memory_id, status=status)

cli.add_command(init)
cli.add_command(explain)
cli.add_command(diagram)
cli.add_command(api)
cli.add_command(onboard)
cli.add_command(digest)
cli.add_command(update)
cli.add_command(watch)
cli.add_command(search)
cli.add_command(context)
cli.add_command(graph)
cli.add_command(reindex)
cli.add_command(remember)
cli.add_command(decide)
cli.add_command(recall)
cli.add_command(forget)
cli.add_command(memories)
cli.add_command(stats)
cli.add_command(status)
cli.add_command(hooks)
cli.add_command(mcp)
cli.add_command(next_file)
cli.add_command(signatures)
cli.add_command(related)
cli.add_command(generate_context)
cli.add_command(callers)
cli.add_command(callees)
cli.add_command(map_cmd)
cli.add_command(workflow)
cli.add_command(deep)
cli.add_command(diff)
cli.add_command(docs)
cli.add_command(ask)
cli.add_command(done)
cli.add_command(commands_cmd)
cli.add_command(doctor)


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
