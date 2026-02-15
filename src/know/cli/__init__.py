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
from know.cli.knowledge import remember, recall, forget, memories
from know.cli.stats import stats, status
from know.cli.hooks import hooks
from know.cli.mcp import mcp
from know.cli.agent import next_file, signatures, related, generate_context
from know.cli.diff import diff

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
cli.add_command(diff)


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
