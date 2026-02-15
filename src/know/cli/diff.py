"""Diff command: show architectural changes."""

import sys
from pathlib import Path
from typing import Optional

import click

from know.cli import console


@click.command()
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
