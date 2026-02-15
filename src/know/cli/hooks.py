"""Hooks commands: hooks group with install/uninstall."""

import sys

import click

from know.cli import console
from know.git_hooks import GitHookManager


@click.group()
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
