"""Hooks commands: install/uninstall/status git hooks."""

import sys

import click

from know.cli import console
from know.git_hooks import GitHookManager


@click.group()
def hooks() -> None:
    """Manage git hooks for docs + index freshness."""
    pass


@hooks.command()
@click.option(
    "--pre-commit",
    is_flag=True,
    help="Also install a pre-commit docs validation hook.",
)
@click.option(
    "--index-hooks/--no-index-hooks",
    default=False,
    show_default=True,
    help="Install post-merge/post-checkout hooks that refresh know's index cache.",
)
@click.pass_context
def install(ctx: click.Context, pre_commit: bool, index_hooks: bool) -> None:
    """Install git hooks for docs refresh and index auto-refresh."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)

    try:
        results: list[dict] = []
        results.append(manager.install_post_commit())
        if index_hooks:
            results.append(manager.install_post_merge())
            results.append(manager.install_post_checkout())
        if pre_commit:
            results.append(manager.install_pre_commit())

        if any(r.get("status") == "not_git_repo" for r in results):
            raise RuntimeError("Not a git repository")

        installed = [r["hook"] for r in results if r.get("status") == "installed"]
        command_status = "installed" if installed else "no_change"

        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({
                "status": command_status,
                "installed_hooks": installed,
                "results": results,
                "index_hooks": index_hooks,
                "pre_commit": pre_commit,
            }))
        elif ctx.obj.get("quiet"):
            pass
        else:
            if installed:
                console.print("[green]✓[/green] Git hooks installed")
            else:
                console.print("[yellow]⚠[/yellow] No hook changes")
            if results:
                for row in results:
                    hook = row.get("hook", "unknown")
                    status = row.get("status", "unknown")
                    console.print(f"  [dim]{hook}: {status}[/dim]")
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)


@hooks.command()
@click.option(
    "--pre-commit",
    is_flag=True,
    help="Also remove pre-commit docs validation hook.",
)
@click.option(
    "--index-hooks/--no-index-hooks",
    default=False,
    show_default=True,
    help="Remove post-merge/post-checkout index refresh hooks.",
)
@click.pass_context
def uninstall(ctx: click.Context, pre_commit: bool, index_hooks: bool) -> None:
    """Remove know-managed git hooks."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)

    try:
        results: list[dict] = []
        results.append(manager.uninstall_post_commit())
        if index_hooks:
            results.append(manager.uninstall_post_merge())
            results.append(manager.uninstall_post_checkout())
        if pre_commit:
            results.append(manager.uninstall_pre_commit())

        if any(r.get("status") == "not_git_repo" for r in results):
            raise RuntimeError("Not a git repository")

        removed = [r["hook"] for r in results if r.get("status") == "removed"]
        command_status = "uninstalled" if removed else "no_change"

        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({
                "status": command_status,
                "removed_hooks": removed,
                "results": results,
                "index_hooks": index_hooks,
                "pre_commit": pre_commit,
            }))
        elif ctx.obj.get("quiet"):
            pass
        else:
            if removed:
                console.print("[green]✓[/green] Git hooks removed")
            else:
                console.print("[yellow]⚠[/yellow] No hook changes")
            if results:
                for row in results:
                    hook = row.get("hook", "unknown")
                    status = row.get("status", "unknown")
                    console.print(f"  [dim]{hook}: {status}[/dim]")
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)


@hooks.command("status")
@click.pass_context
def hooks_status(ctx: click.Context) -> None:
    """Show status of know-managed hooks."""
    config = ctx.obj["config"]
    manager = GitHookManager(config)

    try:
        statuses = manager.status()
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"hooks": statuses}))
            return

        if not statuses:
            console.print("[red]✗[/red] Not a git repository")
            return

        console.print("[bold]Git hook status:[/bold]")
        for hook_name in ("post-commit", "post-merge", "post-checkout", "pre-commit"):
            state = statuses.get(hook_name, "missing")
            if state == "installed_by_know":
                console.print(f"  [green]✓[/green] {hook_name}: installed by know")
            elif state == "present_other":
                console.print(f"  [yellow]⚠[/yellow] {hook_name}: exists (not managed by know)")
            else:
                console.print(f"  [dim]✗ {hook_name}: not installed[/dim]")
    except Exception as e:
        if ctx.obj.get("json"):
            import json
            click.echo(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]✗[/red] Error: {e}")
        sys.exit(1)
