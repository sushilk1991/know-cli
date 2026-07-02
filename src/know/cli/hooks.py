"""Hooks commands: install/uninstall/status/suggest."""

import json
import sys

import click

from know.cli import console
from know.git_hooks import GitHookManager


@click.group()
def hooks() -> None:
    """Manage git hooks for index freshness and optional docs validation."""
    pass


def _suggest_payload(agent: str) -> dict:
    """Build safe, non-mutating adoption guidance for agent integrations."""
    if agent == "claude":
        snippet = (
            "{\n"
            "  \"hooks\": {\n"
            "    \"PreToolUse\": [\n"
            "      {\n"
            "        \"matcher\": \"Bash\",\n"
            "        \"hooks\": [\n"
            "          {\n"
            "            \"type\": \"command\",\n"
            "            \"command\": \"know workflow \\\"$CLAUDE_TOOL_INPUT\\\" --session auto\"\n"
            "          }\n"
            "        ]\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}"
        )
        instructions = [
            "Use PreToolUse hook suggestions only; do not mutate tool input automatically.",
            "Start with know workflow in session mode for retrieval-first context.",
            "Keep fallback path explicit (map/context/deep) when retrieval signal is sparse.",
        ]
        strategy = "claude_pretooluse_suggest_only"
    else:
        snippet = (
            "Use skill + command mode in Codex:\n"
            "1) Install know-cli skill into CODEX_HOME/skills/know-cli/SKILL.md\n"
            "2) Use: know workflow \"<task>\" --session auto\n"
            "3) Escalate only when needed: map -> context -> deep"
        )
        instructions = [
            "Codex path is guidance-only until official hook rewrites are supported.",
            "Keep know-cli as retrieval layer, not shell command mutation layer.",
            "Use workflow as default and preserve explicit fallback commands.",
        ]
        strategy = "codex_skill_command_guidance_only"

    return {
        "mode": "suggest",
        "agent": agent,
        "strategy": strategy,
        "mutation_safe": True,
        "instructions": instructions,
        "snippet": snippet,
        "references": [
            "https://docs.anthropic.com/en/docs/claude-code/hooks",
            "https://developers.openai.com/codex/config",
        ],
    }


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
    """Install git hooks for index auto-refresh."""
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

        installed = [
            r["hook"] for r in results if r.get("status") in {"installed", "updated"}
        ]
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


@hooks.command("suggest")
@click.option(
    "--agent",
    type=click.Choice(["claude", "codex"]),
    required=True,
    help="Target agent integration.",
)
@click.pass_context
def suggest(ctx: click.Context, agent: str) -> None:
    """Suggest hook-first retrieval adoption without command mutation."""
    payload = _suggest_payload(agent)

    if ctx.obj.get("json"):
        click.echo(json.dumps(payload))
        return

    console.print(f"[bold]Hook Suggestion ({agent})[/bold]")
    console.print("[dim]Mode: suggest-only (no automatic command rewrite)[/dim]")
    for idx, item in enumerate(payload["instructions"], start=1):
        console.print(f"  {idx}. {item}")
    console.print("\n[bold]Suggested Snippet:[/bold]")
    console.print(payload["snippet"], highlight=False)
