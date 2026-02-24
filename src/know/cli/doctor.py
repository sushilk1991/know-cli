"""Reliability command: diagnose and repair local runtime issues."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import click

from know.cli import console


def _default_fastembed_cache_root() -> Path:
    """Return the effective fastembed cache root path used by know."""
    env_path = os.environ.get("FASTEMBED_CACHE_PATH")
    if env_path:
        return Path(env_path)
    return Path.home() / ".cache" / "know-cli" / "fastembed"


def _probe_embedding_model(model_name: str) -> Tuple[bool, Optional[str]]:
    """Return whether embedding model can be loaded, plus optional error text."""
    from know import embeddings as emb

    try:
        emb._model_cache.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    model = emb.get_model(model_name)
    if model is None:
        return False, "embedding model unavailable"
    return True, None


def _repair_fastembed_cache(cache_root: Path) -> str:
    """Clear fastembed cache root and recreate directory."""
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    return f"cleared cache at {cache_root}"


def _run_reindex_silent(config) -> Tuple[bool, Optional[int], Optional[str]]:
    """Reindex chunks without emitting CLI output."""
    try:
        from know.semantic_search import SemanticSearcher

        searcher = SemanticSearcher(project_root=config.root)
        searcher.clear_cache()
        count = searcher.index_chunks(config.root)
        return True, count, None
    except Exception as e:
        return False, None, str(e)


@click.command("doctor")
@click.option("--repair", is_flag=True, help="Attempt automatic repairs")
@click.option("--reindex", "run_reindex", is_flag=True, help="Run `know reindex` after successful repair")
@click.pass_context
def doctor(ctx: click.Context, repair: bool, run_reindex: bool) -> None:
    """Diagnose and repair local cache/model issues."""
    from know.embeddings import DEFAULT_MODEL, _configure_fastembed_cache_dir
    from know.skill_installer import skill_install_status

    config = ctx.obj["config"]
    _configure_fastembed_cache_dir()
    cache_root = _default_fastembed_cache_root()

    report = {
        "ok": True,
        "checks": {},
        "actions": [],
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "know_executable": sys.argv[0],
            "workflow_command_available": bool(
                getattr(ctx.find_root().command, "commands", {}).get("workflow")
            ),
        },
    }

    report["checks"]["fastembed_cache"] = {
        "path": str(cache_root),
        "exists": cache_root.exists(),
    }
    report["checks"]["agent_skill"] = skill_install_status()

    model_ok, model_error = _probe_embedding_model(DEFAULT_MODEL)
    report["checks"]["embedding_model"] = {
        "model": DEFAULT_MODEL,
        "ok": model_ok,
        "error": model_error,
    }

    if not model_ok:
        report["ok"] = False

    if repair and not model_ok:
        action = _repair_fastembed_cache(cache_root)
        report["actions"].append(action)

        model_ok_after, model_error_after = _probe_embedding_model(DEFAULT_MODEL)
        report["checks"]["embedding_model_after_repair"] = {
            "model": DEFAULT_MODEL,
            "ok": model_ok_after,
            "error": model_error_after,
        }
        report["ok"] = model_ok_after

        if model_ok_after and run_reindex:
            ok, count, error = _run_reindex_silent(config)
            if ok:
                report["actions"].append(f"reindex indexed {count} chunks")
            else:
                report["actions"].append(f"reindex failed: {error}")

    if ctx.obj.get("json"):
        click.echo(json.dumps(report))
        return

    status = "healthy" if report["ok"] else "issues found"
    color = "green" if report["ok"] else "yellow"
    console.print(f"[{color}]doctor: {status}[/{color}]")
    console.print(f"  fastembed_cache: {report['checks']['fastembed_cache']['path']}")
    skill_check = report["checks"].get("agent_skill", {})
    skill_targets = skill_check.get("targets", {})
    installed_targets = [
        name for name, meta in skill_targets.items() if meta.get("exists")
    ]
    if installed_targets:
        console.print(
            "  agent_skill: [green]installed[/green] "
            + ", ".join(sorted(installed_targets))
        )
    else:
        console.print("  agent_skill: [yellow]not found[/yellow]")

    embed_check = report["checks"].get("embedding_model", {})
    if embed_check.get("ok"):
        console.print(f"  embedding_model: [green]ok[/green] ({embed_check.get('model')})")
    else:
        console.print(
            f"  embedding_model: [yellow]failed[/yellow] ({embed_check.get('error')})"
        )

    post = report["checks"].get("embedding_model_after_repair")
    if post is not None:
        if post.get("ok"):
            console.print("  embedding_model_after_repair: [green]ok[/green]")
        else:
            console.print("  embedding_model_after_repair: [red]failed[/red]")

    if report["actions"]:
        console.print("  actions:")
        for action in report["actions"]:
            console.print(f"    - {action}")
