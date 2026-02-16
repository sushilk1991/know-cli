"""Stats commands: stats, status."""

import sys
from pathlib import Path
from typing import Optional

import click

from know.cli import console, logger
from know.scanner import CodebaseScanner


@click.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show usage statistics and ROI."""
    config = ctx.obj["config"]

    from know.stats import StatsTracker
    from know.knowledge_base import KnowledgeBase

    tracker = StatsTracker(config)
    summary = tracker.get_summary()

    # Codebase info
    scanner = CodebaseScanner(config)
    try:
        structure = scanner.get_structure()
        py_files = structure.get("file_count", len(structure.get("modules", [])))
        functions = structure.get("function_count", 0)
    except Exception as e:
        logger.debug(f"Codebase stats scan failed: {e}")
        py_files = 0
        functions = 0

    # Memory info
    try:
        kb = KnowledgeBase(config)
        total_mem = kb.count()
        manual_mem = kb.count(source="manual")
        auto_mem = total_mem - manual_mem
    except Exception as e:
        logger.debug(f"Memory stats retrieval failed: {e}")
        total_mem = manual_mem = auto_mem = 0

    if ctx.obj.get("json"):
        import json
        data = {**summary, "memories_total": total_mem,
                "memories_manual": manual_mem, "memories_auto": auto_mem,
                "project_files": py_files, "project_functions": functions}
        click.echo(json.dumps(data, indent=2))
        return

    console.print("\n[bold]📊 know-cli Statistics[/bold]")
    console.print("─────────────────────")
    console.print(f"  Project: {config.project.name or config.root.name} ({py_files} files, {functions} functions)")
    console.print()

    console.print(f"  [bold]Knowledge Base:[/bold]")
    console.print(f"    {total_mem} memories ({manual_mem} manual, {auto_mem} auto)")
    console.print()

    console.print(f"  [bold]Context Engine:[/bold]")
    console.print(f"    Queries served: {summary['context_queries']}")
    if summary["context_queries"] > 0:
        console.print(f"    Avg budget utilization: {summary['context_budget_util']}%")
        console.print(f"    Avg response time: {summary['context_avg_ms']}ms")
    console.print()

    console.print(f"  [bold]Search:[/bold]")
    console.print(f"    Queries: {summary['search_queries']}")
    if summary["search_queries"] > 0:
        console.print(f"    Avg response time: {summary['search_avg_ms']}ms")
    console.print()

    console.print(f"  [bold]Memory Ops:[/bold]")
    console.print(f"    Remember calls: {summary['remember_count']}")
    console.print(f"    Recall calls: {summary['recall_count']}")
    console.print()


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Quick project health check."""
    config = ctx.obj["config"]

    from know import __version__

    # Codebase info
    scanner = CodebaseScanner(config)
    try:
        structure = scanner.get_structure()
        modules = structure.get("modules", [])
        n_files = structure.get("file_count", len(modules))
        n_functions = structure.get("function_count", 0)
    except Exception as e:
        logger.debug(f"Status codebase scan failed: {e}")
        n_files = n_functions = 0

    # Index info
    index_age = "unknown"
    cache_dir = config.root / ".know" / "cache"
    if cache_dir.exists():
        for db_file in cache_dir.glob("*.db"):
            try:
                import time as _time
                mtime = db_file.stat().st_mtime
                age_s = _time.time() - mtime
                if age_s < 3600:
                    index_age = f"{int(age_s / 60)}m ago"
                elif age_s < 86400:
                    index_age = f"{int(age_s / 3600)}h ago"
                else:
                    index_age = f"{int(age_s / 86400)}d ago"
            except Exception as e:
                logger.debug(f"Index age detection failed: {e}")

    # Memories
    mem_count = 0
    try:
        from know.knowledge_base import KnowledgeBase
        mem_count = KnowledgeBase(config).count()
    except Exception as e:
        logger.debug(f"Memory count retrieval failed: {e}")

    # Cache size
    cache_size = "0 B"
    total_bytes = 0
    know_dir = config.root / ".know"
    if know_dir.exists():
        for f in know_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
        if total_bytes > 1024 * 1024:
            cache_size = f"{total_bytes / 1024 / 1024:.1f} MB"
        elif total_bytes > 1024:
            cache_size = f"{total_bytes / 1024:.1f} KB"
        else:
            cache_size = f"{total_bytes} B"

    # Config check
    config_ok = (config.root / ".know" / "config.yaml").exists()

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({
            "version": __version__,
            "project": str(config.root),
            "files": n_files,
            "functions": n_functions,
            "index_age": index_age,
            "memories": mem_count,
            "cache_size": cache_size,
            "config_ok": config_ok,
        }, indent=2))
        return

    console.print(f"\n[green]✓[/green] [bold]know-cli v{__version__}[/bold]")
    console.print(f"  Project: {config.root}")
    # Build language breakdown
    lang_counts = {}
    try:
        for _, lang in scanner._discover_files():
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    except Exception:
        lang_counts = {"files": n_files}
    lang_str = ", ".join(f"{c} {l.title()}" for l, c in sorted(lang_counts.items(), key=lambda x: -x[1]))
    console.print(f"  Files: {lang_str or n_files}")
    console.print(f"  Functions: {n_functions}")
    console.print(f"  Indexed: {index_age}")
    console.print(f"  Memories: {mem_count}")
    console.print(f"  Cache: {cache_size}")
    if config_ok:
        console.print("  Config: .know/config.yaml [green]✓[/green]")
    else:
        console.print("  Config: [red]not initialized[/red]")
    console.print()
