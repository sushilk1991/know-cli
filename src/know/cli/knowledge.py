"""Knowledge commands: remember, recall, forget, memories group."""

import sys
from pathlib import Path
from typing import Optional

import click

from know.cli import console, logger


@click.command()
@click.argument("text")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--source", "-s", default="manual", help="Memory source (manual, auto-explain, auto-digest)")
@click.pass_context
def remember(ctx: click.Context, text: str, tags: str, source: str) -> None:
    """Store a memory for cross-session recall.

    Example: know remember "The auth system uses JWT with Redis session store"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    mem_id = kb.remember(text, source=source, tags=tags)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_remember(text, source)
    except Exception as e:
        logger.debug(f"Stats tracking (remember) failed: {e}")

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": mem_id, "text": text, "source": source}))
    elif ctx.obj.get("quiet"):
        click.echo(str(mem_id))
    else:
        console.print(f"[green]✓[/green] Remembered (id={mem_id}): {text[:80]}")


@click.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=5, help="Max results")
@click.pass_context
def recall(ctx: click.Context, query: str, top_k: int) -> None:
    """Recall memories semantically similar to a query.

    Example: know recall "how does auth work?"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    import time as _time
    t0 = _time.monotonic()

    kb = KnowledgeBase(config)
    memories = kb.recall(query, top_k=top_k)
    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_recall(query, len(memories), duration_ms)
    except Exception as e:
        logger.debug(f"Stats tracking (recall) failed: {e}")

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"query": query, "results": [m.to_dict() for m in memories]}))
    elif ctx.obj.get("quiet"):
        for m in memories:
            click.echo(f"{m.id}\t{m.text}")
    else:
        if not memories:
            console.print("[yellow]No matching memories found[/yellow]")
            sys.exit(2)
        console.print(f"\n[bold]Recalled {len(memories)} memories:[/bold]\n")
        for m in memories:
            console.print(f"  [cyan]#{m.id}[/cyan] [{m.source}] {m.text}")
            console.print(f"       [dim]{m.created_at}[/dim]")
        console.print()


@click.command()
@click.argument("memory_id", type=int)
@click.pass_context
def forget(ctx: click.Context, memory_id: int) -> None:
    """Delete a memory by ID.

    Example: know forget 3
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    deleted = kb.forget(memory_id)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": memory_id, "deleted": deleted}))
    elif ctx.obj.get("quiet"):
        click.echo("1" if deleted else "0")
    else:
        if deleted:
            console.print(f"[green]✓[/green] Forgot memory #{memory_id}")
        else:
            console.print(f"[red]✗[/red] Memory #{memory_id} not found")
            sys.exit(1)


@click.group()
def memories() -> None:
    """Manage stored memories (list, export, import)."""
    pass


@memories.command(name="list")
@click.option("--source", "-s", default=None, help="Filter by source")
@click.pass_context
def memories_list(ctx: click.Context, source: Optional[str]) -> None:
    """List all stored memories."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    mems = kb.list_all(source=source)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps([m.to_dict() for m in mems]))
    elif ctx.obj.get("quiet"):
        for m in mems:
            click.echo(f"{m.id}\t{m.source}\t{m.text}")
    else:
        if not mems:
            console.print("[yellow]No memories stored yet[/yellow]")
            sys.exit(2)
        console.print(f"\n[bold]📝 {len(mems)} memories:[/bold]\n")
        for m in mems:
            console.print(f"  [cyan]#{m.id}[/cyan] [{m.source}] {m.text}")
            if m.tags:
                console.print(f"       tags: {m.tags}")
            console.print(f"       [dim]{m.created_at}[/dim]")
        console.print()


@memories.command(name="export")
@click.pass_context
def memories_export(ctx: click.Context) -> None:
    """Export all memories as JSON (to stdout)."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    click.echo(kb.export_json())


@memories.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.pass_context
def memories_import(ctx: click.Context, file: str) -> None:
    """Import memories from a JSON file."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    data = Path(file).read_text()
    kb = KnowledgeBase(config)
    count = kb.import_json(data)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"imported": count}))
    elif ctx.obj.get("quiet"):
        click.echo(str(count))
    else:
        console.print(f"[green]✓[/green] Imported {count} memories")
