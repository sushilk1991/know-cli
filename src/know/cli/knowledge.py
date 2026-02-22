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
@click.option("--type", "memory_type", default="note", help="Memory type (note, decision, fact, constraint, todo, risk)")
@click.option("--status", "decision_status", default="active", help="Decision status (active, resolved, superseded, rejected)")
@click.option("--confidence", type=float, default=0.5, help="Confidence score [0-1]")
@click.option("--evidence", default="", help="Evidence pointer (path:line or note)")
@click.option("--session-id", default="", help="Session identifier")
@click.option("--agent", default="", help="Originating agent name")
@click.option("--trust-level", default="local_verified", help="Trust level (local_verified, imported_unverified, blocked)")
@click.option("--expires-in-hours", type=float, default=None, help="Optional expiry in hours from now")
@click.pass_context
def remember(
    ctx: click.Context,
    text: str,
    tags: str,
    source: str,
    memory_type: str,
    decision_status: str,
    confidence: float,
    evidence: str,
    session_id: str,
    agent: str,
    trust_level: str,
    expires_in_hours: Optional[float],
) -> None:
    """Store a memory for cross-session recall.

    Example: know remember "The auth system uses JWT with Redis session store"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase
    import time as _time

    kb = KnowledgeBase(config)
    expires_at = None
    if expires_in_hours is not None:
        expires_at = _time.time() + (expires_in_hours * 3600)
    mem_id = kb.remember(
        text,
        source=source,
        tags=tags,
        memory_type=memory_type,
        decision_status=decision_status,
        confidence=confidence,
        evidence=evidence,
        session_id=session_id,
        agent=agent,
        trust_level=trust_level,
        expires_at=expires_at,
    )

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
@click.argument("decision")
@click.option("--why", default="", help="Why this decision was made")
@click.option("--tags", "-t", default="decision", help="Comma-separated tags")
@click.option("--confidence", type=float, default=0.7, help="Confidence score [0-1]")
@click.option("--evidence", default="", help="Evidence pointer (path:line)")
@click.option("--session-id", default="", help="Session identifier")
@click.option("--agent", default="", help="Originating agent name")
@click.option("--status", "decision_status", default="active", help="Decision status")
@click.pass_context
def decide(
    ctx: click.Context,
    decision: str,
    why: str,
    tags: str,
    confidence: float,
    evidence: str,
    session_id: str,
    agent: str,
    decision_status: str,
) -> None:
    """Store a structured decision memory."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    text = decision.strip()
    if why.strip():
        text = f"{text} Why: {why.strip()}"

    kb = KnowledgeBase(config)
    mem_id = kb.remember(
        text=text,
        source="manual",
        tags=tags,
        memory_type="decision",
        decision_status=decision_status,
        confidence=confidence,
        evidence=evidence,
        session_id=session_id,
        agent=agent,
        trust_level="local_verified",
    )

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": mem_id, "decision": decision, "status": decision_status}))
    elif ctx.obj.get("quiet"):
        click.echo(str(mem_id))
    else:
        console.print(f"[green]✓[/green] Decision stored (id={mem_id})")


@click.command()
@click.argument("query")
@click.option("--top-k", "-k", type=int, default=5, help="Max results")
@click.option("--type", "memory_type", default=None, help="Filter by memory type")
@click.option("--status", "decision_status", default=None, help="Filter by decision status")
@click.option("--include-blocked", is_flag=True, help="Include blocked memories")
@click.option("--include-expired", is_flag=True, help="Include expired memories")
@click.pass_context
def recall(
    ctx: click.Context,
    query: str,
    top_k: int,
    memory_type: Optional[str],
    decision_status: Optional[str],
    include_blocked: bool,
    include_expired: bool,
) -> None:
    """Recall memories semantically similar to a query.

    Example: know recall "how does auth work?"
    """
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    import time as _time
    t0 = _time.monotonic()

    kb = KnowledgeBase(config)
    memories = kb.recall(
        query,
        top_k=top_k,
        memory_type=memory_type,
        decision_status=decision_status,
        include_blocked=include_blocked,
        include_expired=include_expired,
    )
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
@click.option("--type", "memory_type", default=None, help="Filter by memory type")
@click.option("--status", "decision_status", default=None, help="Filter by decision status")
@click.option("--session-id", default=None, help="Filter by session id")
@click.pass_context
def memories_list(
    ctx: click.Context,
    source: Optional[str],
    memory_type: Optional[str],
    decision_status: Optional[str],
    session_id: Optional[str],
) -> None:
    """List all stored memories."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    mems = kb.list_all(
        source=source,
        memory_type=memory_type,
        decision_status=decision_status,
        session_id=session_id,
    )

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
            console.print(
                f"  [cyan]#{m.id}[/cyan] [{m.source}] [{m.memory_type}/{m.decision_status}] {m.text}"
            )
            if m.tags:
                console.print(f"       tags: {m.tags}")
            console.print(f"       [dim]{m.created_at}[/dim]")
        console.print()


@memories.command(name="resolve")
@click.argument("memory_id", type=int)
@click.option("--status", default="resolved", help="resolved|superseded|rejected|active")
@click.pass_context
def memories_resolve(ctx: click.Context, memory_id: int, status: str) -> None:
    """Resolve/supersede/reject a memory."""
    config = ctx.obj["config"]
    from know.knowledge_base import KnowledgeBase

    kb = KnowledgeBase(config)
    ok = kb.resolve(memory_id, status=status)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"id": memory_id, "updated": ok, "status": status}))
    elif ctx.obj.get("quiet"):
        click.echo("1" if ok else "0")
    else:
        if ok:
            console.print(f"[green]✓[/green] Memory #{memory_id} -> {status}")
        else:
            console.print(f"[red]✗[/red] Memory #{memory_id} not found")


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
