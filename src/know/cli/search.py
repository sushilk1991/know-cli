"""Search commands: search, grep, context, graph, reindex."""

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel

from know.cli import console, logger
from know.cli.usage import attach_usage, build_usage_payload, render_usage
from know.scanner import CodebaseScanner

_DEFAULT_GREP_GLOBS = (
    "*.py",
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.go",
    "*.rs",
    "*.java",
    "*.rb",
    "*.php",
    "*.swift",
    "*.kt",
)


def _estimate_tokens_from_search_results(results: list[dict]) -> int:
    """Estimate tokens represented by returned search results."""
    if not results:
        return 0

    from know.token_counter import count_tokens

    total = 0
    for row in results:
        row_tokens = int(row.get("token_count", 0) or 0)
        if row_tokens > 0:
            total += row_tokens
            continue

        preview = str(row.get("preview") or row.get("signature") or "")
        if preview:
            total += count_tokens(preview)
    return total


def _extract_grep_terms(query: str, max_terms: int) -> list[str]:
    """Extract compact search terms from a natural-language query."""
    terms: list[str] = []
    try:
        from know.query import analyze_query

        plan = analyze_query(query)
        for token in plan.all_search_terms:
            token = token.strip()
            if len(token) < 2:
                continue
            if token not in terms:
                terms.append(token)
            if len(terms) >= max_terms:
                break
    except Exception:
        pass

    if not terms:
        for token in query.strip().split():
            token = token.strip()
            if len(token) < 2:
                continue
            if token not in terms:
                terms.append(token)
            if len(terms) >= max_terms:
                break

    return terms or [query.strip()]


def _extract_path_from_rg_event(event: dict) -> str:
    """Extract file path from an `rg --json` event payload."""
    if not isinstance(event, dict):
        return ""
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    path_obj = data.get("path")
    if not isinstance(path_obj, dict):
        return ""
    text = path_obj.get("text")
    return text.strip() if isinstance(text, str) else ""


def _format_context_payload_markdown(payload: dict) -> str:
    """Render daemon/agent JSON context payload as markdown."""
    lines = [
        f'# Context for: "{payload.get("query", "")}"',
        f'## Token Budget: {payload.get("budget_utilization", "")}',
        "",
    ]

    for warning in payload.get("warnings", []) or []:
        lines.append(f"> {warning}")
    if payload.get("warnings"):
        lines.append("")

    if payload.get("code"):
        lines.append("### Relevant Code")
        lines.append("")
        for chunk in payload["code"]:
            line_start = (chunk.get("lines") or [0])[0]
            lines.append(f"#### {chunk.get('file', '')}:{line_start}::{chunk.get('name', '')}")
            lines.append(f"```python\n{chunk.get('body', '')}\n```")
            lines.append("")

    if payload.get("dependencies"):
        lines.append("### Dependencies")
        lines.append("")
        for chunk in payload["dependencies"]:
            lines.append(f"#### {chunk.get('file', '')} (signatures only)")
            lines.append(f"```python\n{chunk.get('body', '')}\n```")
            lines.append("")

    if payload.get("tests"):
        lines.append("### Related Tests")
        lines.append("")
        for chunk in payload["tests"]:
            line_start = (chunk.get("lines") or [0])[0]
            lines.append(f"#### {chunk.get('file', '')}:{line_start}::{chunk.get('name', '')}")
            lines.append(f"```python\n{chunk.get('body', '')}\n```")
            lines.append("")

    if payload.get("summaries"):
        lines.append("### File Summaries")
        lines.append("")
        for chunk in payload["summaries"]:
            lines.append(chunk.get("body", ""))
            lines.append("")

    memories = payload.get("memories")
    if memories:
        lines.append("### Memories (Cross-Session Knowledge)")
        lines.append("")
        lines.append(memories)
        lines.append("")

    overview = payload.get("overview")
    if overview:
        lines.append("### Project Context")
        lines.append("")
        lines.append(overview)
        lines.append("")

    return "\n".join(lines)


@click.command()
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=10,
    help="Number of results to show"
)
@click.option(
    "--index",
    is_flag=True,
    help="Index the codebase before searching"
)
@click.option(
    "--chunk",
    is_flag=True,
    help="Search at function/class level (chunk embeddings)"
)
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, index: bool, chunk: bool) -> None:
    """Search code semantically using embeddings.

    Falls back to BM25 keyword search if fastembed/numpy not installed.
    """
    config = ctx.obj["config"]

    try:
        from know.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(project_root=config.root)
        _search_semantic(ctx, config, searcher, query, top_k, index, chunk)
    except ImportError as e:
        _search_bm25_fallback(ctx, config, query, top_k, cause=e)


def _search_semantic(ctx, config, searcher, query, top_k, index, chunk):
    """Semantic search using fastembed embeddings."""
    is_json = ctx.obj.get("json")
    if index:
        if not ctx.obj.get("quiet") and not is_json:
            console.print(f"[dim]Indexing {config.root}...[/dim]")
        if chunk:
            count = searcher.index_chunks(config.root)
        else:
            count = searcher.index_directory(config.root)
        if not ctx.obj.get("quiet") and not is_json:
            console.print(f"[green]✓[/green] Indexed {count} {'chunks' if chunk else 'files'}")

    if not ctx.obj.get("quiet") and not is_json:
        console.print(f"[dim]Searching for: {query}[/dim]")

    import time as _time
    t0 = _time.monotonic()

    if chunk:
        results = searcher.search_chunks(query, config.root, top_k, auto_index=not index)
    else:
        results = searcher.search_code(query, config.root, top_k, auto_index=not index)

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_search(query, len(results), duration_ms)
    except Exception as e:
        logger.debug(f"Stats tracking (search) failed: {e}")

    usage = build_usage_payload(
        source="semantic_search",
        tokens_used=_estimate_tokens_from_search_results(results),
        elapsed_ms=duration_ms,
        details={"results": len(results)},
    )

    if is_json:
        payload = {"results": results}
        click.echo(json.dumps(attach_usage(payload, usage)))
    elif ctx.obj.get("quiet"):
        for r in results:
            click.echo(f"{r['score']:.3f} {r.get('path', r.get('name', ''))}")
    else:
        if not results:
            console.print("[yellow]No results found[/yellow]")
            render_usage(ctx, usage)
            sys.exit(2)

        console.print(f"\n[bold]Top {len(results)} results:[/bold]\n")
        for i, r in enumerate(results, 1):
            score_color = "green" if r['score'] > 0.7 else "yellow" if r['score'] > 0.4 else "dim"
            label = r.get("path", r.get("name", ""))
            if chunk and r.get("name"):
                label = f"{r['path']}:{r['name']}" if r.get("path") else r["name"]
            console.print(f"{i}. [{score_color}]{r['score']:.3f}[/{score_color}] {label}")
            if r.get('preview'):
                preview = r['preview'][:200].replace('\n', ' ')
                console.print(f"   [dim]{preview}...[/dim]")
            console.print()
        render_usage(ctx, usage)


def _embedding_runtime_diagnostics() -> dict:
    """Inspect local environment for embedding dependency integrity."""
    diag = {
        "fastembed_installed": bool(importlib.util.find_spec("fastembed")),
        "onnxruntime_installed": bool(importlib.util.find_spec("onnxruntime")),
        "distribution_version": None,
        "module_version": None,
        "editable_install": False,
        "version_mismatch": False,
    }

    try:
        dist = importlib.metadata.distribution("know-cli")
        diag["distribution_version"] = dist.version
        direct_url = dist.read_text("direct_url.json")
        if direct_url and '"editable": true' in direct_url:
            diag["editable_install"] = True
    except Exception:
        pass

    try:
        import know

        diag["module_version"] = getattr(know, "__version__", None)
    except Exception:
        pass

    if diag["distribution_version"] and diag["module_version"]:
        diag["version_mismatch"] = diag["distribution_version"] != diag["module_version"]

    return diag


def _search_bm25_fallback(ctx, config, query, top_k, cause: Optional[Exception] = None):
    """BM25 keyword search fallback when fastembed/numpy not installed."""
    from know.cli.agent import _get_daemon_client, _get_db_fallback

    diag = _embedding_runtime_diagnostics()

    if not ctx.obj.get("quiet") and not ctx.obj.get("json"):
        detail = "semantic embeddings unavailable"
        if not diag["fastembed_installed"] or not diag["onnxruntime_installed"]:
            detail = "embedding runtime missing"
        elif cause:
            detail = "embedding runtime unavailable"
        console.print(f"[dim]Using BM25 keyword search ({detail})[/dim]")

    import time as _time
    t0 = _time.monotonic()

    client = _get_daemon_client(config)
    if client:
        try:
            result = client.call_sync("search", {"query": query, "limit": top_k})
            results = result.get("results", [])
        except Exception:
            client = None

    if not client:
        db = _get_db_fallback(config)
        results = db.search_chunks(query, top_k)

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_search(query, len(results), duration_ms)
    except Exception as e:
        logger.debug(f"Stats tracking (search) failed: {e}")

    usage = build_usage_payload(
        source="bm25_search",
        tokens_used=_estimate_tokens_from_search_results(results),
        elapsed_ms=duration_ms,
        details={"results": len(results)},
    )

    if ctx.obj.get("json"):
        payload = {"results": results}
        click.echo(json.dumps(attach_usage(payload, usage)))
    elif ctx.obj.get("quiet"):
        for r in results:
            click.echo(f"{r.get('file_path', '')}:{r.get('chunk_name', '')}")
    else:
        if not results:
            console.print("[yellow]No results found[/yellow]")
            if not diag["fastembed_installed"] or not diag["onnxruntime_installed"]:
                console.print("[dim]Tip: python -m pip install -U know-cli[/dim]")
            if diag["editable_install"] or diag["version_mismatch"]:
                console.print("[dim]Tip: editable install detected; re-run python -m pip install -e .[/dim]")
            console.print("[dim]Tip: know doctor --repair --reindex[/dim]")
            render_usage(ctx, usage)
            sys.exit(2)

        console.print(f"\n[bold]Top {len(results)} results (BM25):[/bold]\n")
        for i, r in enumerate(results, 1):
            label = f"{r.get('file_path', '')}:{r.get('chunk_name', '')}"
            console.print(f"{i}. {label}")
            if r.get('signature'):
                console.print(f"   [dim]{r['signature'][:200]}[/dim]")
            console.print()

        if not diag["fastembed_installed"] or not diag["onnxruntime_installed"]:
            console.print("[dim]Tip: python -m pip install -U know-cli[/dim]")
        if diag["editable_install"] or diag["version_mismatch"]:
            console.print("[dim]Tip: editable install detected; re-run python -m pip install -e .[/dim]")
        console.print("[dim]Tip: know doctor --repair --reindex[/dim]")
        render_usage(ctx, usage)


@click.command("grep")
@click.argument("query")
@click.option(
    "--max-files",
    type=click.IntRange(min=1),
    default=12,
    show_default=True,
    help="Maximum matched files to read for token accounting.",
)
@click.option(
    "--max-terms",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
    help="Maximum query terms to probe with ripgrep.",
)
@click.option(
    "--ignore-case/--case-sensitive",
    default=True,
    show_default=True,
    help="Case sensitivity for ripgrep term matching.",
)
@click.option(
    "--include",
    "include_globs",
    multiple=True,
    help="Additional ripgrep include globs (defaults to common code extensions).",
)
@click.pass_context
def grep_cmd(
    ctx: click.Context,
    query: str,
    max_files: int,
    max_terms: int,
    ignore_case: bool,
    include_globs: tuple[str, ...],
) -> None:
    """Run a grep+read baseline and report token/time usage."""
    if shutil.which("rg") is None:
        raise click.ClickException("ripgrep (rg) is required for `know grep`.")

    config = ctx.obj["config"]
    import time as _time
    from know.token_counter import count_tokens

    t0 = _time.monotonic()
    terms = _extract_grep_terms(query, max_terms=max_terms)
    globs = list(_DEFAULT_GREP_GLOBS)
    for glob in include_globs:
        if glob not in globs:
            globs.append(glob)

    match_counts: dict[str, int] = {}
    for term in terms:
        cmd = [
            "rg",
            "--json",
            "--line-number",
            "--no-heading",
            "--color=never",
            "--fixed-strings",
        ]
        if ignore_case:
            cmd.append("--ignore-case")
        for glob in globs:
            cmd.extend(["-g", glob])
        cmd.extend(["--", term, "."])

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(config.root),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise click.ClickException(
                f"ripgrep timed out while searching term '{term}' (timeout={exc.timeout}s).",
            ) from exc

        if proc.returncode not in (0, 1):
            err = (proc.stderr or "").strip() or "unknown ripgrep error"
            raise click.ClickException(f"ripgrep failed for term '{term}': {err}")

        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            path = _extract_path_from_rg_event(event)
            if path:
                match_counts[path] = match_counts.get(path, 0) + 1

    ranked = sorted(match_counts.items(), key=lambda item: item[1], reverse=True)
    top_paths = ranked[:max_files]

    total_tokens = 0
    rows: list[dict] = []
    for rel_path, hits in top_paths:
        file_path = config.root / rel_path
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        file_tokens = count_tokens(content, provider="anthropic")
        total_tokens += file_tokens
        rows.append(
            {
                "file_path": rel_path,
                "matches": hits,
                "tokens": file_tokens,
            },
        )

    duration_ms = int((_time.monotonic() - t0) * 1000)
    usage = build_usage_payload(
        source="grep_read",
        tokens_used=total_tokens,
        elapsed_ms=duration_ms,
        details={
            "files_matched": len(match_counts),
            "files_read": len(rows),
            "terms": ",".join(terms),
        },
    )

    # Track in generic search stats for trend visibility.
    try:
        from know.stats import StatsTracker

        StatsTracker(config).record_search(query, len(rows), duration_ms)
    except Exception as e:
        logger.debug(f"Stats tracking (grep) failed: {e}")

    payload = {
        "query": query,
        "strategy": "grep_read",
        "terms": terms,
        "files_matched": len(match_counts),
        "files_read": len(rows),
        "results": rows,
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(attach_usage(payload, usage)))
        return

    if ctx.obj.get("quiet"):
        for row in rows:
            click.echo(row["file_path"])
        return

    if not rows:
        console.print("[yellow]No matches found[/yellow]")
        render_usage(ctx, usage)
        sys.exit(2)

    console.print(f"\n[bold]Top {len(rows)} files (grep+read):[/bold]\n")
    for idx, row in enumerate(rows, 1):
        console.print(
            f"{idx}. [green]{row['file_path']}[/green] "
            f"[dim](matches={row['matches']}, tokens={row['tokens']:,})[/dim]",
        )
    render_usage(ctx, usage)


@click.command()
@click.argument("query", required=False, default=None)
@click.option(
    "--budget",
    "-b",
    type=int,
    default=8000,
    help="Token budget (default 8000)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "agent"]),
    default="markdown",
    help="Output format (markdown or agent JSON)",
)
@click.option(
    "--no-tests",
    is_flag=True,
    help="Skip test file inclusion",
    hidden=True,
)
@click.option(
    "--no-imports",
    is_flag=True,
    help="Skip import expansion",
    hidden=True,
)
@click.option(
    "--include",
    "include_patterns",
    multiple=True,
    help="Include only files matching glob pattern (e.g., 'src/**')",
    hidden=True,
)
@click.option(
    "--exclude",
    "exclude_patterns",
    multiple=True,
    help="Exclude files matching glob pattern (e.g., 'tests/**')",
    hidden=True,
)
@click.option(
    "--chunk-types",
    type=str,
    default=None,
    help="Comma-separated chunk types to include (function,class,method,module)",
    hidden=True,
)
@click.option(
    "--legacy",
    is_flag=True,
    help="Use legacy filesystem scan instead of DaemonDB (debugging only)",
    hidden=True,
)
@click.option(
    "--session",
    "session_id",
    default=None,
    help="Session ID for cross-query dedup ('auto' creates new, 'new' forces fresh)",
)
@click.pass_context
def context(
    ctx: click.Context,
    query: Optional[str],
    budget: int,
    output_format: str,
    no_tests: bool,
    no_imports: bool,
    include_patterns: tuple,
    exclude_patterns: tuple,
    chunk_types: Optional[str],
    legacy: bool,
    session_id: Optional[str],
) -> None:
    """Build LLM-optimized context for a query.

    Supports STDIN: echo "query" | know context --budget 4000

    Example: know context "help me fix the auth bug" --budget 8000

    Filtering:
      --include "src/**" --exclude "tests/**"
      --chunk-types function,class
    """
    config = ctx.obj["config"]

    # STDIN support: read query from pipe if not provided as argument
    if query is None:
        if not sys.stdin.isatty():
            query = sys.stdin.read().strip()
        if not query:
            click.echo("Error: query is required (pass as argument or via STDIN)", err=True)
            sys.exit(1)

    if not ctx.obj.get("quiet") and not ctx.obj.get("json") and output_format != "agent":
        console.print(f'[dim]Building context for: "{query}" (budget {budget} tokens)[/dim]')

    import time as _time
    t0 = _time.monotonic()

    from know.context_engine import ContextEngine
    from know.cli.agent import _get_daemon_client

    # Parse chunk types
    parsed_chunk_types = None
    if chunk_types:
        parsed_chunk_types = [t.strip() for t in chunk_types.split(",")]

    # Handle session ID
    resolved_session_id = None
    if session_id:
        if session_id in ("auto", "new"):
            import uuid
            resolved_session_id = uuid.uuid4().hex[:8]
        else:
            resolved_session_id = session_id
        if resolved_session_id:
            try:
                from know.runtime_context import set_active_session_id

                set_active_session_id(config, resolved_session_id)
            except Exception as e:
                logger.debug(f"Failed to persist active session id: {e}")

    engine = ContextEngine(config)
    include_markdown = not (ctx.obj.get("json") or output_format == "agent")
    payload = None

    # Daemon-first path for lower latency and single-process retrieval.
    if not legacy:
        client = _get_daemon_client(config)
        if client:
            try:
                payload = client.call_sync("context", {
                    "query": query,
                    "budget": budget,
                    "include_tests": not no_tests,
                    "include_imports": not no_imports,
                    "include_patterns": list(include_patterns) if include_patterns else None,
                    "exclude_patterns": list(exclude_patterns) if exclude_patterns else None,
                    "chunk_types": parsed_chunk_types,
                    "session_id": resolved_session_id,
                    "include_markdown": include_markdown,
                })
            except Exception as e:
                logger.debug(f"Daemon context failed, falling back: {e}")

    # Local fallback retains existing behavior.
    if payload is None:
        result = engine.build_context(
            query,
            budget=budget,
            include_tests=not no_tests,
            include_imports=not no_imports,
            legacy=legacy,
            include_patterns=list(include_patterns) if include_patterns else None,
            exclude_patterns=list(exclude_patterns) if exclude_patterns else None,
            chunk_types=parsed_chunk_types,
            session_id=resolved_session_id,
        )

        # Inject relevant memories into context
        try:
            from know.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(config)
            memory_ctx = kb.get_relevant_context(query, max_tokens=min(500, budget // 10))
            if memory_ctx:
                result["memories_context"] = memory_ctx
        except Exception as e:
            logger.debug(f"Memory injection into context failed: {e}")

        payload = json.loads(engine.format_agent_json(result))
        if include_markdown:
            payload["markdown"] = engine.format_markdown(result)

    duration_ms = int((_time.monotonic() - t0) * 1000)

    # Track stats
    try:
        from know.stats import StatsTracker
        StatsTracker(config).record_context(
            query, budget, int(payload.get("used_tokens", 0)), duration_ms,
        )
    except Exception as e:
        logger.debug(f"Stats tracking (context) failed: {e}")

    usage = build_usage_payload(
        source="context",
        tokens_used=int(payload.get("used_tokens", 0) or 0),
        elapsed_ms=duration_ms,
        details={"budget": budget},
    )

    if ctx.obj.get("json") or output_format == "agent":
        click.echo(json.dumps(attach_usage(payload, usage)))
    elif ctx.obj.get("quiet"):
        click.echo(payload.get("markdown") or _format_context_payload_markdown(payload))
    else:
        md = payload.get("markdown") or _format_context_payload_markdown(payload)
        from rich.markup import escape
        console.print(Panel(
            escape(md),
            title=f"🧠 Context ({payload.get('budget_utilization', '')})",
            border_style="blue",
        ))
        render_usage(ctx, usage)


@click.command()
@click.argument("file_path")
@click.pass_context
def graph(ctx: click.Context, file_path: str) -> None:
    """Show import graph for a file.

    Example: know graph src/know/ai.py
    """
    config = ctx.obj["config"]

    from know.import_graph import ImportGraph

    # Ensure graph is built
    scanner = CodebaseScanner(config)
    structure = scanner.get_structure()
    ig = ImportGraph(config)
    ig.build(structure["modules"])

    # Resolve the module name from the file path
    rel = str(Path(file_path).with_suffix("")).replace("/", ".").replace("\\", ".")

    output = ig.format_graph(rel)

    if ctx.obj.get("json"):
        import json
        data = {
            "module": rel,
            "imports": ig.imports_of(rel),
            "imported_by": ig.imported_by(rel),
        }
        click.echo(json.dumps(data, indent=2))
    elif ctx.obj.get("quiet"):
        click.echo(output)
    else:
        console.print(Panel(
            output,
            title=f"📊 Import Graph: {file_path}",
            border_style="green",
        ))


@click.command()
@click.option("--chunks", is_flag=True, help="Index at function/class level (default)")
@click.option("--files", "file_level", is_flag=True, help="Index at file level (legacy)")
@click.pass_context
def reindex(ctx: click.Context, chunks: bool, file_level: bool) -> None:
    """Rebuild search embeddings from scratch.

    By default indexes at function/class level for precise search.
    """
    config = ctx.obj["config"]

    from know.semantic_search import SemanticSearcher

    if not ctx.obj.get("quiet"):
        console.print("[dim]Clearing existing embeddings...[/dim]")

    searcher = SemanticSearcher(project_root=config.root)
    searcher.clear_cache()

    if file_level and not chunks:
        if not ctx.obj.get("quiet"):
            console.print("[dim]Indexing at file level...[/dim]")
        count = searcher.index_directory(config.root)
    else:
        if not ctx.obj.get("quiet"):
            console.print("[dim]Indexing at function/class level...[/dim]")
        count = searcher.index_chunks(config.root)

    if ctx.obj.get("json"):
        import json
        click.echo(json.dumps({"indexed": count}))
    elif ctx.obj.get("quiet"):
        click.echo(count)
    else:
        console.print(f"[green]✓[/green] Indexed [bold]{count}[/bold] chunks")
