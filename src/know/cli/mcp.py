"""MCP commands: mcp group with serve/config."""

import sys
from pathlib import Path

import click

from know.cli import console


@click.group()
def mcp() -> None:
    """MCP (Model Context Protocol) server for AI agents."""
    pass


@mcp.command(name="serve")
@click.option("--sse", is_flag=True, help="Use SSE transport instead of stdio")
@click.option("--port", type=int, default=3000, help="Port for SSE transport (default 3000)")
@click.pass_context
def mcp_serve(ctx: click.Context, sse: bool, port: int) -> None:
    """Start the MCP server.

    Default: stdio transport (for Claude Desktop).
    Use --sse for web clients.

    \b
    Examples:
      know mcp serve                    # stdio transport
      know mcp serve --sse --port 3000  # SSE transport
    """
    try:
        from know.mcp_server import run_server
    except ImportError:
        click.echo(
            "Error: The 'mcp' package is required.\n\n"
            "  pip install mcp\n\n"
            "Or install know-cli with:\n\n"
            "  pip install know-cli[mcp]\n",
            err=True,
        )
        sys.exit(1)

    config = ctx.obj["config"]
    run_server(sse=sse, port=port, project_root=config.root)


@mcp.command(name="config")
@click.pass_context
def mcp_config(ctx: click.Context) -> None:
    """Print Claude Desktop configuration snippet.

    Copy the output into your Claude Desktop config file.
    """
    try:
        from know.mcp_server import print_config
    except ImportError:
        click.echo(
            "Error: The 'mcp' package is required.\n\n"
            "  pip install mcp\n",
            err=True,
        )
        sys.exit(1)

    config = ctx.obj["config"]
    print_config(project_root=config.root)
