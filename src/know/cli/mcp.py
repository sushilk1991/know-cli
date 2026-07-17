"""MCP commands: mcp group with serve/config."""

import click


def _mcp_dependency_failure(error: ImportError) -> click.ClickException:
    """Translate lazy optional-dependency failures into an actionable CLI error."""
    detail = str(error).strip()
    message = detail or "The 'mcp' package is required for the MCP server."
    if "know-cli[mcp]" not in message:
        message += "\n\nInstall it with:\n\n  pip install know-cli[mcp]"
    return click.ClickException(message)


@click.group()
def mcp() -> None:
    """MCP (Model Context Protocol) server for AI agents."""
    pass


@mcp.command(name="serve")
@click.option("--sse", is_flag=True, help="Use SSE transport instead of stdio")
@click.option(
    "--port",
    type=click.IntRange(min=1, max=65535),
    default=3000,
    help="Port for SSE transport (default 3000)",
)
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
    from know import mcp_server

    try:
        config = ctx.obj["config"]
        mcp_server.run_server(sse=sse, port=port, project_root=config.root)
    except ImportError as error:
        if not mcp_server._MCP_AVAILABLE:
            raise _mcp_dependency_failure(error) from error
        raise


@mcp.command(name="config")
@click.pass_context
def mcp_config(ctx: click.Context) -> None:
    """Print Claude Desktop configuration snippet.

    Copy the output into your Claude Desktop config file.
    """
    from know.mcp_server import print_config

    config = ctx.obj["config"]
    print_config(project_root=config.root)
