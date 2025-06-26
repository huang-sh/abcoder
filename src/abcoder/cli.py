"""
Command-line interface for abcoder-mcp.

This module provides a CLI entry point for the abcoder-mcp package.
"""

import os
import typer
from enum import Enum

from .server import nb_mcp


app = typer.Typer(
    name="abcoder-mcp",
    help="Abcoder MCP Server CLI",
    add_completion=False,
    no_args_is_help=True,  # Show help if no args provided
)


class Transport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    SHTTP = "shttp"


@app.command(name="run")
def run(
    transport: Transport = typer.Option(
        Transport.STDIO,
        "-t",
        "--transport",
        help="Specify transport type",
        case_sensitive=False,
    ),
    port: int = typer.Option(8000, "-p", "--port", help="transport port"),
    host: str = typer.Option("127.0.0.1", "--host", help="transport host"),
):
    """Start Abcoder MCP Server"""
    # Set environment variables
    os.environ["ABCODER_TRANSPORT"] = transport.value
    os.environ["ABCODER_HOST"] = host
    os.environ["ABCODER_PORT"] = str(port)

    if transport == Transport.STDIO:
        nb_mcp.run()
    elif transport == Transport.SSE:
        nb_mcp.run(transport="sse", host=host, port=port, log_level="info")
    elif transport == Transport.SHTTP:
        nb_mcp.run(transport="streamable-http", host=host, port=port, log_level="info")


@app.callback()
def main():
    """Abcoder MCP CLI root command."""
    pass
