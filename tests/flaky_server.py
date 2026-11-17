"""A stdio MCP server whose tools misbehave on purpose (used by test_robustness.py).

ping -> "pong"; die -> kills the server process; hang -> never returns.
"""

import os
import sys

import anyio

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

s = Server("flaky")


@s.tool()
def ping(text: str = "") -> str:
    """Reply with pong."""
    return f"pong {text}".strip()


@s.tool()
def die() -> str:
    """Kill the server process (simulates a crashed MCP server)."""
    sys.stderr.write("flaky: dying on purpose\n")
    sys.stderr.flush()
    os._exit(1)


@s.tool()
async def hang(seconds: float = 3600) -> str:
    """Sleep for a long time (simulates a hung tool)."""
    await anyio.sleep(seconds)
    return "woke up"


if __name__ == "__main__":
    s.run()
