"""Serve the demo `shop` tools over streamable HTTP on 127.0.0.1:<port>/mcp (used by test_http.py).

    python http_server.py <port>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "demo"))
import servers  # noqa: E402

port = int(sys.argv[1])
s = servers.shop()
try:  # mcp 2.x: MCPServer.run forwards host/port to run_streamable_http_async
    from mcp.server.mcpserver import MCPServer  # noqa: F401
    s.run("streamable-http", host="127.0.0.1", port=port)
except ImportError:  # mcp 1.x: FastMCP reads them from settings
    s.settings.host, s.settings.port = "127.0.0.1", port
    s.run("streamable-http")
