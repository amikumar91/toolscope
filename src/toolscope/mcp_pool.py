"""Connect to N MCP servers at once, namespace their tools, and route calls. Compatible with mcp SDK 1.x and 2.x."""
from __future__ import annotations

import os
import threading
import time
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from typing import IO, Any, Callable, Iterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ServerConfig

SEP = "__"


@dataclass
class ToolSpec:
    qualified: str          # server__tool
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def openai(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.qualified,
            "description": self.description,
            "parameters": self.input_schema or {"type": "object", "properties": {}},
        }}


@dataclass
class CallOutcome:
    ok: bool
    text: str
    latency_ms: float
    error: str | None = None


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _content_text(result: Any) -> str:
    parts = []
    for c in _field(result, "content", default=[]) or []:
        t = _field(c, "text")
        parts.append(t if t is not None else str(c))
    structured = _field(result, "structured_content", "structuredContent")
    if not parts and structured is not None:
        parts.append(str(structured))
    return "\n".join(parts)


async def _http_transport(stack: AsyncExitStack, cfg: ServerConfig):
    try:  # mcp 2.x
        from mcp.client.streamable_http import streamable_http_client as client
    except ImportError:  # mcp 1.x
        from mcp.client.streamable_http import streamablehttp_client as client
        streams = await stack.enter_async_context(client(cfg.url, headers=cfg.headers or {}))
        return streams[0], streams[1]
    # mcp 2.x takes a pre-built HTTP client for headers; build it with the SDK's own helper, since the
    # underlying library changed (httpx -> httpx2) across 2.x releases
    try:
        from mcp.client.streamable_http import create_mcp_http_client
    except ImportError:
        from mcp.shared._httpx_utils import create_mcp_http_client
    http = await stack.enter_async_context(create_mcp_http_client(headers=cfg.headers or {}))
    streams = await stack.enter_async_context(client(cfg.url, http_client=http))
    return streams[0], streams[1]


@contextmanager
def _redacting_errlog(path: str, redact: Callable[[str], str]) -> Iterator[IO[str]]:
    """Yield a real file (the write end of a pipe) for a server's stderr; a thread copies each line to `path`
    after `redact`, so secrets a server prints never reach the log. The subprocess needs an OS-level fd,
    which is why a Python wrapper around the log file won't do."""
    rfd, wfd = os.pipe()
    reader = os.fdopen(rfd, "r", encoding="utf-8", errors="replace")
    writer = os.fdopen(wfd, "w", encoding="utf-8")

    def pump() -> None:
        with reader, open(path, "a", encoding="utf-8") as log:
            for line in reader:
                log.write(redact(line))
                log.flush()

    t = threading.Thread(target=pump, name="toolscope-errlog", daemon=True)
    t.start()
    try:
        yield writer
    finally:
        writer.close()      # the server has exited by now, so this was the last write end: the pump sees EOF
        t.join(timeout=5)


class MCPPool:
    """Async context manager owning all server sessions for a run."""

    def __init__(self, servers: list[ServerConfig], base_dir: str | None = None, errlog_path: str | None = None,
                 redact: Callable[[str], str] | None = None):
        self.servers = servers
        self.redact = redact
        self.base_dir = base_dir
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, ToolSpec] = {}
        self._stack = AsyncExitStack()
        # server stderr (tracebacks from tools that raise) goes to a log file, not the console
        import tempfile
        self.errlog_path = errlog_path or tempfile.mktemp(prefix="toolscope-servers-", suffix=".log")

    async def __aenter__(self) -> "MCPPool":
        for cfg in self.servers:
            if cfg.transport == "stdio":
                params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env,
                                               cwd=self.base_dir)
                errlog = self._stack.enter_context(
                    _redacting_errlog(self.errlog_path, self.redact) if self.redact else open(self.errlog_path, "a"))
                read, write = await self._stack.enter_async_context(stdio_client(params, errlog=errlog))
            else:
                read, write = await _http_transport(self._stack, cfg)
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[cfg.name] = session
            listed = await session.list_tools()
            for t in listed.tools:
                spec = ToolSpec(
                    qualified=f"{cfg.name}{SEP}{t.name}", server=cfg.name, name=t.name,
                    description=t.description or "",
                    input_schema=_field(t, "input_schema", "inputSchema", default={}) or {},
                )
                self.tools[spec.qualified] = spec
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def call(self, qualified: str, args: dict[str, Any]) -> CallOutcome:
        spec = self.tools.get(qualified)
        if spec is None:  # accept bare names when unambiguous
            hits = [s for s in self.tools.values() if s.name == qualified]
            spec = hits[0] if len(hits) == 1 else None
        if spec is None:
            return CallOutcome(False, f"Error: unknown tool '{qualified}'", 0.0, "unknown_tool")
        t0 = time.perf_counter()
        try:
            res = await self.sessions[spec.server].call_tool(spec.name, args)
            ms = (time.perf_counter() - t0) * 1000
            text = _content_text(res)
            is_err = bool(_field(res, "is_error", "isError", default=False))
            return CallOutcome(not is_err, text, ms, text[:300] if is_err else None)
        except Exception as e:  # transport / validation errors surface to the model like real clients do
            ms = (time.perf_counter() - t0) * 1000
            return CallOutcome(False, f"Error: {type(e).__name__}: {e}", ms, str(e)[:300])
