"""Connect to N MCP servers at once, namespace their tools, and route calls. Compatible with mcp SDK 1.x and 2.x."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ServerConfig

SEP = "__"
# JSON-RPC code the SDK uses when the transport closes mid-request (mcp.types.CONNECTION_CLOSED)
CONNECTION_CLOSED = -32000


@dataclass
class ToolSpec:
    qualified: str  # server__tool
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.qualified,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass
class CallOutcome:
    ok: bool
    text: str
    latency_ms: float
    error: str | None = None
    infra: bool = False  # the server, not the tool, failed (connection lost / server down)


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


async def _http_transport(stack: AsyncExitStack, cfg: ServerConfig) -> tuple[Any, Any]:
    # Symbols differ between mcp 1.x and 2.x, so look them up on Any-typed modules: this keeps
    # mypy happy against either SDK version without version-specific `type: ignore`s.
    assert cfg.url is not None  # guaranteed by ServerConfig validation for http transport
    sh: Any = importlib.import_module("mcp.client.streamable_http")
    if not hasattr(sh, "streamable_http_client"):  # mcp 1.x
        streams = await stack.enter_async_context(
            sh.streamablehttp_client(cfg.url, headers=cfg.headers or {})
        )
        return streams[0], streams[1]
    # mcp 2.x takes a pre-built HTTP client for headers; build it with the SDK's own helper, since the
    # underlying library changed (httpx -> httpx2) across 2.x releases
    make_http = getattr(sh, "create_mcp_http_client", None)
    if make_http is None:
        make_http = importlib.import_module("mcp.shared._httpx_utils").create_mcp_http_client
    http = await stack.enter_async_context(make_http(headers=cfg.headers or {}))
    streams = await stack.enter_async_context(sh.streamable_http_client(cfg.url, http_client=http))
    return streams[0], streams[1]


@contextmanager
def _redacting_errlog(path: str, redact: Callable[[str], str]) -> Iterator[TextIO]:
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
        writer.close()  # the server has exited by now, so this was the last write end: the pump sees EOF
        t.join(timeout=5)


def _is_connection_lost(e: BaseException) -> bool:
    """The session's transport is gone (server process died, stream closed), as opposed to a tool error."""
    code = _field(e, "code", default=None)
    if not isinstance(code, int):
        code = _field(_field(e, "error", default=None), "code", default=None)
    if code == CONNECTION_CLOSED:
        return True
    names = {c.__name__ for c in type(e).__mro__}
    if names & {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}:
        return True
    return isinstance(e, (ConnectionError, EOFError)) or "connection closed" in str(e).lower()


class _Conn:
    """One server's session, owned by a dedicated task so it can be torn down and restarted from any job
    (anyio cancel scopes must be exited by the task that entered them)."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.session: ClientSession | None = None
        self.task: asyncio.Task[None] | None = None
        self.stop = asyncio.Event()
        self.lock = asyncio.Lock()
        self.generation = 0  # bumped on every (re)start, so concurrent failures restart only once
        self.restarts = 0
        self.dead: str | None = None  # set once the server is down for good

    @property
    def alive(self) -> bool:
        return self.task is not None and not self.task.done()


class MCPPool:
    """Async context manager owning all server sessions for a run.

    Tool calls time out after the server's `call_timeout_s`. A server whose connection is lost is restarted
    (up to `max_restarts` times); the call that hit the dead server returns an `infra` outcome.
    """

    def __init__(
        self,
        servers: list[ServerConfig],
        base_dir: str | None = None,
        errlog_path: str | None = None,
        redact: Callable[[str], str] | None = None,
    ):
        self.servers = servers
        self.redact = redact
        self.base_dir = base_dir
        self.tools: dict[str, ToolSpec] = {}
        self._conns: dict[str, _Conn] = {}
        # server stderr (tracebacks from tools that raise) goes to a log file, not the console
        import tempfile

        self.errlog_path = errlog_path or tempfile.mktemp(
            prefix="toolscope-servers-", suffix=".log"
        )

    @property
    def sessions(self) -> dict[str, ClientSession]:
        return {n: c.session for n, c in self._conns.items() if c.session is not None}

    async def _connect(self, stack: AsyncExitStack, cfg: ServerConfig) -> ClientSession:
        if cfg.transport == "stdio":
            assert cfg.command is not None  # guaranteed by ServerConfig validation
            params = StdioServerParameters(
                command=cfg.command, args=cfg.args, env=cfg.env, cwd=self.base_dir
            )
            errlog = stack.enter_context(
                _redacting_errlog(self.errlog_path, self.redact)
                if self.redact
                else open(self.errlog_path, "a", encoding="utf-8")  # noqa: SIM115 - closed by the stack
            )
            read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        else:
            read, write = await _http_transport(stack, cfg)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _start(self, conn: _Conn) -> None:
        ready: asyncio.Future[ClientSession] = asyncio.get_running_loop().create_future()
        conn.stop = asyncio.Event()
        stop = conn.stop

        async def own() -> None:
            try:
                async with AsyncExitStack() as stack:
                    ready.set_result(await self._connect(stack, conn.cfg))
                    await stop.wait()
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # startup failure, or teardown of a dead server
                if not ready.done():
                    ready.set_exception(e)

        conn.task = asyncio.create_task(own(), name=f"toolscope-mcp-{conn.cfg.name}")
        conn.session = await ready
        conn.generation += 1

    async def _close(self, conn: _Conn, timeout: float = 10.0) -> None:
        conn.stop.set()
        task, conn.session = conn.task, None
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except BaseException:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    def _log(self, line: str) -> None:
        with contextlib.suppress(OSError), open(self.errlog_path, "a", encoding="utf-8") as f:
            f.write((self.redact or str)(f"[toolscope] {line}\n"))

    async def _restart(self, conn: _Conn, generation: int, reason: str) -> None:
        async with conn.lock:
            if conn.generation != generation or conn.dead:
                return  # another job already restarted (or gave up on) this server
            await self._close(conn)
            if conn.restarts >= conn.cfg.max_restarts:
                conn.dead = f"server '{conn.cfg.name}' is down ({reason})"
                self._log(f"server '{conn.cfg.name}' lost: {reason}; restart limit reached")
                return
            conn.restarts += 1
            self._log(f"server '{conn.cfg.name}' lost: {reason}; restarting ({conn.restarts})")
            try:
                await self._start(conn)
            except Exception as e:
                conn.dead = f"server '{conn.cfg.name}' failed to restart: {type(e).__name__}: {e}"
                self._log(conn.dead)

    async def __aenter__(self) -> MCPPool:
        try:
            for cfg in self.servers:
                conn = self._conns[cfg.name] = _Conn(cfg)
                await self._start(conn)
                assert conn.session is not None
                listed = await conn.session.list_tools()
                for t in listed.tools:
                    spec = ToolSpec(
                        qualified=f"{cfg.name}{SEP}{t.name}",
                        server=cfg.name,
                        name=t.name,
                        description=t.description or "",
                        input_schema=_field(t, "input_schema", "inputSchema", default={}) or {},
                    )
                    self.tools[spec.qualified] = spec
        except BaseException:
            await self.__aexit__()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for conn in self._conns.values():
            await self._close(conn)

    async def call(self, qualified: str, args: dict[str, Any]) -> CallOutcome:
        spec = self.tools.get(qualified)
        if spec is None:  # accept bare names when unambiguous
            hits = [s for s in self.tools.values() if s.name == qualified]
            spec = hits[0] if len(hits) == 1 else None
        if spec is None:
            return CallOutcome(False, f"Error: unknown tool '{qualified}'", 0.0, "unknown_tool")
        conn = self._conns[spec.server]
        if conn.dead or conn.session is None:
            why = conn.dead or f"server '{spec.server}' is not connected"
            return CallOutcome(False, f"Error: {why}", 0.0, f"infra: {why}", infra=True)
        session, generation, timeout = conn.session, conn.generation, conn.cfg.call_timeout_s
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(session.call_tool(spec.name, args), timeout)
            ms = (time.perf_counter() - t0) * 1000
            text = _content_text(res)
            is_err = bool(_field(res, "is_error", "isError", default=False))
            return CallOutcome(not is_err, text, ms, text[:300] if is_err else None)
        except asyncio.TimeoutError:
            ms = (time.perf_counter() - t0) * 1000
            if not conn.alive:  # the server died while we waited
                return await self._lost(conn, generation, "timed out on a dead server", ms)
            return CallOutcome(
                False, f"Error: tool call timed out after {timeout:g}s", ms, "timeout"
            )
        except (
            Exception
        ) as e:  # transport / validation errors surface to the model like real clients do
            ms = (time.perf_counter() - t0) * 1000
            if _is_connection_lost(e) or not conn.alive:
                return await self._lost(conn, generation, f"{type(e).__name__}: {e}", ms)
            return CallOutcome(False, f"Error: {type(e).__name__}: {e}", ms, str(e)[:300])

    async def _lost(self, conn: _Conn, generation: int, reason: str, ms: float) -> CallOutcome:
        await self._restart(conn, generation, reason)
        why = f"server '{conn.cfg.name}' connection lost: {reason}"
        return CallOutcome(False, f"Error: {why}", ms, f"infra: {why}"[:300], infra=True)
