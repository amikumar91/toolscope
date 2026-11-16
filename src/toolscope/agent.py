"""Two ways to drive tools under test.

1. Built-in loop (`run_builtin`): a minimal, provider-neutral tool-calling loop. Use it to compare *models*
   and *exposure modes* on equal footing.

2. Your agent (`kind: python`, `ref: pkg.module:fn`): toolscope hands your callable an AgentContext and
   records every call routed through it. Use it to evaluate *your real agent* (pydantic-ai, LangChain,
   LangGraph, OpenAI Agents SDK, hand-rolled) without rewriting it.

       async def my_agent(prompt: str, ctx: toolscope.AgentContext) -> str | list[dict]:
           # Option A: route tool execution through ctx.call_tool(...)  -> calls are recorded live
           # Option B: return the OpenAI-format message history          -> calls are parsed from it
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .config import AgentConfig, Task
from .llm import LLM
from .mcp_pool import MCPPool, ToolSpec
from .search import SEARCH_TOOL_SPEC, BM25Retriever, Retriever
from .tokens import count
from .trace import SEARCH_TOOL, SearchEvent, ToolCall, Trace


def _search_result_text(specs: list[ToolSpec]) -> str:
    if not specs:
        return "Found tools: (none matched; try different words)"
    lines = [f"- {s.qualified}: {s.description[:160]}" for s in specs]
    return "Found tools (now callable):\n" + "\n".join(lines)


async def run_builtin(
    pool: MCPPool,
    task: Task,
    cfg: AgentConfig,
    llm: LLM,
    mode: str,
    repeat: int,
    top_k: int = 5,
    retriever: Retriever | None = None,
) -> Trace:
    trace = Trace(
        task_id=task.id, agent_id=cfg.id, mode=mode, repeat=repeat, exposed_tools=list(pool.tools)
    )
    specs = pool.tools
    retriever = retriever or BM25Retriever(list(specs.values()))
    loaded: dict[str, ToolSpec] = dict(specs) if mode == "all" else {}

    def visible() -> list[dict]:
        base = [s.openai() for s in loaded.values()]
        return base if mode == "all" else [SEARCH_TOOL_SPEC] + base

    messages: list[dict[str, Any]] = []
    if task.system:
        messages.append({"role": "system", "content": task.system})
    messages.append({"role": "user", "content": task.prompt})

    t0 = time.perf_counter()
    try:
        for turn in range(1, cfg.max_turns + 1):
            tools = visible()
            trace.usage.schema_tokens += count(tools)
            resp = await llm.complete(messages, tools, seed=repeat)
            trace.turns = turn
            trace.usage.input_tokens += resp.input_tokens
            trace.usage.output_tokens += resp.output_tokens
            if resp.cost_usd is not None:
                trace.usage.cost_usd = (trace.usage.cost_usd or 0) + resp.cost_usd
            if not resp.tool_calls:
                trace.final_answer = resp.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in resp.tool_calls
                    ],
                }
            )
            for c in resp.tool_calls:
                try:
                    args = json.loads(c.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be an object")
                except Exception as e:
                    trace.calls.append(
                        ToolCall(
                            name=c.name,
                            args={"__unparseable__": c.arguments},
                            ok=False,
                            error=f"bad_json: {e}",
                            turn=turn,
                            known=c.name in specs,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": c.id,
                            "content": f"Error: invalid JSON arguments: {e}",
                        }
                    )
                    continue

                if c.name == SEARCH_TOOL and mode == "search":
                    hits = retriever.search(str(args.get("query", "")), top_k)
                    trace.searches.append(
                        SearchEvent(query=str(args.get("query", "")), results=hits, turn=turn)
                    )
                    for h in hits:
                        loaded[h] = specs[h]
                    trace.calls.append(ToolCall(name=SEARCH_TOOL, args=args, turn=turn))
                    content = _search_result_text([specs[h] for h in hits])
                elif c.name not in loaded:
                    reason = "not_loaded" if c.name in specs else "unknown_tool"
                    trace.calls.append(
                        ToolCall(
                            name=c.name,
                            args=args,
                            ok=False,
                            error=reason,
                            turn=turn,
                            known=c.name in specs,
                        )
                    )
                    content = (
                        f"Error: tool '{c.name}' is not loaded; use {SEARCH_TOOL} first"
                        if reason == "not_loaded"
                        else f"Error: unknown tool '{c.name}'"
                    )
                else:
                    out = await pool.call(c.name, args)
                    trace.calls.append(
                        ToolCall(
                            name=c.name,
                            args=args,
                            ok=out.ok,
                            error=out.error,
                            result_preview=out.text[:300],
                            latency_ms=out.latency_ms,
                            turn=turn,
                        )
                    )
                    content = (
                        out.text
                        if out.ok
                        else (
                            out.text
                            if out.text.lower().startswith("error")
                            else f"Error: {out.text}"
                        )
                    )
                messages.append({"role": "tool", "tool_call_id": c.id, "content": content})
        else:
            trace.error = "max_turns"
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
    trace.latency_ms = (time.perf_counter() - t0) * 1000
    return trace


# ---------------------------------------------------------------------------------------------------
class AgentContext:
    """Handed to your agent. Everything routed through it lands in the trace."""

    def __init__(
        self,
        pool: MCPPool,
        trace: Trace,
        mode: str,
        task: Task,
        params: dict,
        top_k: int,
        retriever: Retriever,
    ):
        self._pool, self._trace, self._retriever, self._top_k = pool, trace, retriever, top_k
        self.mode, self.task, self.params = mode, task, params
        self.system = task.system

    @property
    def servers(self):
        """The suite's ServerConfigs, if your agent prefers to open its own MCP connections
        (then return the message history so toolscope can score it)."""
        return self._pool.servers

    @property
    def base_dir(self) -> str | None:
        """Suite directory; use as cwd when your agent spawns stdio servers with relative paths."""
        return self._pool.base_dir

    @property
    def tools(self) -> list[ToolSpec]:
        return list(self._pool.tools.values())

    @property
    def tools_openai(self) -> list[dict]:
        """OpenAI-format schemas (all mode). In search mode, start from `search_tool_openai` instead."""
        return [s.openai() for s in self._pool.tools.values()]

    search_tool_openai = SEARCH_TOOL_SPEC

    def search(self, query: str) -> list[ToolSpec]:
        hits = self._retriever.search(query, self._top_k)
        self._trace.searches.append(SearchEvent(query=query, results=hits))
        self._trace.calls.append(ToolCall(name=SEARCH_TOOL, args={"query": query}))
        return [self._pool.tools[h] for h in hits]

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        out = await self._pool.call(name, args)
        self._trace.calls.append(
            ToolCall(
                name=name,
                args=args,
                ok=out.ok,
                error=out.error,
                result_preview=out.text[:300],
                latency_ms=out.latency_ms,
                known=name in self._pool.tools,
            )
        )
        return out.text

    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        schema_tokens: int = 0,
        cost_usd: float | None = None,
    ) -> None:
        u = self._trace.usage
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.schema_tokens += schema_tokens
        if cost_usd is not None:
            u.cost_usd = (u.cost_usd or 0) + cost_usd


AgentFn = Callable[[str, AgentContext], Awaitable[Any]]


class AgentError(Exception):
    """Raise from your agent when it fails mid-run, passing whatever history it had, so the calls it
    made before failing are still scored (the run is attributed to `crash`)."""

    def __init__(self, cause: BaseException | str, messages: list[dict] | None = None):
        super().__init__(str(cause))
        self.cause, self.messages = cause, messages or []


def load_ref(ref: str, base_dir: str | None = None) -> AgentFn:
    if base_dir and base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    mod, _, attr = ref.partition(":")
    fn = getattr(importlib.import_module(mod), attr)
    return fn


async def run_custom(
    pool: MCPPool,
    task: Task,
    cfg: AgentConfig,
    fn: AgentFn,
    mode: str,
    repeat: int,
    top_k: int = 5,
    retriever: Retriever | None = None,
) -> Trace:
    trace = Trace(
        task_id=task.id, agent_id=cfg.id, mode=mode, repeat=repeat, exposed_tools=list(pool.tools)
    )
    retriever = retriever or BM25Retriever(list(pool.tools.values()))
    ctx = AgentContext(pool, trace, mode, task, {**cfg.params, "repeat": repeat}, top_k, retriever)
    t0 = time.perf_counter()
    try:
        try:
            out = fn(task.prompt, ctx)
            out = await out if inspect.isawaitable(out) else out
        except AgentError as e:
            out = e.messages
            c = e.cause
            trace.error = f"{type(c).__name__}: {c}" if isinstance(c, BaseException) else str(c)
        if isinstance(out, list):  # OpenAI-format history
            parsed = Trace.from_openai_messages(
                out, task_id=task.id, agent_id=cfg.id, exposed_tools=list(pool.tools), repeat=repeat
            )
            if not trace.calls:
                trace.calls = parsed.calls
            trace.final_answer, trace.turns = parsed.final_answer, parsed.turns
        else:
            trace.final_answer = str(out or "")
    except Exception as e:
        trace.error = f"{type(e).__name__}: {e}"
    trace.latency_ms = (time.perf_counter() - t0) * 1000
    return trace
