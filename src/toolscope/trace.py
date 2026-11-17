"""Framework-neutral trace of one agent run. Everything in scoring works off this, so any agent that can
produce a Trace (or an OpenAI-format message list) can be evaluated."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Collection
from typing import Any, Literal

from pydantic import BaseModel, Field

from .schema import SCHEMA_VERSION

SEARCH_TOOL = "search_tools"


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True  # transport + server-reported success (isError false)
    error: str | None = None
    result_preview: str = ""
    latency_ms: float = 0.0
    known: bool = True  # name existed in the exposed/loaded toolset
    turn: int = 0


class SearchEvent(BaseModel):
    query: str
    results: list[str]  # qualified tool names, ranked
    turn: int = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    schema_tokens: int = 0  # tokens spent on tool schemas put in context (summed over turns)
    cost_usd: float | None = None


class Trace(BaseModel):
    schema_version: str = SCHEMA_VERSION  # of traces.jsonl records; see toolscope.schema
    task_id: str
    agent_id: str
    mode: Literal["all", "search", "external"] = "all"
    repeat: int = 0
    calls: list[ToolCall] = Field(default_factory=list)
    searches: list[SearchEvent] = Field(default_factory=list)
    final_answer: str = ""
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    error: str | None = None  # agent crashed / hit turn limit
    # infrastructure failure (rate limit after retries, dead server, run timeout): run excluded from metrics
    infra_error: str | None = None
    retries: int = 0  # model calls retried after transient errors (not counted as model errors)
    exposed_tools: list[str] = Field(
        default_factory=list
    )  # every tool name the model could have called
    started_at: float = Field(default_factory=time.time)

    @property
    def real_calls(self) -> list[ToolCall]:
        return [c for c in self.calls if c.name != SEARCH_TOOL]

    # ---- interop -------------------------------------------------------------------------------
    @classmethod
    def from_openai_messages(
        cls,
        messages: list[dict[str, Any]],
        *,
        task_id: str,
        agent_id: str = "external",
        exposed_tools: list[str] | None = None,
        repeat: int = 0,
        search_tool: str | Collection[str] = SEARCH_TOOL,
    ) -> Trace:
        """Build a Trace from an OpenAI-style chat history (role/tool_calls/tool messages).

        LangChain (`convert_to_openai_messages`), pydantic-ai, OpenAI Agents SDK, LiteLLM and most custom
        loops can export this shape, so this is the zero-effort integration path.

        Calls to the search meta-tool (`search_tool`, default `search_tools`; pass your own name(s) if
        your agent uses a different one) are recorded as a `SEARCH_TOOL` call plus a `SearchEvent` whose
        ranked results are parsed from the matching tool-result message (see `parse_search_results`).
        """
        t = cls(
            task_id=task_id,
            agent_id=agent_id,
            mode="external",
            repeat=repeat,
            exposed_tools=exposed_tools or [],
        )
        search_names = {search_tool} if isinstance(search_tool, str) else set(search_tool)
        search_names.add(SEARCH_TOOL)
        exposed = set(exposed_tools or [])
        exposed |= {
            n.split("__", 1)[-1] for n in exposed
        }  # frameworks often drop our server__ prefix
        by_id: dict[str, ToolCall] = {}
        searches_by_id: dict[str, SearchEvent] = {}
        turn = 0
        for m in messages:
            role = m.get("role")
            if role == "assistant":
                turn += 1
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    except json.JSONDecodeError:
                        args = {"__unparseable__": raw}
                    if not isinstance(args, dict):
                        args = {"__unparseable__": raw}
                    name = fn.get("name", "?")
                    cid = tc.get("id", str(len(by_id)))
                    is_search = name in search_names
                    call = ToolCall(
                        name=SEARCH_TOOL if is_search else name,
                        args=args,
                        turn=turn,
                        known=is_search or (not exposed) or name in exposed,
                    )
                    t.calls.append(call)
                    by_id[cid] = call
                    if is_search:
                        ev = SearchEvent(query=_search_query(args), results=[], turn=turn)
                        t.searches.append(ev)
                        searches_by_id[cid] = ev
                if m.get("content") and not m.get("tool_calls"):
                    t.final_answer = _text(m["content"])
            elif role == "tool":
                cid = m.get("tool_call_id", "")
                prev = by_id.get(cid)
                if prev:
                    txt = _text(m.get("content", ""))
                    prev.result_preview = txt[:300]
                    if m.get("is_error") or txt.lower().startswith(("error", "tool error")):
                        prev.ok, prev.error = False, txt[:300]
                    if cid in searches_by_id and prev.ok:
                        searches_by_id[cid].results = parse_search_results(
                            m.get("content", ""), exposed_tools or []
                        )
        t.turns = turn
        return t


def _search_query(args: dict[str, Any]) -> str:
    for k in ("query", "q", "search", "text"):
        if isinstance(args.get(k), str):
            return str(args[k])
    strs = [v for v in args.values() if isinstance(v, str)]
    return strs[0] if strs else json.dumps(args, sort_keys=True, default=str)


_NAME_KEYS = ("qualified", "name", "tool", "tool_name", "id")
_LIST_KEYS = ("tools", "results", "matches", "hits")
# "- server__tool: description" (our own search_tools output), also "1. tool - desc", "* `tool`"
_LINE_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*`?([A-Za-z0-9_.\-]+)`?\s*(?:[:\-\u2013(]|$)")


def parse_search_results(content: Any, known_tools: list[str] | None = None) -> list[str]:
    """Ranked tool names from a search meta-tool's result message.

    Understands, in order: JSON (a list of names or of objects with a `name`/`qualified`/`tool` key, or
    an object holding such a list under `tools`/`results`/`matches`/`hits`); toolscope's own text format
    (one `- server__tool: description` line per hit); and, as a tolerant fallback, any of `known_tools`
    mentioned in the text, in order of first appearance. Bare names are qualified when `known_tools`
    makes that unambiguous. Duplicates are dropped, keeping the first (best) rank.
    """
    known = list(known_tools or [])
    bare: dict[str, list[str]] = {}
    for q in known:
        bare.setdefault(q.split("__", 1)[-1], []).append(q)
    known_set = set(known)

    def qualify(n: str) -> str:
        if n in known_set:
            return n
        qs = bare.get(n, [])
        return qs[0] if len(qs) == 1 else n

    def is_known(n: str) -> bool:
        return not known or n in known_set or n in bare

    def dedupe(names: list[str]) -> list[str]:
        out: list[str] = []
        for n in names:
            q = qualify(n)
            if q not in out:
                out.append(q)
        return out

    # 1. structured content (already-decoded JSON, or a JSON string)
    data: Any = content
    if isinstance(content, (str, list)):
        txt = _text(content) if isinstance(content, list) else content
        try:
            data = json.loads(txt)
        except (json.JSONDecodeError, TypeError):
            data = None if isinstance(content, str) else content
    names = _names_from_json(data)
    if names:
        return dedupe([n for n in names if is_known(n)] if known else names)

    # 2. one hit per bulleted/numbered line
    txt = _text(content)
    names = [m.group(1) for ln in txt.splitlines() if (m := _LINE_RE.match(ln))]
    names = [n for n in names if is_known(n)] if known else [n for n in names if "__" in n]
    if names:
        return dedupe(names)

    # 3. fallback: any known tool name mentioned, in order of first appearance
    found: list[tuple[int, str]] = []
    for n in sorted(known_set | set(bare), key=len, reverse=True):
        mt = re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", txt)
        if mt and not any(s <= mt.start() < s + len(o) for s, o in found):
            found.append((mt.start(), n))
    return dedupe([n for _, n in sorted(found)])


def _names_from_json(data: Any) -> list[str]:
    if isinstance(data, dict):
        for k in _LIST_KEYS:
            if isinstance(data.get(k), list):
                return _names_from_json(data[k])
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            inner = item.get("function")
            fn: dict[str, Any] = inner if isinstance(inner, dict) else item
            v = next((fn[k] for k in _NAME_KEYS if isinstance(fn.get(k), str)), None)
            if v:
                out.append(v)
    return out


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)
