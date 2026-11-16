"""Framework-neutral trace of one agent run. Everything in scoring works off this, so any agent that can
produce a Trace (or an OpenAI-format message list) can be evaluated."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

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
    ) -> "Trace":
        """Build a Trace from an OpenAI-style chat history (role/tool_calls/tool messages).

        LangChain (`convert_to_openai_messages`), pydantic-ai, OpenAI Agents SDK, LiteLLM and most custom
        loops can export this shape, so this is the zero-effort integration path.
        """
        t = cls(
            task_id=task_id,
            agent_id=agent_id,
            mode="external",
            repeat=repeat,
            exposed_tools=exposed_tools or [],
        )
        exposed = set(exposed_tools or [])
        exposed |= {
            n.split("__", 1)[-1] for n in exposed
        }  # frameworks often drop our server__ prefix
        by_id: dict[str, ToolCall] = {}
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
                    call = ToolCall(
                        name=fn.get("name", "?"),
                        args=args,
                        turn=turn,
                        known=(not exposed)
                        or fn.get("name") in exposed
                        or fn.get("name") == SEARCH_TOOL,
                    )
                    t.calls.append(call)
                    by_id[tc.get("id", str(len(by_id)))] = call
                if m.get("content") and not m.get("tool_calls"):
                    t.final_answer = _text(m["content"])
            elif role == "tool":
                call = by_id.get(m.get("tool_call_id", ""))
                if call:
                    txt = _text(m.get("content", ""))
                    call.result_preview = txt[:300]
                    if m.get("is_error") or txt.lower().startswith(("error", "tool error")):
                        call.ok, call.error = False, txt[:300]
        t.turns = turn
        return t


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)
