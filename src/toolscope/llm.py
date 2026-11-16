"""Model adapters. One tiny interface: complete(messages, tools) -> LLMResponse in OpenAI tool-calling shape.

- LiteLLMClient: any provider LiteLLM supports (anthropic/, openai/, gemini/, bedrock/, azure/, ollama/, vllm ...)
- MockLLM: deterministic offline model for demos and CI; `mock/sloppy` injects realistic failure modes.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from .search import _tokens
from .trace import SEARCH_TOOL


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model emitted it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class LLM(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict], **params: Any) -> LLMResponse: ...


# ---------------------------------------------------------------------------------------------------
class LiteLLMClient:
    def __init__(self, model: str, temperature: float = 0.0, **params: Any):
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise RuntimeError("kind=litellm needs `pip install toolscope[litellm]`") from e
        self.model, self.temperature, self.params = model, temperature, params

    async def complete(self, messages, tools, **params) -> LLMResponse:
        import litellm

        params.pop("seed", None)  # repeat index; real providers sample naturally (set temperature>0 for k-run variance)
        kw = {**self.params, **params}
        resp = await litellm.acompletion(model=self.model, messages=messages, tools=tools or None,
                                         temperature=self.temperature, **kw)
        msg = resp.choices[0].message
        calls = [LLMToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
                 for tc in (msg.tool_calls or [])]
        usage = getattr(resp, "usage", None)
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            pass
        return LLMResponse(msg.content, calls,
                           getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0, cost)


# ---------------------------------------------------------------------------------------------------
_STOP = set("a an the of for to and or in on at is are be my me i please can you with what whats which "
            "it this that from by do does show tell get".split())
_ID = re.compile(r"\b[A-Z]{2,}-\d+\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_NUM = re.compile(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])")
_QUOTED = re.compile(r"[\"']([^\"']+)[\"']")


class MockLLM:
    """Keyword-matching 'model' with no network. Good enough to exercise every metric end to end.

    mock/good   -> searches when needed, picks best-matching tool, fills args from the prompt.
    mock/sloppy -> same, but (seeded per repeat) sometimes picks the runner-up tool, drops a required
                   argument, skips search and guesses a name, or answers without tools. Gives you
                   non-trivial pass@k vs pass^k spreads to look at.
    """

    def __init__(self, model: str = "mock/good", **_: Any):
        self.sloppy = model.endswith("sloppy")

    async def complete(self, messages, tools, seed: int = 0, **_: Any) -> LLMResponse:
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        rng = random.Random(int(hashlib.md5(f"{prompt}|{seed}".encode()).hexdigest()[:8], 16))
        names = {t["function"]["name"]: t["function"] for t in tools}
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        called = [tc["function"]["name"] for m in messages if m["role"] == "assistant"
                  for tc in (m.get("tool_calls") or [])]
        last_err = tool_msgs and str(tool_msgs[-1]["content"]).lower().startswith("error")
        n_out = 12

        def call(name: str, args: dict) -> LLMResponse:
            return LLMResponse(None, [LLMToolCall(f"c{len(called)}", name, json.dumps(args))],
                               _approx(messages, tools), n_out)

        real = {k: v for k, v in names.items() if k != SEARCH_TOOL}
        ranked = _rank(prompt, real)
        wanted = 2 if re.search(r"\b(and then|then|and also|after that)\b", prompt, re.I) else 1
        real_called = [c for c in called if c != SEARCH_TOOL]

        # 1) recover from an error by retrying the same tool with fully-filled args
        if last_err and real_called:
            name = real_called[-1]
            if name in real and called.count(name) < 2:
                return call(name, _fill(prompt, real[name]["parameters"], drop=False))

        # 2) tool-search mode: nothing loaded yet -> search (sloppy sometimes guesses a name instead)
        if SEARCH_TOOL in names and not real:
            if self.sloppy and rng.random() < 0.15 and not called:
                guess = "_".join(w for w in _tokens(prompt) if w not in _STOP)[:30] or "do_task"
                return call(guess, {})
            if called.count(SEARCH_TOOL) < 2:
                q = " ".join(w for w in _tokens(prompt) if w not in _STOP)
                return call(SEARCH_TOOL, {"query": q})

        # 3) pick the next tool to call
        if len(set(real_called)) < wanted:
            remaining = [(s, n) for s, n in ranked if n not in real_called]
            if len(remaining) > 1 and wanted > 1 and not real_called:
                # multi-step: do first whatever the prompt mentions first
                top = [r for r in remaining[:wanted] if r[0] >= 1.0]
                top.sort(key=lambda r: _first_pos(prompt, r[1]))
                remaining = top + [r for r in remaining if r not in top]
            if remaining and remaining[0][0] >= 2.5:
                if self.sloppy and rng.random() < 0.12 and not real_called:
                    return LLMResponse("I can answer that directly: it should be fine.", [], _approx(messages, tools), n_out)
                pick = remaining[0][1]
                if self.sloppy and len(remaining) > 1 and rng.random() < 0.25:
                    pick = remaining[1][1]
                drop = self.sloppy and rng.random() < 0.25
                return call(pick, _fill(prompt, real[pick]["parameters"], drop=drop))
            if SEARCH_TOOL in names and called.count(SEARCH_TOOL) < 2 and real_called:
                return call(SEARCH_TOOL, {"query": " ".join(_tokens(prompt)[-6:])})

        # 4) answer from tool results
        if tool_msgs:
            body = " | ".join(str(m["content"])[:200] for m in tool_msgs if not str(m["content"]).startswith("Found tools"))
            return LLMResponse(f"Result: {body}", [], _approx(messages, tools), 40)
        return LLMResponse("I don't need any tools for this. Here's my answer.", [], _approx(messages, tools), 20)


def _rank(prompt: str, fns: dict[str, dict]) -> list[tuple[float, str]]:
    q = [w for w in _tokens(prompt) if w not in _STOP]
    docs = {n: _tokens(n.split("__")[-1] + " " + (f.get("description") or "")) for n, f in fns.items()}
    df = Counter(t for d in docs.values() for t in set(d))
    N = max(len(docs), 1)
    out = []
    for n, d in docs.items():
        name_toks = set(_tokens(n.split("__")[-1]))
        s = sum((3.0 if t in name_toks else 1.0) * math.log(1 + N / df[t]) for t in set(q) if t in set(d))
        out.append((s, n))
    return sorted(out, key=lambda x: -x[0])


def _first_pos(prompt: str, name: str) -> int:
    low = prompt.lower()
    pos = [low.find(t) for t in _tokens(name.split("__")[-1]) if t not in _STOP and low.find(t) >= 0]
    return min(pos) if pos else 10**6


_CODE = re.compile(r"\b[A-Z]{2,}\d{3,}\b")


def _fill(prompt: str, schema: dict, drop: bool) -> dict:
    props = schema.get("properties") or {}
    req = schema.get("required") or []
    ids, emails = _ID.findall(prompt), _EMAIL.findall(prompt)
    nums, quoted, codes = _NUM.findall(prompt), _QUOTED.findall(prompt), _CODE.findall(prompt)
    args: dict[str, Any] = {}
    for name, p in props.items():
        lname, typ = name.lower(), p.get("type")
        if "enum" in p:
            hit = [e for e in p["enum"] if str(e).lower() in prompt.lower()]
            if hit or name in req:
                args[name] = hit[0] if hit else p["enum"][0]
        elif lname.endswith("id") and ids:
            args[name] = ids.pop(0)
        elif any(w in lname for w in ("number", "code", "ref")) and codes:
            args[name] = codes.pop(0)
        elif "email" in lname and emails:
            args[name] = emails.pop(0)
        elif typ in ("integer", "number") and nums:
            v = nums.pop(0)
            args[name] = int(float(v)) if typ == "integer" else float(v)
        elif typ == "string" and quoted:
            args[name] = quoted.pop(0)
        elif typ == "string" and (m := re.search(rf"\b(\w+)\s+{re.escape(lname)}\b", prompt, re.I)) \
                and m.group(1).lower() not in _STOP:
            args[name] = m.group(1).lower()          # e.g. "high priority" -> priority=high
        elif typ == "string" and name in req:
            args[name] = " ".join(w for w in _tokens(prompt) if w not in _STOP)[:80]
    if drop and req:
        args.pop(req[0], None)
    return args


def _approx(messages: list[dict], tools: list[dict]) -> int:
    return (len(json.dumps(messages)) + len(json.dumps(tools))) // 4


def make_llm(kind: str, model: str | None, temperature: float, params: dict) -> LLM:
    if kind == "mock":
        return MockLLM(model or "mock/good")
    if kind == "litellm":
        if not model:
            raise ValueError("kind=litellm needs `model`")
        return LiteLLMClient(model, temperature, **params)
    raise ValueError(f"no LLM for kind={kind}")
