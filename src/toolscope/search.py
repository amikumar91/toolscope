"""Built-in tool-search exposure mode.

Mirrors provider-side tool search / deferred loading (Anthropic Tool Search Tool, Codex CLI tool search):
the model starts with one `search_tools` meta-tool, and every tool it finds gets its full schema loaded
for later turns. BM25 over name + description + parameter names, no extra deps.

Swap in your own retriever (embeddings, hybrid) by passing any object with `.search(query, k) -> list[str]`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .mcp_pool import ToolSpec
from .trace import SEARCH_TOOL

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ")
    return _TOKEN.findall(text.lower())


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]: ...


class BM25Retriever:
    def __init__(self, tools: list[ToolSpec], k1: float = 1.5, b: float = 0.75):
        self.ids = [t.qualified for t in tools]
        self.docs = []
        for t in tools:
            props = " ".join((t.input_schema.get("properties") or {}).keys())
            self.docs.append(_tokens(f"{t.name} {t.name} {t.description} {props}"))
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(tok for d in self.docs for tok in set(d))
        n = len(self.docs)
        self.idf = {tok: math.log(1 + (n - f + 0.5) / (f + 0.5)) for tok, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int) -> list[str]:
        q = _tokens(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for tok in q:
                if tok in tf:
                    f = tf[tok]
                    s += self.idf.get(tok, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, i))
        scores.sort(key=lambda x: -x[0])
        return [self.ids[i] for s, i in scores[:k] if s > 0]


SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL,
        "description": (
            "Search the catalogue of available tools. Tools are NOT loaded until you search for them. "
            "Call this with a short natural-language query describing the capability you need; matching "
            "tools become callable on your next turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What you need to do, e.g. 'cancel an order'"}},
            "required": ["query"],
        },
    },
}
