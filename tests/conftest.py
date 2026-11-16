"""Shared fixtures: an in-memory MCP pool stand-in and a scripted LLM, so agent-loop tests need no servers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from toolscope.llm import LLMResponse, LLMToolCall
from toolscope.mcp_pool import CallOutcome, ToolSpec

DEMO_DIR = Path(__file__).parents[1] / "examples" / "demo"
DEMO_SUITE = DEMO_DIR / "suite.yaml"


def spec(server: str, name: str, description: str = "", props: dict | None = None,
         required: list[str] | None = None) -> ToolSpec:
    schema = {"type": "object", "properties": props or {}, "required": required or []}
    return ToolSpec(qualified=f"{server}__{name}", server=server, name=name, description=description,
                    input_schema=schema)


SHOP_TOOLS = [
    spec("shop", "get_order_status", "Get the current fulfilment status of an order.",
         {"order_id": {"type": "string"}}, ["order_id"]),
    spec("shop", "get_order_details", "Get line items and total for an order.",
         {"order_id": {"type": "string"}}, ["order_id"]),
    spec("shop", "cancel_order", "Cancel an order. Irreversible.",
         {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id", "reason"]),
    spec("crm", "find_customer_by_email", "Look up a customer by email address.",
         {"email": {"type": "string"}}, ["email"]),
]


class FakePool:
    """Quacks like MCPPool for agent.py: `.tools`, `.call`, `.servers`, `.base_dir`."""

    def __init__(self, tools: list[ToolSpec] | None = None, responses: dict[str, CallOutcome] | None = None):
        self.tools = {t.qualified: t for t in (tools if tools is not None else SHOP_TOOLS)}
        self.responses = responses or {}
        self.servers: list = []
        self.base_dir = None
        self.calls: list[tuple[str, dict]] = []

    async def call(self, qualified: str, args: dict[str, Any]) -> CallOutcome:
        self.calls.append((qualified, args))
        if qualified not in self.tools:
            return CallOutcome(False, f"Error: unknown tool '{qualified}'", 0.0, "unknown_tool")
        return self.responses.get(qualified, CallOutcome(True, f"{qualified} ok {json.dumps(args)}", 1.0))


class ScriptedLLM:
    """Returns pre-baked responses in order; repeats the last one when the script runs out."""

    def __init__(self, *steps: LLMResponse | list[tuple[str, Any]] | str):
        self.steps = [self._coerce(s, i) for i, s in enumerate(steps)]
        self.seen_tools: list[list[str]] = []
        self.i = 0

    @staticmethod
    def _coerce(step, i: int) -> LLMResponse:
        if isinstance(step, LLMResponse):
            return step
        if isinstance(step, str):
            return LLMResponse(step, [], 10, 5)
        calls = [LLMToolCall(f"c{i}_{j}", name, args if isinstance(args, str) else json.dumps(args))
                 for j, (name, args) in enumerate(step)]
        return LLMResponse(None, calls, 10, 5)

    async def complete(self, messages, tools, **_: Any) -> LLMResponse:
        self.seen_tools.append([t["function"]["name"] for t in tools])
        r = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return r


@pytest.fixture
def pool() -> FakePool:
    return FakePool()
