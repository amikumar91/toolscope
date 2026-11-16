"""Example `kind: python` agent: a hand-rolled loop standing in for "your agent".

Real agents only need to do one of two things:
  A) execute tools via `await ctx.call_tool(name, args)` (recorded live), or
  B) return the OpenAI-format message history (parsed and scored).
This one does B, using toolscope's offline MockLLM as its "model" so the demo runs without API keys.
"""
import json

from toolscope import AgentContext
from toolscope.llm import MockLLM


async def agent(prompt: str, ctx: AgentContext):
    llm = MockLLM("mock/good")
    tools = ctx.tools_openai
    messages = [{"role": "user", "content": prompt}]
    for _ in range(6):
        resp = await llm.complete(messages, tools, seed=ctx.params["repeat"])
        if not resp.tool_calls:
            messages.append({"role": "assistant", "content": resp.content})
            break
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in resp.tool_calls]})
        for c in resp.tool_calls:
            # Executing through the pool directly (not ctx.call_tool) to mimic a framework that owns
            # its own MCP client; toolscope will reconstruct the calls from the returned history.
            out = await ctx._pool.call(c.name, json.loads(c.arguments))
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": out.text if out.ok else f"Error: {out.text}"})
    return messages
