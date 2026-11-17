"""Evaluate a pydantic-ai agent with MCP toolsets.

    pip install pydantic-ai
    # suite.yaml
    agents:
      - {id: pai-sonnet, kind: python, ref: "pydantic_ai_agent:agent", modes: [all],
         params: {model: "anthropic:claude-sonnet-5-5"}}

pydantic-ai keeps its own message types, so this file includes a small converter to the
OpenAI shape toolscope scores. Written for pydantic-ai v2 (MCPToolset on FastMCP transports);
verified end to end with pydantic-ai's offline TestModel (params: {model: test}).
"""

import json

from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from toolscope import AgentContext, AgentError


def _toolset(s, cwd):
    if s.transport == "stdio":
        return MCPToolset(
            StdioTransport(s.command, s.args, env=s.env, cwd=cwd), tool_error_behavior="retry"
        )
    return MCPToolset(StreamableHttpTransport(s.url, headers=s.headers))


def _model(name: str):
    if (
        name == "test"
    ):  # offline: pydantic-ai's TestModel calls every tool with schema-generated args
        from pydantic_ai.models.test import TestModel

        return TestModel()
    return name


def to_openai(messages) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    out.append({"role": "user", "content": str(p.content)})
                elif isinstance(p, ToolReturnPart):
                    out.append(
                        {"role": "tool", "tool_call_id": p.tool_call_id, "content": str(p.content)}
                    )
                elif isinstance(p, RetryPromptPart) and p.tool_name:
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(p, "tool_call_id", ""),
                            "is_error": True,
                            "content": f"Error: {p.model_response()}",
                        }
                    )
        elif isinstance(m, ModelResponse):
            calls = [p for p in m.parts if isinstance(p, ToolCallPart)]
            text = "".join(p.content for p in m.parts if isinstance(p, TextPart))
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": c.tool_name,
                            "arguments": json.dumps(c.args_as_dict()),
                        },
                    }
                    for c in calls
                ]
            out.append(msg)
    return out


async def agent(prompt: str, ctx: AgentContext):
    pai = Agent(
        _model(ctx.params.get("model", "anthropic:claude-sonnet-5-5")),
        toolsets=[_toolset(s, ctx.base_dir) for s in ctx.servers],
    )
    with capture_run_messages() as history:
        try:
            async with pai:
                result = await pai.run(prompt)
        except (
            Exception
        ) as e:  # keep the partial history so the calls made before failing get scored
            raise AgentError(e, to_openai(history)) from e
    usage = result.usage()
    ctx.record_usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0)
    return to_openai(result.all_messages())
