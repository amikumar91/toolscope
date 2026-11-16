"""Evaluate a LangGraph ReAct agent that uses langchain-mcp-adapters.

    pip install langchain langgraph langchain-mcp-adapters langchain-anthropic
    # note: langchain-mcp-adapters 0.3.x needs mcp<2 (verified with mcp 1.30)
    # suite.yaml
    agents:
      - {id: langgraph-sonnet, kind: python, ref: "langgraph_agent:agent", modes: [all],
         params: {model: "anthropic:claude-sonnet-4-5"}}

Your agent keeps its own MCP client; toolscope only needs the message history back.
Verified end to end with a scripted chat model against the demo servers.
"""

from langchain_core.messages import convert_to_openai_messages
from langchain_mcp_adapters.client import MultiServerMCPClient

try:  # LangChain/LangGraph 1.x
    from langchain.agents import create_agent as _make_agent
except ImportError:  # older langgraph
    from langgraph.prebuilt import create_react_agent as _make_agent

from toolscope import AgentContext


def _connection(s, cwd):
    if s.transport == "stdio":
        return {
            "transport": "stdio",
            "command": s.command,
            "args": s.args,
            "env": s.env,
            "cwd": cwd,
        }
    return {"transport": "streamable_http", "url": s.url, "headers": s.headers or {}}


async def agent(prompt: str, ctx: AgentContext):
    client = MultiServerMCPClient({s.name: _connection(s, ctx.base_dir) for s in ctx.servers})
    tools = await client.get_tools()
    graph = _make_agent(ctx.params.get("model", "anthropic:claude-sonnet-4-5"), tools=tools)
    out = await graph.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    return convert_to_openai_messages(out["messages"])  # toolscope parses tool calls + final answer
