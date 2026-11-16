# toolscope

**Does the model actually find and use your tools?** Plug in any MCP servers, any models (or your own agent), and a set of tasks. toolscope runs each task *k* times in each tool-exposure mode and tells you **where** tool use breaks, not just whether the task passed.

```
            ┌───────────┐   ┌───────────┐   ┌────────────┐   ┌───────────────┐   ┌─────────┐
  task ───▶ │ discovery │──▶│ selection │──▶│ invocation │──▶│ orchestration │──▶│ outcome │──▶ pass
            └───────────┘   └───────────┘   └────────────┘   └───────────────┘   └─────────┘
             tool never      right tool      right tool,      wrong order /       answer or
             surfaced by     available but   bad / invalid    ran out of turns    end-state
             search, or      not picked,     args, errors                         check failed
             invented names  decoy picked
```

Each failed run is attributed to the **earliest** broken stage, so "pass rate dropped" turns into "search recall for `get_order_status` is 40%, fix its description".

## Why

When tool search / deferred loading fixes MCP context bloat, it adds a new place to fail: the tool has to be *found* before it can be *chosen*. Public benchmarks (BFCL, τ²-bench, MCP-Atlas, MCPMark) rank *models on their servers*. toolscope measures *your* servers, *your* tool descriptions and *your* agent, on *your* tasks, and puts effectiveness next to token cost.

## Quick start (offline, no API keys)

```bash
git clone https://github.com/amikumar91/toolscope.git && cd toolscope
pip install -e .            # Python 3.10+; works with mcp SDK 1.x and 2.x
cd examples/demo
toolscope tools suite.yaml  # sanity check: list tools + schema token cost
toolscope run suite.yaml -o results
open results/report.html
```

The demo ships two MCP servers (`shop`, `crm`) with deliberately confusable tools, 9 tasks (single call, decoys, cross-server, write ops, an enum arg, an ordered multi-step, a no-tool irrelevance task), and two offline models: `mock/good` and `mock/sloppy`. The sloppy one picks decoys, drops required args and skips search under a seeded RNG, so you can see what a real failing model looks like:

| agent | mode | pass@1 | pass^k | recall@k | selection | schema ✓ | schema tok | tok/success |
|---|---|---|---|---|---|---|---|---|
| mock-good | all | 100% | 100% | — | 100% | 100% | 1,660 | 1,947 |
| mock-good | search | 100% | 100% | 100% | 100% | 100% | 771 | 1,302 |
| mock-sloppy | all | 67% | **11%** | — | 65% | 74% | 1,918 | 3,491 |
| mock-sloppy | search | 64% | **11%** | 100% | 69% | 80% | 879 | 2,453 |

Two things stand out. pass@1 hides inconsistency: 67% of runs pass, but only 11% of tasks pass on all 5 repeats. And search mode halves schema tokens at equal accuracy.

## Real models

```bash
pip install -e ".[litellm]"     # any LiteLLM provider
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY, GEMINI_API_KEY, ...
```

```yaml
agents:
  - {id: sonnet, kind: litellm, model: anthropic/claude-sonnet-4-5, temperature: 0.7}
  - {id: gpt,    kind: litellm, model: openai/gpt-4.1-mini,          temperature: 0.7}
  - {id: local,  kind: litellm, model: ollama/qwen3:8b}
```

Use `temperature > 0` with `repeats >= 3`. pass^k only means something if runs can differ.

### Secrets and environment variables

Keep tokens out of the suite file by referencing environment variables:

```yaml
servers:
  - {name: github, transport: http, url: "https://api.githubcopilot.com/mcp/",
     headers: {Authorization: "Bearer ${GITHUB_TOKEN}"}}
  - {name: db, command: "${PYTHON:-python}", args: [db_server.py], env: {DB_PASSWORD: "${DB_PASSWORD}"}}
agents:
  - {id: gpt, kind: litellm, model: openai/gpt-4.1-mini, params: {api_base: "${OPENAI_BASE:-https://api.openai.com/v1}"}}
```

- `${VAR}` expands to the variable's value. If it isn't set, loading fails with `server 'github': env var GITHUB_TOKEN is not set`.
- `${VAR:-default}` falls back to `default` when `VAR` is unset or empty.
- `$$` writes a literal `$`. A bare `$VAR` (no braces) is left as-is.
- Expansion applies only to `servers[].env`, `headers`, `url`, `command`, `args` and `agents[].params`. Task prompts and expectations are never expanded, so they can contain `$` freely.
- Values expanded into `env`, `headers` and `params` are replaced with `***` in `traces.jsonl`, `scores.jsonl`, `summary.json` and `report.html`. `servers.log` is raw server stderr and isn't scrubbed.

## Evaluate your own agent (LangGraph, pydantic-ai, OpenAI Agents SDK, hand-rolled)

Point `ref` at an async callable. It receives the prompt and an `AgentContext`, and does **one** of these:

```python
from toolscope import AgentContext, AgentError

# A) execute tools through toolscope, which records every call live
async def agent(prompt: str, ctx: AgentContext) -> str:
    tools = ctx.tools_openai                  # OpenAI-format schemas, server__tool names
    ...                                       # your loop
    text = await ctx.call_tool(name, args)    # bare names work if unambiguous
    ctx.record_usage(input_tokens=..., output_tokens=...)
    return final_answer

# B) keep your framework's own MCP client and return the OpenAI-format message history
async def agent(prompt: str, ctx: AgentContext) -> list[dict]:
    ...  # connect to ctx.servers (cwd=ctx.base_dir for stdio)
    try:
        return history
    except Exception as e:
        raise AgentError(e, partial_history)  # calls made before the crash still get scored
```

```yaml
agents:
  - {id: my-langgraph, kind: python, ref: "langgraph_agent:agent", modes: [all],
     params: {model: "anthropic:claude-sonnet-4-5"}}
```

Ready-made adapters in `examples/integrations/`, both verified end to end against the demo servers:

| file | framework | notes |
|---|---|---|
| `langgraph_agent.py` | LangGraph / LangChain + `langchain-mcp-adapters` | uses `convert_to_openai_messages`; adapters 0.3.x need `mcp<2` |
| `pydantic_ai_agent.py` | pydantic-ai v2 (`MCPToolset`) | message converter + `capture_run_messages` so crashes keep partial history; `params: {model: test}` runs offline |
| `pytest_regression.py` | pytest | CI gate on pass^k and search recall |

**Already have logs?** Score them offline, with no MCP connection needed:

```python
from toolscope import Task, score_messages
s = score_messages(openai_messages, Task.model_validate(task_dict), schemas={"get_order_status": {...}})
print(s.success, s.failure_stage, s.failure_reason)
```

## Writing tasks

```yaml
- id: cancel-processing
  prompt: "Cancel order ORD-1043 because the customer said 'ordered by mistake'"
  expect:
    tools: [{name: cancel_order, args: {order_id: ORD-1043, reason: "re:mistake"}}]  # subset match; re: = regex
    ordered: false            # true -> expected tools must succeed in this order
    answer_contains: [cancelled]
    # answer_regex: "..."
    # check: "my_checks:order_is_cancelled"   # fn(trace, task) -> bool, for real end-state checks
    # max_calls: 3                             # efficiency budget (default len(tools)+2)
  forbid: [issue_refund]      # destructive tools / known decoys
  tags: [write]

- id: chit-chat
  prompt: "Write a one-line greeting"
  expect: {no_tools: true}    # irrelevance: correct behaviour is calling nothing
```

Good task sets mix single calls, **decoys** (two tools with overlapping descriptions), cross-server tasks, multi-step ordered tasks, write operations with `forbid`, enum/format-sensitive args, and a few **no-tool** tasks to catch over-calling.

## Exposure modes

| mode | what the model sees | models |
|---|---|---|
| `all` | every tool schema, every turn | classic MCP clients (the bloat baseline) |
| `search` | one `search_tools` meta-tool. Hits get their schemas loaded for later turns | Anthropic Tool Search Tool, Codex CLI tool search, deferred loading |

The built-in retriever is BM25 over name + description + parameter names. To test your own (embeddings, hybrid), pass any object with `.search(query, k) -> list[str]` to `run_builtin`. `recall@k` then tells you whether a regression is the retriever's fault or the model's.

## Metrics

| stage | metric | meaning |
|---|---|---|
| headline | **pass@1** | share of runs that fully succeeded |
| | **pass^k** | share of tasks that passed on *all* k repeats (consistency) |
| | pass@k | share of tasks that passed at least once (capability ceiling) |
| discovery | recall@k, MRR | did search surface the expected tool, and how high? |
| | skipped search, hallucinated | called before searching / called names that don't exist |
| selection | selection, precision | expected tools called / share of called tools that were expected |
| | irrelevance | correctly called nothing on no-tool tasks |
| invocation | schema ✓ | args validate against the tool's `inputSchema` (jsonschema) |
| | args | expected calls with a successful, arg-matching call |
| | err/call, first-call ok, recovery | error rate, first try right, fixed itself after an error |
| orchestration | order, redundant, over-budget | ordering, identical repeated calls, calls beyond budget |
| cost | schema tok, tok/success, $/success, p50 ms | what the effectiveness cost you |

## Outputs

`toolscope run` writes `report.html` (failure funnel, all metrics, per-task matrix with the most common failure reason), `summary.json`, `scores.jsonl` (one row per run), `traces.jsonl` (every call, arg, error and search) and `servers.log` (server stderr).

CI: `toolscope run suite.yaml --fail-under 0.9` exits 1 if any agent×mode group's pass^k is below 90%.

## Layout

```
src/toolscope/
  config.py    suite / task / agent schema (pydantic)
  mcp_pool.py  multi-server MCP client, server__tool namespacing, mcp 1.x + 2.x
  search.py    BM25 retriever + search_tools meta-tool
  llm.py       LiteLLM adapter + offline MockLLM
  agent.py     built-in tool loop, AgentContext, AgentError, custom-agent runner
  trace.py     framework-neutral Trace, OpenAI-message import
  metrics.py   per-run scoring, funnel attribution, aggregation
  runner.py    agents × modes × tasks × repeats
  report.py    console + HTML report
  cli.py       toolscope run | report | tools
```

## Caveats

- Runs share one MCP session per server. Set `concurrency: 1`, or reset state in a `check`, for stateful write tools.
- `answer_contains` is a cheap proxy. For write tools, prefer a `check:` that inspects real end state.
- The mock models exist to exercise the pipeline. Don't read their numbers as model rankings.

## Roadmap ideas

LLM-as-judge outcome checks · OpenTelemetry span import (score production traffic) · description-rewrite A/B (same model, two description sets) · per-tool confusion matrix (asked for X, called Y) · ctx-prof integration for runtime context profiling.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
