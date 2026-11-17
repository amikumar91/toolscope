# toolscope

[![CI](https://github.com/amikumar91/toolscope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/amikumar91/toolscope/actions/workflows/ci.yml)

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
- Values expanded into `env`, `headers`, `url`, `args` and `params` are treated as secrets and replaced with `***` inside string values of `traces.jsonl`, `scores.jsonl`, `summary.json` and `report.html`, and in each line of `servers.log` (server stderr). JSON structure and non-string values are never touched, so outputs always stay parseable.
- Values shorter than 4 characters, booleans (`true`, `false`, `yes`, `no`, `on`, `off`) and plain numbers (e.g. `DEBUG=true`, `PORT=8080`) aren't treated as secrets. `${VAR:-default}` fallbacks and `command` expansions aren't either.

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

**Search metrics from a returned history.** `search_calls`, `recall@k`, `reciprocal_rank` and `skipped_search` work for histories (option B and `score_messages`) when the history contains:

1. an assistant `tool_calls` entry for the search meta-tool, `search_tools` by default. If your agent's tool has another name, set `search_tool: <name>` on the agent in YAML, or pass `search_tool=` to `score_messages` / `Trace.from_openai_messages`. The query is read from the `query` argument (or `q`, or the first string argument);
2. the matching `role: tool` message (same `tool_call_id`) holding the ranked hits, in any of these formats: a JSON list of names (`["get_order_status", ...]`), a JSON list of objects with a `name`/`qualified`/`tool` key, an object holding that list under `tools`/`results`/`matches`/`hits`, or toolscope's own text format (one `- server__tool: description` line per hit). As a fallback, known tool names mentioned in free text count, in order of appearance. Bare names are qualified when unambiguous. A search result that is an error (`is_error`, or text starting with `Error`) counts as a search with no hits;
3. the real tool calls in later assistant turns. Calls made in a turn before the first search count as `skipped_search`.

A `search`-mode agent that routes calls through `ctx` gets these from `ctx.search` instead; if it records any call via `ctx`, the returned history is not parsed. Imported histories (`score_messages`) get discovery metrics and `discovery` attribution only when they contain at least one search.

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
| infra | infra, retries | runs lost to infrastructure (excluded from everything above), model calls retried |

Every failed run is attributed to the earliest broken stage: discovery → selection → invocation → orchestration → outcome, or `crash` if the agent raised. Runs where the *infrastructure* failed go to a separate `infra` stage instead. These are rate limits, network or auth errors that persist after retries, a dead MCP server, or a run timeout. They are **excluded** from pass@1, pass^k and every other metric, and counted in `infra_errors` (summary.json) and the report's `infra` column. A flaky provider never makes a model look worse.

## Outputs

`toolscope run` writes `report.html` (failure funnel, all metrics, per-task matrix with the most common failure reason), `summary.json`, `scores.jsonl` (one row per run), `traces.jsonl` (every call, arg, error and search) and `servers.log` (server stderr).

**Schema versioning.** `summary.json` and every record in `scores.jsonl` and `traces.jsonl` carry `"schema_version"` (currently `"1.1"`; `toolscope.SCHEMA_VERSION`). New optional fields may be added within a major version, possibly as a minor bump (`"1.1"`). Removing, renaming or changing the meaning of a field bumps the major version. `toolscope report` reads any `1.x` file and treats files without the field (written before versioning) as `"1"`. It exits 1 with a clear error on another major version instead of misreading the file. Later outputs such as `tool_stats.json` use the same key. 1.1 added the `infra` failure stage, `infra_error` and `retries` on traces, `retries` on scores, and `infra_errors` and `retries` on groups. In 1.1, a group's `runs` counts only the runs included in the metrics, so infra runs aren't in it.

CI: `toolscope run suite.yaml --fail-under 0.9` exits 1 if any agent×mode group's pass^k is below 90%.

### Timeouts, retries and restarts

Nothing waits forever. Defaults shown; set them in the suite YAML:

```yaml
servers:
  - {name: shop, command: python, args: [servers.py, shop],
     call_timeout_s: 60,   # per tool call; a timed-out call returns an error to the model, the run continues
     max_restarts: 1}      # a server whose connection dies is restarted this many times, then marked down
agents:
  - {id: sonnet, kind: litellm, model: anthropic/claude-sonnet-4-5,
     timeout_s: 120,       # per model call (built-in loop)
     max_retries: 3,       # retries on 429 / 5xx / timeouts / connection errors
     retry_backoff_s: 1}   # exponential backoff with full jitter: attempt n waits up to 1s * 2^(n-1), max 30s
run_timeout_s: 600         # per run (agent x mode x task x repeat); null disables
```

- **Model calls** (built-in loop) that fail transiently are retried. The count goes in `retries` (trace, score and group stats) and never counts as a model error. If a call still fails, or fails with an auth error (401/403, never retried), the run is `infra`. Leave LiteLLM's own `num_retries` unset: retries done inside LiteLLM can't be counted.
- **Tool calls** that exceed `call_timeout_s` return `Error: tool call timed out after Ns` (call error `timeout`). The model sees it like any tool error, so that's an `invocation` problem, not infra.
- **Dead servers**: if a server's connection drops (stdio process exited, stream closed), the call that hit it records an error starting `infra:`, the run is `infra`, and the server is restarted for the remaining jobs (logged in `servers.log`). After `max_restarts`, every call to that server fails as `infra`.
- **Custom agents**: an exception from your agent (or the `cause` of an `AgentError`) that looks like a rate limit, timeout, connection or auth error, or a `toolscope.InfraError` you raise yourself, makes the run `infra` instead of `crash`. `ctx.call_tool` raises `InfraError` when the server is down, and the run stays `infra` even if your agent catches it.
- **Run timeout**: a run that exceeds `run_timeout_s` is `infra` with reason `run_timeout: ...`.

### Resuming an interrupted run

```bash
toolscope run suite.yaml -o results            # Ctrl-C, crash, or infra errors part-way
toolscope run suite.yaml -o results --resume   # keeps finished runs, runs only the rest
```

`--resume` reads `traces.jsonl` and `scores.jsonl` from `-o`, keeps every (agent, mode, task, repeat) that has both a trace and a score, and runs the remaining jobs. Torn last lines are ignored. `infra` runs and agent config errors are run again. The final summary is the same as an uninterrupted run. Use the same suite file and flags: runs that aren't part of the current job set (other agents, modes or repeats) are dropped. Raising `-k` with `--resume` runs only the new repeats. From Python: `run_suite(suite, out_dir=d, resume_from=toolscope.load_results(d))`.

Each job is isolated: if one run raises, it is recorded as a `crash` run (with the error) and the rest of the suite carries on. An agent that cannot be built at all (an unresolvable `ref`, a `litellm` agent without `model`, ...) is reported once as a `CONFIG ERROR`, every one of its runs is recorded as `crash` with an error starting `config_error:`, and `toolscope run` exits 1. `traces.jsonl` and `scores.jsonl` are appended and flushed as each job finishes, so an interrupted run still leaves one valid line per completed job; `summary.json` and `report.html` are written at the end (and the `.jsonl` files rewritten in suite order).

## Layout

```
src/toolscope/
  config.py    suite / task / agent schema (pydantic)
  mcp_pool.py  multi-server MCP client, server__tool namespacing, call timeouts, dead-server restart, mcp 1.x + 2.x
  search.py    BM25 retriever + search_tools meta-tool
  llm.py       LiteLLM adapter + offline MockLLM
  agent.py     built-in tool loop (timeouts, retries), AgentContext, AgentError, custom-agent runner
  errors.py    infra-error classification (rate limit / timeout / network / auth), InfraError, backoff
  schema.py    output schema_version + check
  trace.py     framework-neutral Trace, OpenAI-message import
  metrics.py   per-run scoring, funnel attribution, aggregation
  runner.py    agents × modes × tasks × repeats, run timeout, --resume
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
ruff check . && ruff format --check . && mypy src
pytest
```

CI runs these on Python 3.10–3.13 against both mcp 1.x and 2.x (Linux) and on Windows and macOS, plus a gate that fails if `mock-good` drops below 100% pass^k on the offline demo. The demo's HTML report is attached to each run as the `demo-results` artifact.

## License

[MIT](LICENSE)
