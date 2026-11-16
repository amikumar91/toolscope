# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

toolscope evaluates whether an LLM (or a user's own agent) actually finds and uses MCP tools. It runs agents × exposure modes × tasks × repeats against live MCP servers, scores each run, and attributes every failure to the **earliest** broken funnel stage: `discovery → selection → invocation → orchestration → outcome` (plus `crash`). README.md is the user-facing spec for task YAML, metrics and outputs.

## Commands

```bash
pip install -e ".[dev]"              # add ,litellm / ,tiktoken for real models / exact token counts
pytest                               # asyncio_mode=auto; async tests need no decorator
pytest tests/test_metrics.py::test_decoy_is_selection_failure   # single test

# Offline demo (no API keys) — run from examples/demo so relative server paths resolve
cd examples/demo
toolscope tools suite.yaml           # list tools + schema token cost
toolscope run suite.yaml -o results  # writes report.html, summary.json, scores.jsonl, traces.jsonl, servers.log
toolscope run suite.yaml -a mock-good -m search -k 1 --fail-under 0.9
toolscope report results             # re-render report from scores.jsonl
```

`test_demo_suite_end_to_end` spawns the demo stdio servers (`examples/demo/servers.py`) and asserts `tool_count == 10` and 100% pass for `mock/good` in both modes — changing demo tools, tasks or `MockLLM` heuristics will affect it.

## Architecture

Data flow: `Suite.load` (config.py) → `run_suite` (runner.py) opens one `MCPPool` for the whole run → for each job, either `run_builtin` (kind `litellm`/`mock`) or `run_custom` (kind `python`) in agent.py produces a `Trace` → `metrics.score(trace, task, schemas)` → `RunScore` → `aggregate` → `GroupStats` per (agent, mode) → report.py.

Key cross-file invariants:

- **Tool naming**: tools are namespaced `server__tool` (`SEP` in mcp_pool.py); server names must not contain `__`. Task expectations, `forbid`, and custom agents may use bare names — `metrics.name_matches` and `MCPPool.call` resolve bare names when unambiguous. Keep these consistent if touching naming.
- **`Trace` is the framework-neutral contract** (trace.py). Everything that produces calls (built-in loop, `AgentContext.call_tool`/`search`, `Trace.from_openai_messages` for returned histories and `score_messages` offline scoring) must populate the same `ToolCall`/`SearchEvent`/`Usage` fields that metrics.py reads. The `search_tools` meta-tool call is recorded as a `ToolCall` named `SEARCH_TOOL`; `Trace.real_calls` excludes it.
- **Exposure modes**: `all` exposes every schema each turn; `search` exposes only `SEARCH_TOOL_SPEC` and loads hit schemas into `loaded` for later turns. Calling an existing-but-unloaded tool records error `not_loaded`; a nonexistent one records `unknown_tool` (`known=False` → counted as hallucinated). `schema_tokens` is accumulated per turn from what was visible, which is how search-mode savings are measured.
- **Retriever** is a `Protocol` (`.search(query, k) -> list[str]` of qualified names); `BM25Retriever` is the default and is built once per run from the pool.
- **Failure attribution** lives in `metrics.score`; `STAGES` order defines precedence. Arg matching is subset-based with `re:` regex values, case-insensitive strings and numeric coercion (`_val_eq`). Custom end-state checks are `module:function(trace, task) -> bool` refs.
- **LLM layer** (llm.py): `LLM` protocol with `complete(messages, tools, seed=...)` using OpenAI-format messages/tools. `MockLLM` is deterministic, seeded by `md5(prompt|repeat)`; `mock/sloppy` deliberately picks decoys, drops args, and skips search. `make_llm` dispatches on agent `kind`.
- **Custom agents**: `load_ref` adds the suite `base_dir` to `sys.path`, so `ref: "my_agent:agent"` resolves relative to the suite file. An agent either routes calls through `AgentContext` (recorded live) or returns OpenAI-format history (parsed); raising `AgentError(cause, partial_history)` still scores the partial calls.
- **mcp SDK compatibility**: must work with both mcp 1.x and 2.x — use `_field(obj, "snake_name", "camelName")` in mcp_pool.py when reading SDK objects rather than attribute access.
- All runs share one MCP session per server; stdio servers start with `cwd=base_dir`, and their stderr goes to `servers.log`, not the console.
