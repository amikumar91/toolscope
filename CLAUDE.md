# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

toolscope evaluates whether an LLM (or a user's own agent) actually finds and uses MCP tools. It runs agents × exposure modes × tasks × repeats against live MCP servers, scores each run, and attributes every failure to the **earliest** broken funnel stage: `discovery → selection → invocation → orchestration → outcome` (plus `crash`). Infrastructure failures go to a separate `infra` stage that is excluded from all metrics. README.md is the user-facing spec for task YAML, metrics and outputs.

## Commands

```bash
pip install -e ".[dev]"              # add ,litellm / ,tiktoken for real models / exact token counts
ruff check . && ruff format --check . && mypy src   # lint/format/type gates (config in pyproject.toml)
pytest                               # asyncio_mode=auto; async tests need no decorator
pytest --cov                         # coverage on src/toolscope; fails under 85% (pyproject)
pytest tests/test_metrics.py::test_decoy_is_selection_failure   # single test

# Offline demo (no API keys) — run from examples/demo so relative server paths resolve
cd examples/demo
toolscope tools suite.yaml           # list tools + schema token cost
toolscope run suite.yaml -o results  # writes report.html, summary.json, scores.jsonl, traces.jsonl, servers.log
toolscope run suite.yaml -a mock-good -m search -k 1 --fail-under 0.9
toolscope report results             # re-render report from scores.jsonl
```

`tests/flaky_server.py` is a stdio server whose `die` tool kills the process and whose `hang` tool never returns; `test_robustness.py` drives it with the mock model (prompts `ping` / `please die` / `hang`) to test restarts and timeouts.

`test_demo_suite_end_to_end` spawns the demo stdio servers (`examples/demo/servers.py`) and asserts `tool_count == 10` and 100% pass for `mock/good` in both modes — changing demo tools, tasks or `MockLLM` heuristics will affect it.

## Workflow for roadmap issues

Work is tracked as GitHub issues (pinned Roadmap: #27; milestones v0.2 → v0.4). One issue per branch and per PR.

1. **Start**: `git checkout main && git pull`, read the issue (`gh issue view <N>`), and check its "Depends on" section. If a dependency is still open, say so before starting. Then claim it: `gh issue edit <N> --add-assignee amikumar91` and `gh issue comment <N> --body "Working on this."`.
2. **Branch**: `<type>/<N>-<short-slug>` from `main`, e.g. `fix/2-env-interpolation`, `feat/9-per-tool-stats`, `chore/1-ci`. Never commit directly to `main`.
3. **Commits**: Conventional Commits, `<type>(<scope>): <summary>` (types: feat, fix, test, docs, refactor, chore, ci; scope = module, e.g. `metrics`, `config`). Reference the issue in the body (`Refs #N`). Commits are authored by the repo owner only: **never add a `Co-Authored-By` trailer or any Claude/AI attribution** to commits or PR descriptions.
4. **Before pushing, all must pass** (the same gates CI runs, see `.github/workflows/ci.yml`): `ruff check . && ruff format --check . && mypy src && pytest`, then the offline demo from `examples/demo`: `toolscope run suite.yaml -a mock-good -k 2 --fail-under 1.0 -o results` (`results/` is gitignored; the commands work unchanged on Linux, macOS and Windows). Add or update tests for every behaviour change, and tick the issue's acceptance criteria.
5. **Docs**: update README.md (user-facing behaviour, YAML keys, CLI flags), and this file if architecture or invariants change. Keep output schemas (`summary.json`, `scores.jsonl`, `traces.jsonl`, `tool_stats.json`) backward compatible, or bump their `schema_version` major (`SCHEMA_VERSION` in schema.py; readers call `check_schema_version`). Every new output file must carry the same `schema_version` key.
6. **PR**: `gh pr create --base main` with title in commit format, body containing a summary, test evidence, and `Closes #N`. Don't merge. Leave merging to the owner unless asked.
   **After merge**: confirm the issue closed (`gh issue view <N> --json state`); if `Closes #N` didn't close it, run `gh issue close <N> --reason completed --comment "Fixed in #<PR>"`. Then `git checkout main && git pull`.
7. Don't expand scope beyond the issue. Note follow-ups as new issues (ask first) instead of folding them in.

## Architecture

Data flow: `Suite.load` (config.py) → `run_suite` (runner.py) opens one `MCPPool` for the whole run → for each job, either `run_builtin` (kind `litellm`/`mock`) or `run_custom` (kind `python`) in agent.py produces a `Trace` → `metrics.score(trace, task, schemas)` → `RunScore` → `aggregate` → `GroupStats` per (agent, mode) → report.py.

Key cross-file invariants:

- **Tool naming**: tools are namespaced `server__tool` (`SEP` in mcp_pool.py); server names must not contain `__`. Task expectations, `forbid`, and custom agents may use bare names — `metrics.name_matches` and `MCPPool.call` resolve bare names when unambiguous. Keep these consistent if touching naming.
- **`Trace` is the framework-neutral contract** (trace.py). Everything that produces calls (built-in loop, `AgentContext.call_tool`/`search`, `Trace.from_openai_messages` for returned histories and `score_messages` offline scoring) must populate the same `ToolCall`/`SearchEvent`/`Usage` fields that metrics.py reads. The `search_tools` meta-tool call is recorded as a `ToolCall` named `SEARCH_TOOL`; `Trace.real_calls` excludes it. `from_openai_messages` normalises the configured search tool name (`AgentConfig.search_tool` / `search_tool=`) to `SEARCH_TOOL` and builds a `SearchEvent` per search call, with hits from `parse_search_results` on the tool-result message. `metrics.score` computes discovery metrics for mode `search`, and for mode `external` only when the trace has searches.
- **Exposure modes**: `all` exposes every schema each turn; `search` exposes only `SEARCH_TOOL_SPEC` and loads hit schemas into `loaded` for later turns. Calling an existing-but-unloaded tool records error `not_loaded`; a nonexistent one records `unknown_tool` (`known=False` → counted as hallucinated). `schema_tokens` is accumulated per turn from what was visible, which is how search-mode savings are measured.
- **Retriever** is a `Protocol` (`.search(query, k) -> list[str]` of qualified names); `BM25Retriever` is the default and is built once per run from the pool.
- **Failure attribution** lives in `metrics.score`; `STAGES` order defines precedence, but `INFRA` (`trace.infra_error`) is checked before everything, including `crash`. `INFRA` is not in `STAGES`/`funnel`: `aggregate` drops infra runs before computing any metric (`GroupStats.runs` excludes them; `infra_errors` counts them). Keep that exclusion if you add metrics.
- **Infra vs crash** (errors.py): `is_retryable` (429/5xx/timeouts/connection, matched by MRO class name or status code, so no provider SDK import) and `is_infra_error` (+ auth, `InfraError`). `agent._record_failure` routes an exception to `trace.infra_error` or `trace.error`; use it for any new failure path. Model-call retries live in `complete_with_retry` (own loop rather than LiteLLM `num_retries`, so `trace.retries` is accurate). Arg matching is subset-based with `re:` regex values, case-insensitive strings and numeric coercion (`_val_eq`). Custom end-state checks are `module:function(trace, task) -> bool` refs.
- **LLM layer** (llm.py): `LLM` protocol with `complete(messages, tools, seed=...)` using OpenAI-format messages/tools. `MockLLM` is deterministic, seeded by `md5(prompt|repeat)`; `mock/sloppy` deliberately picks decoys, drops args, and skips search. `make_llm` dispatches on agent `kind`.
- **Custom agents**: `load_ref` adds the suite `base_dir` to `sys.path`, so `ref: "my_agent:agent"` resolves relative to the suite file. An agent either routes calls through `AgentContext` (recorded live) or returns OpenAI-format history (parsed); raising `AgentError(cause, partial_history)` still scores the partial calls.
- **mcp SDK compatibility**: must work with both mcp 1.x and 2.x — use `_field(obj, "snake_name", "camelName")` in mcp_pool.py when reading SDK objects rather than attribute access.
- **Secret redaction** is structural: `Suite.redact_obj` scrubs string keys/values of model dumps *before* serialising (`RunResult.redacted()`/`save`, and the CLI renders report.html from the redacted result). Never run `Suite.redact` over finished JSON/HTML text. stdio server stderr goes through a pipe (`_redacting_errlog` in mcp_pool.py) that redacts each line before it reaches `servers.log`.
- **Job isolation**: `run_suite` builds each agent once up front (`load_ref`/`make_llm`); failures go to `RunResult.config_errors` and every job of that agent gets a `crash` trace with error `config_error: ...` (CLI exits 1). Any other exception inside a job becomes a `crash` trace for that job only; nothing may propagate out of `one()`. With `out_dir`, `_JsonlSink` appends+flushes redacted trace/score lines per finished job; `RunResult.save` later rewrites them sorted.
- All runs share one MCP session per server; stdio servers start with `cwd=base_dir`, and their stderr goes to `servers.log`, not the console. Each session is owned by a dedicated asyncio task (`_Conn` in mcp_pool.py) because anyio cancel scopes must be exited by the task that entered them; that is what makes restarting a server from any job possible. `MCPPool.call` returns `CallOutcome(infra=True)` on a lost connection (MCP error code -32000 / closed stream) after restarting the server (guarded by a per-server lock + generation counter); callers turn that into `InfraError`. Tool-call timeouts are plain tool errors (`error="timeout"`), not infra.
- **Resume**: `load_results` + `run_suite(resume_from=...)` keep jobs keyed `(agent_id, mode, task_id, repeat)` that have both a trace and a score, except `infra` runs and `config_error` crashes. The sink is opened only after the pool starts and rewrites kept runs first, so a failed start never truncates resumable results.
