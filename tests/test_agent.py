import sys

import pytest

from conftest import FakePool, ScriptedLLM
from toolscope import AgentContext, AgentError, Task
from toolscope.agent import load_ref, run_builtin, run_custom
from toolscope.config import AgentConfig
from toolscope.llm import LLMResponse
from toolscope.mcp_pool import CallOutcome
from toolscope.metrics import score
from toolscope.search import BM25Retriever
from toolscope.trace import SEARCH_TOOL

TASK = Task.model_validate(
    {
        "id": "t",
        "prompt": "status of ORD-1",
        "system": "be brief",
        "expect": {"tools": [{"name": "get_order_status", "args": {"order_id": "ORD-1"}}]},
    }
)
CFG = AgentConfig(id="bot", kind="mock", max_turns=3)
STATUS = "shop__get_order_status"


async def test_all_mode_happy_path(pool):
    llm = ScriptedLLM(
        [(STATUS, {"order_id": "ORD-1"})], LLMResponse("shipped", [], 10, 5, cost_usd=0.01)
    )
    tr = await run_builtin(pool, TASK, CFG, llm, "all", 0)
    assert tr.error is None and tr.turns == 2 and tr.final_answer == "shipped"
    assert tr.calls[0].ok and tr.calls[0].name == STATUS
    assert tr.usage.input_tokens == 20 and tr.usage.cost_usd == 0.01 and tr.usage.schema_tokens > 0
    assert SEARCH_TOOL not in llm.seen_tools[0] and STATUS in llm.seen_tools[0]


async def test_hits_max_turns(pool):
    llm = ScriptedLLM([(STATUS, {"order_id": "ORD-1"})])  # never stops calling tools
    tr = await run_builtin(pool, TASK, CFG, llm, "all", 0)
    assert tr.error == "max_turns" and tr.turns == CFG.max_turns and len(tr.calls) == CFG.max_turns
    assert score(tr, TASK, {}).failure_stage == "orchestration"


async def test_search_mode_not_loaded_vs_unknown_tool(pool):
    llm = ScriptedLLM(
        [(STATUS, {"order_id": "ORD-1"}), ("shop__teleport", {})],  # before searching
        [(SEARCH_TOOL, {"query": "order status"})],
        [(STATUS, {"order_id": "ORD-1"})],
        "shipped",
    )
    tr = await run_builtin(
        pool, TASK, CFG.model_copy(update={"max_turns": 5}), llm, "search", 0, top_k=2
    )
    not_loaded, unknown, search, ok = tr.calls
    assert (not_loaded.error, not_loaded.known, not_loaded.ok) == ("not_loaded", True, False)
    assert (unknown.error, unknown.known) == ("unknown_tool", False)
    assert search.name == SEARCH_TOOL and tr.searches[0].results[0] == STATUS
    assert ok.ok and ok.turn == 3
    assert llm.seen_tools[0] == [SEARCH_TOOL]
    assert STATUS in llm.seen_tools[2]
    assert pool.calls == [
        (STATUS, {"order_id": "ORD-1"})
    ]  # only the loaded call reached the server


async def test_search_with_no_hits(pool):
    llm = ScriptedLLM([(SEARCH_TOOL, {"query": "zzzz"})], "nothing")
    tr = await run_builtin(pool, TASK, CFG, llm, "search", 0)
    assert tr.searches[0].results == [] and llm.seen_tools[1] == [SEARCH_TOOL]


async def test_bad_json_and_non_object_args(pool):
    llm = ScriptedLLM([(STATUS, "{oops"), (STATUS, "[1, 2]")], "done")
    tr = await run_builtin(pool, TASK, CFG, llm, "all", 0)
    assert all(c.error.startswith("bad_json") and not c.ok for c in tr.calls)
    assert tr.calls[0].args == {"__unparseable__": "{oops"} and pool.calls == []


async def test_server_error_is_recorded():
    pool = FakePool(
        responses={STATUS: CallOutcome(False, "order not found", 2.0, "order not found")}
    )
    llm = ScriptedLLM([(STATUS, {"order_id": "ORD-1"})], "sorry")
    tr = await run_builtin(pool, TASK, CFG, llm, "all", 0)
    assert tr.calls[0].ok is False and tr.calls[0].error == "order not found"


async def test_llm_exception_is_crash(pool):
    class Boom:
        async def complete(self, *a, **k):
            raise RuntimeError("rate limited")

    tr = await run_builtin(pool, TASK, CFG, Boom(), "all", 0)
    assert tr.error == "RuntimeError: rate limited"
    assert score(tr, TASK, {}).failure_stage == "crash"


# ---- custom agents ---------------------------------------------------------------------------
async def test_custom_agent_routes_through_context(pool):
    async def agent(prompt, ctx: AgentContext):
        assert (
            ctx.mode == "search"
            and ctx.system == "be brief"
            and ctx.params == {"x": 1, "repeat": 2}
        )
        assert (
            len(ctx.tools) == len(ctx.tools_openai) == 4
            and ctx.servers == []
            and ctx.base_dir is None
        )
        assert ctx.search_tool_openai["function"]["name"] == SEARCH_TOOL
        hits = ctx.search("order status")
        ctx.record_usage(input_tokens=7, output_tokens=3, schema_tokens=11, cost_usd=0.5)
        ctx.record_usage(cost_usd=0.25)
        return await ctx.call_tool(hits[0].qualified, {"order_id": "ORD-1"})

    cfg = AgentConfig(id="mine", kind="python", params={"x": 1})
    tr = await run_custom(pool, TASK, cfg, agent, "search", 2)
    assert tr.final_answer.startswith(STATUS) and tr.searches[0].query == "order status"
    assert [c.name for c in tr.calls] == [SEARCH_TOOL, STATUS] and tr.calls[1].known
    assert (tr.usage.input_tokens, tr.usage.schema_tokens, tr.usage.cost_usd) == (7, 11, 0.75)
    assert score(tr, TASK, {}).success


async def test_custom_sync_agent_returning_none(pool):
    tr = await run_custom(
        pool, TASK, AgentConfig(id="s", kind="python"), lambda p, c: None, "all", 0
    )
    assert tr.final_answer == "" and tr.error is None


def asst(cid, name, args):
    return {
        "role": "assistant",
        "tool_calls": [{"id": cid, "function": {"name": name, "arguments": args}}],
    }


HISTORY = [
    {"role": "user", "content": "status of ORD-1"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "function": {"name": "get_order_status", "arguments": '{"order_id": "ORD-1"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "1", "content": "shipped"},
]


async def test_custom_agent_returning_history(pool):
    async def agent(prompt, ctx):
        return HISTORY + [{"role": "assistant", "content": "It shipped"}]

    tr = await run_custom(pool, TASK, AgentConfig(id="h", kind="python"), agent, "all", 0)
    assert (
        tr.final_answer == "It shipped" and tr.turns == 2 and tr.calls[0].name == "get_order_status"
    )
    assert score(tr, TASK, {}).success


async def test_agent_error_with_partial_history_is_scored(pool):
    async def agent(prompt, ctx):
        raise AgentError(RuntimeError("parser blew up"), HISTORY)

    tr = await run_custom(pool, TASK, AgentConfig(id="e", kind="python"), agent, "all", 0)
    assert tr.error == "RuntimeError: parser blew up" and len(tr.calls) == 1
    s = score(tr, TASK, {})
    assert s.failure_stage == "crash" and s.real_calls == 1 and s.arg_accuracy == 1


async def test_agent_error_with_string_cause(pool):
    async def agent(prompt, ctx):
        raise AgentError("gave up")

    tr = await run_custom(pool, TASK, AgentConfig(id="e", kind="python"), agent, "all", 0)
    assert tr.error == "gave up" and tr.calls == []


async def test_custom_agent_unexpected_exception(pool):
    async def agent(prompt, ctx):
        await ctx.call_tool(STATUS, {"order_id": "ORD-1"})
        raise KeyError("x")

    tr = await run_custom(pool, TASK, AgentConfig(id="e", kind="python"), agent, "all", 0)
    assert tr.error == "KeyError: 'x'" and len(tr.calls) == 1


def test_load_ref_resolves_relative_to_base_dir(tmp_path, monkeypatch):
    (tmp_path / "ts_ref_mod.py").write_text("def agent(p, c):\n    return 'hi'\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    fn = load_ref("ts_ref_mod:agent", str(tmp_path))
    assert fn("p", None) == "hi" and sys.path[0] == str(tmp_path)
    load_ref("ts_ref_mod:agent", str(tmp_path))
    assert sys.path.count(str(tmp_path)) == 1
    with pytest.raises(AttributeError):
        load_ref("ts_ref_mod:nope")


async def test_history_with_search_scores_like_ctx_search(pool):
    """#32: a search-mode agent returning its history must score like one using ctx.search."""
    retriever = BM25Retriever(list(pool.tools.values()))
    hits = retriever.search("order status", 5)

    async def via_ctx(prompt, ctx):
        found = ctx.search("order status")
        await ctx.call_tool(found[0].qualified, {"order_id": "ORD-1"})
        return "shipped"

    async def via_history(prompt, ctx):
        found = "\n".join(f"- {h}: {pool.tools[h].description}" for h in hits)
        return [
            {"role": "user", "content": prompt},
            asst("s", SEARCH_TOOL, '{"query": "order status"}'),
            {
                "role": "tool",
                "tool_call_id": "s",
                "content": "Found tools (now callable):\n" + found,
            },
            asst("c", hits[0], '{"order_id": "ORD-1"}'),
            {"role": "tool", "tool_call_id": "c", "content": "shipped"},
            {"role": "assistant", "content": "shipped"},
        ]

    cfg = AgentConfig(id="a", kind="python")
    ta = await run_custom(pool, TASK, cfg, via_ctx, "search", 0, retriever=retriever)
    tb = await run_custom(pool, TASK, cfg, via_history, "search", 0, retriever=retriever)
    a, b = score(ta, TASK, {}), score(tb, TASK, {})
    fields = [
        "success",
        "failure_stage",
        "search_calls",
        "recall_at_k",
        "reciprocal_rank",
        "skipped_search",
        "real_calls",
        "arg_accuracy",
        "hallucinated_calls",
    ]
    assert {f: getattr(a, f) for f in fields} == {f: getattr(b, f) for f in fields}
    assert b.search_calls == 1 and b.recall_at_k == 1.0 and b.skipped_search is False
    assert tb.searches[0].results == ta.searches[0].results == hits


async def test_history_with_custom_search_tool_name(pool):
    async def agent(prompt, ctx):
        return [
            asst("s", "lookup", '{"query": "status"}'),
            {"role": "tool", "tool_call_id": "s", "content": '["get_order_status"]'},
        ] + HISTORY[1:]

    cfg = AgentConfig(id="a", kind="python", search_tool="lookup")
    tr = await run_custom(pool, TASK, cfg, agent, "search", 0)
    assert tr.searches[0].results == [STATUS] and [c.name for c in tr.real_calls] == [
        "get_order_status"
    ]
    s = score(tr, TASK, {})
    assert s.success and s.recall_at_k == 1.0 and not s.skipped_search
