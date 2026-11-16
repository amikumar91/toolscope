from pathlib import Path

import pytest

from toolscope import Suite, Task, Trace, run_suite, score, score_messages
from toolscope.trace import SearchEvent, ToolCall

SCHEMAS = {
    "shop__get_order_status": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    "shop__get_order_details": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    "shop__cancel_order": {"type": "object", "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
                           "required": ["order_id", "reason"]},
}
TASK = Task.model_validate({
    "id": "t", "prompt": "status of ORD-1",
    "expect": {"tools": [{"name": "get_order_status", "args": {"order_id": "ORD-1"}}], "answer_contains": ["shipped"]},
    "forbid": ["cancel_order"],
})


def tr(calls, answer="shipped", mode="all", searches=()):
    return Trace(task_id="t", agent_id="a", mode=mode, calls=calls, final_answer=answer,
                 searches=list(searches), exposed_tools=list(SCHEMAS))


def test_success():
    s = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})]), TASK, SCHEMAS)
    assert s.success and s.failure_stage is None and s.schema_valid_rate == 1 and s.precision == 1


def test_decoy_is_selection_failure():
    s = score(tr([ToolCall(name="shop__get_order_details", args={"order_id": "ORD-1"})]), TASK, SCHEMAS)
    assert s.failure_stage == "selection" and s.wrong_tool_calls == 1 and s.precision == 0


def test_bad_args_is_invocation_failure():
    s = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-9"})]), TASK, SCHEMAS)
    assert s.failure_stage == "invocation"


def test_missing_required_arg_fails_schema_and_recovers():
    calls = [ToolCall(name="shop__get_order_status", args={}, ok=False, error="missing"),
             ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})]
    s = score(tr(calls), TASK, SCHEMAS)
    assert s.success and s.schema_valid_rate == 0.5 and s.recovered and s.first_call_ok is False


def test_forbidden_tool():
    calls = [ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}),
             ToolCall(name="shop__cancel_order", args={"order_id": "ORD-1", "reason": "x"})]
    assert score(tr(calls), TASK, SCHEMAS).failure_stage == "selection"


def test_outcome_failure():
    s = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})], answer="processing"), TASK, SCHEMAS)
    assert s.failure_stage == "outcome"


def test_search_recall_and_discovery_failure():
    ev = SearchEvent(query="order", results=["shop__get_order_details", "shop__cancel_order"], turn=1)
    s = score(tr([], answer="", mode="search", searches=[ev]), TASK, SCHEMAS)
    assert s.failure_stage == "discovery" and s.recall_at_k == 0 and s.reciprocal_rank == 0

    ev2 = SearchEvent(query="order status", results=["shop__get_order_details", "shop__get_order_status"], turn=1)
    s2 = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}, turn=2)],
                  mode="search", searches=[ev2]), TASK, SCHEMAS)
    assert s2.success and s2.recall_at_k == 1 and s2.reciprocal_rank == 0.5 and not s2.skipped_search


def test_irrelevance():
    t = Task.model_validate({"id": "n", "prompt": "hi", "expect": {"no_tools": True}})
    assert score(tr([], answer="hello"), t, SCHEMAS).success
    s = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "x"})]), t, SCHEMAS)
    assert s.failure_stage == "selection" and s.irrelevance_ok is False


def test_regex_args_and_ordering():
    t = Task.model_validate({"id": "o", "prompt": "", "expect": {"ordered": True, "tools": [
        {"name": "get_order_status", "args": {"order_id": "re:^ORD-\\d+$"}},
        {"name": "cancel_order", "args": {"reason": "re:mistake"}}]}})
    good = [ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}),
            ToolCall(name="shop__cancel_order", args={"order_id": "ORD-1", "reason": "by mistake"})]
    assert score(tr(good, answer=""), t, SCHEMAS).success
    assert score(tr(list(reversed(good)), answer=""), t, SCHEMAS).failure_stage == "orchestration"


def test_openai_messages_with_bare_names():
    msgs = [
        {"role": "user", "content": "status of ORD-1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "get_order_status", "arguments": '{"order_id":"ORD-1"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "ORD-1 shipped"},
        {"role": "assistant", "content": "It has shipped."},
    ]
    s = score_messages(msgs, TASK, SCHEMAS)
    assert s.success and s.schema_valid_rate == 1 and s.hallucinated_calls == 0


async def test_demo_suite_end_to_end():
    suite = Suite.load(Path(__file__).parents[1] / "examples/demo/suite.yaml")
    suite.repeats = 1
    suite.agents = [a for a in suite.agents if a.id == "mock-good"]
    res = await run_suite(suite)
    assert res.tool_count == 10
    assert {g.mode for g in res.stats} == {"all", "search"}
    for g in res.stats:
        assert g.pass_at_1 == 1.0
    search = next(g for g in res.stats if g.mode == "search")
    full = next(g for g in res.stats if g.mode == "all")
    assert search.avg_schema_tokens < full.avg_schema_tokens
