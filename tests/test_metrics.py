from pathlib import Path

import pytest

from toolscope import Suite, Task, Trace, run_suite, score, score_messages
from toolscope.trace import SearchEvent, ToolCall, Usage

SCHEMAS = {
    "shop__get_order_status": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "shop__get_order_details": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    "shop__cancel_order": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["order_id", "reason"],
    },
}
TASK = Task.model_validate(
    {
        "id": "t",
        "prompt": "status of ORD-1",
        "expect": {
            "tools": [{"name": "get_order_status", "args": {"order_id": "ORD-1"}}],
            "answer_contains": ["shipped"],
        },
        "forbid": ["cancel_order"],
    }
)


def tr(calls, answer="shipped", mode="all", searches=()):
    return Trace(
        task_id="t",
        agent_id="a",
        mode=mode,
        calls=calls,
        final_answer=answer,
        searches=list(searches),
        exposed_tools=list(SCHEMAS),
    )


def test_success():
    s = score(
        tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})]), TASK, SCHEMAS
    )
    assert s.success and s.failure_stage is None and s.schema_valid_rate == 1 and s.precision == 1


def test_decoy_is_selection_failure():
    s = score(
        tr([ToolCall(name="shop__get_order_details", args={"order_id": "ORD-1"})]), TASK, SCHEMAS
    )
    assert s.failure_stage == "selection" and s.wrong_tool_calls == 1 and s.precision == 0


def test_bad_args_is_invocation_failure():
    s = score(
        tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-9"})]), TASK, SCHEMAS
    )
    assert s.failure_stage == "invocation"


def test_missing_required_arg_fails_schema_and_recovers():
    calls = [
        ToolCall(name="shop__get_order_status", args={}, ok=False, error="missing"),
        ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}),
    ]
    s = score(tr(calls), TASK, SCHEMAS)
    assert s.success and s.schema_valid_rate == 0.5 and s.recovered and s.first_call_ok is False


def test_forbidden_tool():
    calls = [
        ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}),
        ToolCall(name="shop__cancel_order", args={"order_id": "ORD-1", "reason": "x"}),
    ]
    assert score(tr(calls), TASK, SCHEMAS).failure_stage == "selection"


def test_outcome_failure():
    s = score(
        tr(
            [ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})],
            answer="processing",
        ),
        TASK,
        SCHEMAS,
    )
    assert s.failure_stage == "outcome"


def test_search_recall_and_discovery_failure():
    ev = SearchEvent(
        query="order", results=["shop__get_order_details", "shop__cancel_order"], turn=1
    )
    s = score(tr([], answer="", mode="search", searches=[ev]), TASK, SCHEMAS)
    assert s.failure_stage == "discovery" and s.recall_at_k == 0 and s.reciprocal_rank == 0

    ev2 = SearchEvent(
        query="order status", results=["shop__get_order_details", "shop__get_order_status"], turn=1
    )
    s2 = score(
        tr(
            [ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}, turn=2)],
            mode="search",
            searches=[ev2],
        ),
        TASK,
        SCHEMAS,
    )
    assert (
        s2.success and s2.recall_at_k == 1 and s2.reciprocal_rank == 0.5 and not s2.skipped_search
    )


def test_irrelevance():
    t = Task.model_validate({"id": "n", "prompt": "hi", "expect": {"no_tools": True}})
    assert score(tr([], answer="hello"), t, SCHEMAS).success
    s = score(tr([ToolCall(name="shop__get_order_status", args={"order_id": "x"})]), t, SCHEMAS)
    assert s.failure_stage == "selection" and s.irrelevance_ok is False


def test_regex_args_and_ordering():
    t = Task.model_validate(
        {
            "id": "o",
            "prompt": "",
            "expect": {
                "ordered": True,
                "tools": [
                    {"name": "get_order_status", "args": {"order_id": "re:^ORD-\\d+$"}},
                    {"name": "cancel_order", "args": {"reason": "re:mistake"}},
                ],
            },
        }
    )
    good = [
        ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}),
        ToolCall(name="shop__cancel_order", args={"order_id": "ORD-1", "reason": "by mistake"}),
    ]
    assert score(tr(good, answer=""), t, SCHEMAS).success
    assert score(tr(list(reversed(good)), answer=""), t, SCHEMAS).failure_stage == "orchestration"


def test_openai_messages_with_bare_names():
    msgs = [
        {"role": "user", "content": "status of ORD-1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "get_order_status", "arguments": '{"order_id":"ORD-1"}'},
                }
            ],
        },
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


OK_CALL = ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"})


@pytest.mark.parametrize(
    "stage,trace",
    [
        (None, tr([OK_CALL])),
        ("discovery", tr([], mode="search")),  # never searched
        (
            "discovery",
            tr([ToolCall(name="shop__lookup_order", args={}, known=False)]),
        ),  # hallucinated, all mode
        ("selection", tr([ToolCall(name="shop__get_order_details", args={"order_id": "ORD-1"})])),
        (
            "invocation",
            tr([ToolCall(name="shop__get_order_status", args={"order_id": "ORD-1"}, ok=False)]),
        ),
        ("orchestration", tr([OK_CALL]).model_copy(update={"error": "max_turns"})),
        ("outcome", tr([OK_CALL], answer="no idea")),
        ("crash", tr([OK_CALL]).model_copy(update={"error": "RuntimeError: boom"})),
    ],
)
def test_one_failure_per_funnel_stage(stage, trace):
    s = score(trace, TASK, SCHEMAS)
    assert s.failure_stage == stage and s.success is (stage is None)


def test_crash_takes_precedence_over_everything():
    s = score(
        tr([], answer="", mode="search").model_copy(update={"error": "ValueError: x"}),
        TASK,
        SCHEMAS,
    )
    assert s.failure_stage == "crash" and s.failure_reason == "ValueError: x"


def test_answer_regex_and_custom_check(tmp_path, monkeypatch):
    (tmp_path / "ts_checks.py").write_text(
        "def ok(trace, task):\n    return 'ORD' in trace.final_answer\n"
        "def boom(trace, task):\n    raise RuntimeError\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    base = {"id": "c", "prompt": "", "expect": {"answer_regex": r"ord-\d+"}}
    assert score(tr([], answer="see ORD-1"), Task.model_validate(base), SCHEMAS).success
    for ref, ok in [("ts_checks:ok", True), ("ts_checks:boom", False)]:
        t = Task.model_validate({**base, "expect": {**base["expect"], "check": ref}})
        assert score(tr([], answer="see ORD-1"), t, SCHEMAS).success is ok


@pytest.mark.parametrize(
    "exp,act,ok",
    [
        (5, "5.0", True),
        (5, "five", False),
        ("ORD-1", " ord-1 ", True),
        ({"a": 1}, {"a": 1.0, "b": 2}, True),
        ({"a": 1}, {"b": 1}, False),
        ([1], [1], True),
        (True, "true", False),
    ],
)
def test_value_matching(exp, act, ok):
    from toolscope.metrics import _val_eq

    assert _val_eq(exp, act) is ok


def test_aggregate_rates_and_costs():
    from toolscope import aggregate

    good = score(
        tr([OK_CALL]).model_copy(update={"usage": Usage(input_tokens=100, cost_usd=0.02)}),
        TASK,
        SCHEMAS,
    )
    bad = score(
        tr([ToolCall(name="shop__get_order_status", args={}, ok=False)], answer=""), TASK, SCHEMAS
    )
    bad.repeat = 1
    (g,) = aggregate([good, bad])
    assert (g.pass_at_1, g.pass_at_k, g.pass_hat_k, g.k) == (0.5, 1.0, 0.0, 2)
    assert g.call_error_rate == 0.5 and g.recovery_rate == 0.0 and g.funnel["invocation"] == 1
    assert g.tokens_per_success == 100 and g.cost_per_success == 0.02
