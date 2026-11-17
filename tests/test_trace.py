import pytest

from toolscope.trace import SEARCH_TOOL, ToolCall, Trace, parse_search_results


def asst(*calls, content=None):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in calls
        ],
    }


def tool(cid, content, **kw):
    return {"role": "tool", "tool_call_id": cid, "content": content, **kw}


EXPOSED = ["shop__get_order_status", "shop__cancel_order"]


def parse(msgs, exposed=EXPOSED):
    return Trace.from_openai_messages(msgs, task_id="t", exposed_tools=exposed)


def test_parallel_tool_calls_share_a_turn_and_map_results_by_id():
    t = parse(
        [
            {"role": "user", "content": "go"},
            asst(
                ("a", "shop__get_order_status", '{"order_id": "1"}'),
                ("b", "shop__cancel_order", '{"order_id": "2"}'),
            ),
            tool("b", "cancelled"),
            tool("a", "shipped"),
            {"role": "assistant", "content": "done"},
        ]
    )
    assert [c.name for c in t.calls] == EXPOSED
    assert {c.turn for c in t.calls} == {1}
    assert [c.result_preview for c in t.calls] == ["shipped", "cancelled"]
    assert t.turns == 2 and t.final_answer == "done" and t.mode == "external"


def test_malformed_json_args_are_kept_not_dropped():
    t = parse([asst(("a", "shop__get_order_status", "{not json"))])
    assert t.calls[0].args == {"__unparseable__": "{not json"}


def test_dict_args_and_missing_args():
    t = parse(
        [
            asst(("a", "shop__get_order_status", None)),
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "b",
                        "function": {"name": "shop__cancel_order", "arguments": {"order_id": "x"}},
                    }
                ],
            },
        ]
    )
    assert t.calls[0].args == {} and t.calls[1].args == {"order_id": "x"}


def test_content_part_lists_are_flattened():
    t = parse(
        [
            asst(("a", "shop__get_order_status", "{}")),
            tool(
                "a",
                [{"type": "text", "text": "Error:"}, {"type": "text", "text": "not found"}, "junk"],
            ),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It"}, {"type": "text", "text": "failed"}],
            },
        ]
    )
    assert t.calls[0].ok is False and t.calls[0].error == "Error: not found"
    assert t.final_answer == "It failed"


def test_is_error_flag_and_non_string_content():
    t = parse(
        [
            asst(("a", "shop__get_order_status", "{}"), ("b", "shop__cancel_order", "{}")),
            tool("a", "boom", is_error=True),
            tool("b", 42),
        ]
    )
    assert t.calls[0].ok is False and t.calls[0].error == "boom"
    assert t.calls[1].ok is True and t.calls[1].result_preview == "42"


def test_missing_tool_results_leave_call_ok_without_preview():
    t = parse([asst(("a", "shop__get_order_status", "{}")), tool("zzz", "orphan result")])
    assert t.calls[0].ok is True and t.calls[0].result_preview == ""


def test_calls_without_ids_still_recorded():
    t = parse(
        [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "shop__cancel_order", "arguments": "{}"}}],
            }
        ]
    )
    assert len(t.calls) == 1


def test_qualified_and_bare_names_both_known():
    t = parse(
        [
            asst(
                ("a", "shop__get_order_status", "{}"),
                ("b", "cancel_order", "{}"),
                ("c", "delete_everything", "{}"),
                ("d", SEARCH_TOOL, '{"query": "x"}'),
            )
        ]
    )
    assert [c.known for c in t.calls] == [True, True, False, True]
    assert [c.name for c in t.real_calls] == [
        "shop__get_order_status",
        "cancel_order",
        "delete_everything",
    ]


def test_no_exposed_tools_means_everything_known():
    t = parse([asst(("a", "anything", "{}"))], exposed=None)
    assert t.calls[0].known and t.exposed_tools == []


def test_assistant_with_content_and_tool_calls_is_not_final_answer():
    t = parse([asst(("a", "shop__get_order_status", "{}"), content="let me check")])
    assert t.final_answer == ""


def test_real_calls_excludes_search():
    t = Trace(task_id="t", agent_id="a", calls=[ToolCall(name=SEARCH_TOOL), ToolCall(name="x")])
    assert [c.name for c in t.real_calls] == ["x"]


# ---- search events (#32) ---------------------------------------------------------------------
SEARCH_TEXT = (
    "Found tools (now callable):\n"
    "- shop__cancel_order: Cancel an order.\n"
    "- shop__get_order_status: Get status."
)


def test_search_call_and_result_become_search_event():
    t = parse(
        [
            asst(("s", SEARCH_TOOL, '{"query": "order status"}')),
            tool("s", SEARCH_TEXT),
            asst(("a", "get_order_status", '{"order_id": "1"}')),
        ]
    )
    assert [c.name for c in t.calls] == [SEARCH_TOOL, "get_order_status"]
    (ev,) = t.searches
    assert ev.query == "order status" and ev.turn == 1
    assert ev.results == ["shop__cancel_order", "shop__get_order_status"]


def test_custom_search_tool_name_is_normalised():
    t = Trace.from_openai_messages(
        [asst(("s", "find_tools", '{"q": "cancel"}')), tool("s", '["cancel_order"]')],
        task_id="t",
        exposed_tools=EXPOSED,
        search_tool="find_tools",
    )
    assert [c.name for c in t.calls] == [SEARCH_TOOL] and t.real_calls == []
    assert t.searches[0].query == "cancel" and t.searches[0].results == ["shop__cancel_order"]


def test_search_without_result_or_with_error_has_no_results():
    t = parse(
        [
            asst(("s", SEARCH_TOOL, '{"query": "a"}'), ("e", SEARCH_TOOL, '{"query": "b"}')),
            tool("e", "Error: index unavailable shop__cancel_order"),
        ]
    )
    assert [(e.query, e.results) for e in t.searches] == [("a", []), ("b", [])]


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            '["shop__cancel_order", "get_order_status"]',
            ["shop__cancel_order", "shop__get_order_status"],
        ),
        (
            '{"tools": [{"name": "get_order_status"}, {"function": {"name": "cancel_order"}}]}',
            ["shop__get_order_status", "shop__cancel_order"],
        ),
        ({"results": [{"qualified": "shop__cancel_order"}]}, ["shop__cancel_order"]),
        ([{"type": "text", "text": '["cancel_order"]'}], ["shop__cancel_order"]),
        (
            "1. `cancel_order` - cancel\n2. get_order_status: status\n3. nope: x",
            ["shop__cancel_order", "shop__get_order_status"],
        ),
        (
            "Try get_order_status, or maybe shop__cancel_order.",
            ["shop__get_order_status", "shop__cancel_order"],
        ),
        ('["get_order_status", "get_order_status", "teleport"]', ["shop__get_order_status"]),
        ("nothing useful here", []),
    ],
)
def test_parse_search_results_formats(content, expected):
    assert parse_search_results(content, EXPOSED) == expected


def test_parse_search_results_without_known_tools():
    assert parse_search_results('["a", "b"]') == ["a", "b"]
    assert parse_search_results("- srv__x: does x\n- note: not a tool") == ["srv__x"]
    assert parse_search_results("free text srv__x") == []


def test_non_object_args_are_kept_as_unparseable():
    t = parse([asst(("a", "shop__get_order_status", "[1, 2]"))])
    assert t.calls[0].args == {"__unparseable__": "[1, 2]"}
