from toolscope.trace import SEARCH_TOOL, ToolCall, Trace


def asst(*calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in calls]}


def tool(cid, content, **kw):
    return {"role": "tool", "tool_call_id": cid, "content": content, **kw}


EXPOSED = ["shop__get_order_status", "shop__cancel_order"]


def parse(msgs, exposed=EXPOSED):
    return Trace.from_openai_messages(msgs, task_id="t", exposed_tools=exposed)


def test_parallel_tool_calls_share_a_turn_and_map_results_by_id():
    t = parse([
        {"role": "user", "content": "go"},
        asst(("a", "shop__get_order_status", '{"order_id": "1"}'), ("b", "shop__cancel_order", '{"order_id": "2"}')),
        tool("b", "cancelled"), tool("a", "shipped"),
        {"role": "assistant", "content": "done"},
    ])
    assert [c.name for c in t.calls] == EXPOSED
    assert {c.turn for c in t.calls} == {1}
    assert [c.result_preview for c in t.calls] == ["shipped", "cancelled"]
    assert t.turns == 2 and t.final_answer == "done" and t.mode == "external"


def test_malformed_json_args_are_kept_not_dropped():
    t = parse([asst(("a", "shop__get_order_status", "{not json"))])
    assert t.calls[0].args == {"__unparseable__": "{not json"}


def test_dict_args_and_missing_args():
    t = parse([asst(("a", "shop__get_order_status", None)),
               {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "shop__cancel_order",
                                                                               "arguments": {"order_id": "x"}}}]}])
    assert t.calls[0].args == {} and t.calls[1].args == {"order_id": "x"}


def test_content_part_lists_are_flattened():
    t = parse([
        asst(("a", "shop__get_order_status", "{}")),
        tool("a", [{"type": "text", "text": "Error:"}, {"type": "text", "text": "not found"}, "junk"]),
        {"role": "assistant", "content": [{"type": "text", "text": "It"}, {"type": "text", "text": "failed"}]},
    ])
    assert t.calls[0].ok is False and t.calls[0].error == "Error: not found"
    assert t.final_answer == "It failed"


def test_is_error_flag_and_non_string_content():
    t = parse([asst(("a", "shop__get_order_status", "{}"), ("b", "shop__cancel_order", "{}")),
               tool("a", "boom", is_error=True), tool("b", 42)])
    assert t.calls[0].ok is False and t.calls[0].error == "boom"
    assert t.calls[1].ok is True and t.calls[1].result_preview == "42"


def test_missing_tool_results_leave_call_ok_without_preview():
    t = parse([asst(("a", "shop__get_order_status", "{}")), tool("zzz", "orphan result")])
    assert t.calls[0].ok is True and t.calls[0].result_preview == ""


def test_calls_without_ids_still_recorded():
    t = parse([{"role": "assistant", "tool_calls": [{"function": {"name": "shop__cancel_order", "arguments": "{}"}}]}])
    assert len(t.calls) == 1


def test_qualified_and_bare_names_both_known():
    t = parse([asst(("a", "shop__get_order_status", "{}"), ("b", "cancel_order", "{}"),
                    ("c", "delete_everything", "{}"), ("d", SEARCH_TOOL, '{"query": "x"}'))])
    assert [c.known for c in t.calls] == [True, True, False, True]
    assert [c.name for c in t.real_calls] == ["shop__get_order_status", "cancel_order", "delete_everything"]


def test_no_exposed_tools_means_everything_known():
    t = parse([asst(("a", "anything", "{}"))], exposed=None)
    assert t.calls[0].known and t.exposed_tools == []


def test_assistant_with_content_and_tool_calls_is_not_final_answer():
    t = parse([asst(("a", "shop__get_order_status", "{}"), content="let me check")])
    assert t.final_answer == ""


def test_real_calls_excludes_search():
    t = Trace(task_id="t", agent_id="a", calls=[ToolCall(name=SEARCH_TOOL), ToolCall(name="x")])
    assert [c.name for c in t.real_calls] == ["x"]
