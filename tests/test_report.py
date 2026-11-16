import io
import re

from rich.console import Console

from toolscope import RunScore, aggregate
from toolscope.report import print_console, render_html


def rs(agent, mode, task, rep, ok, stage=None, **kw):
    return RunScore(
        task_id=task,
        agent_id=agent,
        mode=mode,
        repeat=rep,
        success=ok,
        failure_stage=stage,
        failure_reason="why" if stage else "",
        **kw,
    )


SCORES = [
    rs("good", "all", "t1", 0, True, input_tokens=100, cost_usd=0.01),
    rs("good", "all", "t1", 1, True, input_tokens=100, cost_usd=0.01),
    rs("bad", "search", "t1", 0, False, "discovery", recall_at_k=0.0, reciprocal_rank=0.0),
    rs("bad", "search", "t1", 1, False, "selection"),
    rs("bad", "search", "t2", 0, True, no_tools_task=True, irrelevance_ok=True),
    rs("bad", "search", "t2", 1, False, "crash", no_tools_task=True),
]


def capture(stats) -> str:
    buf = io.StringIO()
    print_console(stats, Console(file=buf, width=250))
    return buf.getvalue()


def test_html_contains_every_group_and_task():
    stats = aggregate(SCORES)
    out = render_html(stats, SCORES, suite_name="My <suite>", tool_count=3, catalogue_tokens=1234)
    assert out.startswith("<!DOCTYPE html>") and out.endswith("</html>")
    assert "My &lt;suite&gt;" in out and "My <suite>" not in out
    assert "1,234 tokens" in out and "2 agent×mode groups" in out and "6 runs" in out
    for g in stats:
        assert f"{g.agent_id} <small>· {g.mode}</small>" in out
    for t in ("t1", "t2"):
        assert f"<td>{t}</td>" in out
    assert "discovery: 1 of 4 runs (25%)" in out
    assert "$0.0100" in out
    assert re.search(r"class='cell-ok'>2/2", out)


def test_html_empty_stats():
    out = render_html([], [], suite_name="empty", tool_count=0, catalogue_tokens=0)
    assert "0 runs" in out and "class='tiles'" not in out


def test_html_best_group_without_failures():
    stats = aggregate(SCORES[:2])
    assert "every run passed" in render_html(
        stats, SCORES[:2], suite_name="s", tool_count=1, catalogue_tokens=1
    )


def test_console_output():
    text = capture(aggregate(SCORES))
    assert "good" in text and "bad" in text and "discovery (1)" in text


def test_console_empty_stats_does_not_crash():
    assert "toolscope" in capture([])
