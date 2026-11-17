"""Timeouts, retries, the `infra` stage, dead-server restart and --resume (#5)."""

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import pytest

from conftest import DEMO_SUITE, ScriptedLLM
from toolscope import Suite, Task, run_suite
from toolscope.agent import AgentContext, run_builtin, run_custom
from toolscope.config import AgentConfig
from toolscope.errors import InfraError, backoff_s, is_infra_error, is_retryable
from toolscope.metrics import INFRA, aggregate, score
from toolscope.runner import load_results

FLAKY = str(Path(__file__).with_name("flaky_server.py"))
STATUS = "shop__get_order_status"
TASK = Task.model_validate(
    {
        "id": "t",
        "prompt": "status of ORD-1",
        "expect": {"tools": [{"name": "get_order_status", "args": {"order_id": "ORD-1"}}]},
    }
)
FAST = AgentConfig(id="bot", kind="mock", max_turns=3, retry_backoff_s=0, max_retries=2)


class RateLimitError(Exception):  # same class name as litellm/openai/anthropic
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class FlakyLLM:
    """Raises `errors` in order, then behaves like ScriptedLLM(*steps)."""

    def __init__(self, errors, *steps, delay: float = 0):
        self.errors, self.inner, self.delay, self.attempts = (
            list(errors),
            ScriptedLLM(*steps),
            delay,
            0,
        )

    async def complete(self, messages, tools, **kw):
        self.attempts += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.errors:
            raise self.errors.pop(0)
        return await self.inner.complete(messages, tools, **kw)


# ---- classification ------------------------------------------------------------------------------
class _Resp:
    status_code = 503


class HTTPStatusError(Exception):
    response = _Resp()


@pytest.mark.parametrize(
    "e, retry, infra",
    [
        (RateLimitError("slow down"), True, True),
        (TimeoutError(), True, True),
        (ConnectionResetError(), True, True),
        (HTTPStatusError(), True, True),
        (AuthenticationError("bad key"), False, True),
        (InfraError("server down"), False, True),
        (ValueError("bad"), False, False),
        (KeyError("x"), False, False),
    ],
)
def test_error_classification(e, retry, infra):
    assert (is_retryable(e), is_infra_error(e)) == (retry, infra)


def test_backoff_is_bounded_and_jittered():
    assert backoff_s(3, 0) == 0
    xs = [backoff_s(a, 1.0) for a in range(1, 10) for _ in range(20)]
    assert all(0 <= x <= 30 for x in xs) and len(set(xs)) > 1
    assert all(backoff_s(1, 1.0) <= 1.0 for _ in range(50))


# ---- model call retries --------------------------------------------------------------------------
async def test_transient_llm_errors_are_retried_and_recorded(pool):
    llm = FlakyLLM(
        [RateLimitError("429"), ConnectionError("reset")],
        [(STATUS, {"order_id": "ORD-1"})],
        "shipped",
    )
    tr = await run_builtin(pool, TASK, FAST, llm, "all", 0)
    assert tr.retries == 2 and tr.infra_error is None and tr.error is None
    s = score(tr, TASK, {})
    assert s.success and s.retries == 2


async def test_exhausted_retries_are_infra_and_excluded(pool):
    llm = FlakyLLM([RateLimitError("429")] * 5, "never reached")
    tr = await run_builtin(pool, TASK, FAST, llm, "all", 0)
    assert llm.attempts == 3 and tr.retries == 2 and tr.error is None
    assert tr.infra_error.startswith("model call failed after 3 attempts: RateLimitError")
    bad = score(tr, TASK, {})
    assert (bad.failure_stage, bad.success) == (INFRA, False)

    good = score(
        await run_builtin(
            pool, TASK, FAST, ScriptedLLM([(STATUS, {"order_id": "ORD-1"})], "ok"), "all", 1
        ),
        TASK,
        {},
    )
    (g,) = aggregate([bad, good])
    assert g.runs == 1 and g.infra_errors == 1 and g.pass_at_1 == 1.0 and g.pass_hat_k == 1.0
    assert INFRA not in g.funnel and sum(g.funnel.values()) == 0


async def test_auth_error_is_infra_without_retry(pool):
    llm = FlakyLLM([AuthenticationError("bad key")], "x")
    tr = await run_builtin(pool, TASK, FAST, llm, "all", 0)
    assert llm.attempts == 1 and tr.retries == 0
    assert tr.infra_error == "model call failed: AuthenticationError: bad key"


async def test_model_bug_is_crash_without_retry(pool):
    llm = FlakyLLM([ValueError("bad tool schema")], "x")
    tr = await run_builtin(pool, TASK, FAST, llm, "all", 0)
    assert (
        llm.attempts == 1 and tr.error == "ValueError: bad tool schema" and tr.infra_error is None
    )
    assert score(tr, TASK, {}).failure_stage == "crash"


async def test_hung_model_call_times_out_and_is_retried(pool):
    cfg = FAST.model_copy(update={"timeout_s": 0.05, "max_retries": 1})
    llm = FlakyLLM([], "x", delay=5)
    tr = await run_builtin(pool, TASK, cfg, llm, "all", 0)
    assert llm.attempts == 2 and tr.retries == 1
    assert "timed out after 0.05s" in tr.infra_error


def test_all_infra_group_has_no_counted_runs():
    from toolscope.metrics import RunScore

    rs = [
        RunScore(
            task_id="t", agent_id="a", mode="all", repeat=i, success=False, failure_stage=INFRA
        )
        for i in range(2)
    ]
    (g,) = aggregate(rs)
    assert (g.runs, g.tasks, g.k, g.infra_errors, g.pass_at_1) == (0, 0, 0, 2, 0.0)


# ---- custom agents -------------------------------------------------------------------------------
async def test_custom_agent_infra_exception(pool):
    async def agent(prompt, ctx):
        raise ConnectionError("provider unreachable")

    tr = await run_custom(pool, TASK, AgentConfig(id="c", kind="python"), agent, "all", 0)
    assert tr.infra_error == "ConnectionError: provider unreachable" and tr.error is None


async def test_ctx_call_on_dead_server_marks_infra_even_if_swallowed():
    from conftest import FakePool
    from toolscope.mcp_pool import CallOutcome

    pool = FakePool(
        responses={STATUS: CallOutcome(False, "Error: down", 0, "infra: down", infra=True)}
    )

    async def agent(prompt, ctx: AgentContext):
        with contextlib.suppress(InfraError):
            await ctx.call_tool(STATUS, {"order_id": "ORD-1"})
        return "gave up"

    tr = await run_custom(pool, TASK, AgentConfig(id="c", kind="python"), agent, "all", 0)
    assert tr.infra_error == "infra: down" and score(tr, TASK, {}).failure_stage == INFRA


# ---- live servers: dead server, hung tool, run timeout -------------------------------------------
def flaky_suite(tmp_path, tasks, **server):
    suite = Suite.model_validate(
        {
            "name": "flaky",
            "servers": [{"name": "flaky", "command": sys.executable, "args": [FLAKY], **server}],
            "agents": [{"id": "good", "kind": "mock", "model": "mock/good"}],
            "tasks": [
                {"id": tid, "prompt": prompt, "expect": {"tools": [{"name": tool}]}}
                for tid, prompt, tool in tasks
            ],
            "repeats": 1,
            "concurrency": 1,
            "base_dir": str(tmp_path),
        }
    )
    return suite


PING1, DIE, PING2 = (
    ("ping-1", "ping", "ping"),
    ("die", "please die", "die"),
    ("ping-2", "ping", "ping"),
)


async def test_killed_server_is_infra_and_restarted(tmp_path):
    log = tmp_path / "servers.log"
    res = await run_suite(flaky_suite(tmp_path, [PING1, DIE, PING2]), server_log=str(log))
    by_task = {s.task_id: s for s in res.scores}
    assert by_task["ping-1"].success and by_task["ping-2"].success  # restarted, run continued
    assert (
        by_task["die"].failure_stage == INFRA and "connection lost" in by_task["die"].failure_reason
    )
    die_trace = next(t for t in res.traces if t.task_id == "die")
    assert die_trace.error is None and die_trace.calls[-1].error.startswith("infra:")
    (g,) = res.stats
    assert (g.runs, g.infra_errors, g.pass_at_1) == (2, 1, 1.0) and res.infra_errors == 1
    assert "restarting (1)" in log.read_text(encoding="utf-8")


async def test_server_down_after_restart_limit(tmp_path):
    res = await run_suite(flaky_suite(tmp_path, [DIE, PING2], max_restarts=0))
    assert [s.failure_stage for s in res.scores] == [INFRA, INFRA]
    assert "is down" in res.scores[1].failure_reason
    assert res.stats[0].runs == 0


async def test_hung_tool_call_times_out_and_run_continues(tmp_path):
    hang = ("hang", "hang", "hang")
    res = await run_suite(flaky_suite(tmp_path, [hang, PING2], call_timeout_s=0.3))
    h, p = res.scores
    assert h.failure_stage == "invocation" and h.error_calls >= 1 and p.success
    assert res.traces[0].calls[0].error == "timeout"
    assert "timed out after 0.3s" in res.traces[0].calls[0].result_preview


async def test_run_timeout_is_infra(tmp_path):
    suite = flaky_suite(tmp_path, [("hang", "hang", "hang"), PING2], call_timeout_s=None)
    suite.run_timeout_s = 0.5
    res = await run_suite(suite)
    assert res.scores[0].failure_stage == INFRA and "run_timeout" in res.scores[0].failure_reason
    assert res.scores[1].success


# ---- resume --------------------------------------------------------------------------------------
def _demo(k=2):
    suite = Suite.load(DEMO_SUITE)
    suite.repeats = k
    suite.agents = [a for a in suite.agents if a.id in ("mock-good", "mock-sloppy")]
    return suite


def _groups(res):
    return [g.model_dump(exclude={"p50_latency_ms"}) for g in res.stats]


async def test_resume_runs_only_remaining_jobs_and_matches_clean_run(tmp_path):
    clean = await run_suite(_demo(), out_dir=tmp_path / "clean")
    clean.save(tmp_path / "clean")
    total = len(clean.scores)

    # simulate a run killed part-way: the first third of the lines, a torn last line, and one infra run
    part = tmp_path / "part"
    part.mkdir()
    keep = total // 3
    for f in ("traces.jsonl", "scores.jsonl"):
        lines = (tmp_path / "clean" / f).read_text(encoding="utf-8").splitlines()[:keep]
        if f == "scores.jsonl":
            rec = json.loads(lines[0])
            rec.update(failure_stage=INFRA, success=False)
            lines[0] = json.dumps(rec)
        (part / f).write_text(
            "\n".join(lines) + '\n{"schema_version": "1.1", "task_', encoding="utf-8"
        )

    seen = []
    res = await run_suite(
        _demo(), out_dir=part, resume_from=load_results(part), on_progress=seen.append
    )
    assert res.resumed == keep - 1  # the infra run is retried
    assert len(seen) == len(res.scores) == total
    assert _groups(res) == _groups(clean)
    lines = (part / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == total and all(json.loads(ln)["failure_stage"] != INFRA for ln in lines)


async def test_resume_skips_nothing_without_previous_results(tmp_path):
    traces, scores = load_results(tmp_path)
    assert traces == scores == []
    res = await run_suite(_demo(1), resume_from=(traces, scores))
    assert res.resumed == 0 and len(res.scores) == len(_demo(1).tasks) * 4  # 2 agents x 2 modes
