"""Score a Trace against a Task, then aggregate across repeats/tasks.

Every failed run is attributed to exactly one funnel stage (the earliest that broke):

    discovery     the right tool never surfaced (search mode) / model guessed names that don't exist
    selection     tool was available but not chosen, a forbidden tool was chosen, or tools were used on a
                  task that needed none
    invocation    right tool, but no call with schema-valid, matching arguments succeeded
    orchestration wrong order, or ran out of turns
    outcome       calls were fine but the final answer / end-state check failed
    crash         agent raised
"""

from __future__ import annotations

import importlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

import jsonschema
from pydantic import BaseModel, Field

from .config import ExpectedCall, Task
from .trace import Trace

STAGES = ["discovery", "selection", "invocation", "orchestration", "outcome", "crash"]


def name_matches(expected: str, actual: str) -> bool:
    return actual == expected or actual.split("__", 1)[-1] == expected


def _val_eq(exp: Any, act: Any) -> bool:
    if isinstance(exp, str) and exp.startswith("re:"):
        return re.search(exp[3:], str(act), re.I) is not None
    if isinstance(exp, (int, float)) and isinstance(act, (int, float, str)):
        try:
            return float(exp) == float(act)
        except ValueError:
            return False
    if isinstance(exp, str) and isinstance(act, str):
        return exp.strip().lower() == act.strip().lower()
    if isinstance(exp, dict) and isinstance(act, dict):
        return all(k in act and _val_eq(v, act[k]) for k, v in exp.items())
    return exp == act


def args_match(exp: ExpectedCall, args: dict[str, Any]) -> bool:
    return all(k in args and _val_eq(v, args[k]) for k, v in exp.args.items())


class RunScore(BaseModel):
    task_id: str
    agent_id: str
    mode: str
    repeat: int
    success: bool
    failure_stage: str | None = None
    failure_reason: str = ""
    no_tools_task: bool = False
    # discovery
    search_calls: int = 0
    recall_at_k: float | None = None  # expected tools surfaced by search / expected tools
    reciprocal_rank: float | None = None
    skipped_search: bool = False
    hallucinated_calls: int = 0
    # selection
    expected_recall: float | None = None  # expected tools actually called / expected tools
    precision: float | None = None  # distinct called tools that were expected / distinct called
    wrong_tool_calls: int = 0
    forbidden_called: bool = False
    irrelevance_ok: bool | None = None
    # invocation
    real_calls: int = 0
    schema_valid_rate: float | None = None
    arg_accuracy: float | None = (
        None  # expected calls with a successful, arg-matching call / expected
    )
    error_calls: int = 0
    first_call_ok: bool | None = None
    had_error: bool = False
    recovered: bool = False
    # orchestration
    order_ok: bool | None = None
    redundant_calls: int = 0
    over_budget: bool = False
    # outcome
    answer_ok: bool | None = None
    # cost
    input_tokens: int = 0
    output_tokens: int = 0
    schema_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    turns: int = 0


def _run_check(ref: str, trace: Trace, task: Task) -> bool:
    mod, _, fn = ref.partition(":")
    return bool(getattr(importlib.import_module(mod), fn)(trace, task))


def score(trace: Trace, task: Task, schemas: dict[str, dict]) -> RunScore:
    exp = task.expect
    real = trace.real_calls
    # allow bare names (from external frameworks) to resolve to qualified schemas when unambiguous
    bare: dict[str, list[str]] = defaultdict(list)
    for q in schemas:
        bare[q.split("__", 1)[-1]].append(q)
    schemas = {**{b: schemas[qs[0]] for b, qs in bare.items() if len(qs) == 1}, **schemas}
    s = RunScore(
        task_id=task.id,
        agent_id=trace.agent_id,
        mode=trace.mode,
        repeat=trace.repeat,
        success=False,
        no_tools_task=exp.no_tools,
        turns=trace.turns,
        input_tokens=trace.usage.input_tokens,
        output_tokens=trace.usage.output_tokens,
        schema_tokens=trace.usage.schema_tokens,
        cost_usd=trace.usage.cost_usd,
        latency_ms=trace.latency_ms,
    )

    # ---- discovery ---------------------------------------------------------------------------
    s.search_calls = len(trace.searches)
    s.hallucinated_calls = sum(1 for c in real if not c.known)
    surfaced: set[str] = {r for ev in trace.searches for r in ev.results}
    if trace.mode == "search" and exp.tools:
        first_search_turn = min((ev.turn for ev in trace.searches), default=10**9)
        s.skipped_search = (
            any(c.turn < first_search_turn for c in real) if trace.searches else bool(real)
        )
        hits = [e for e in exp.tools if any(name_matches(e.name, r) for r in surfaced)]
        s.recall_at_k = len(hits) / len(exp.tools)
        rr = 0.0
        for ev in trace.searches:
            for i, r in enumerate(ev.results, 1):
                if any(name_matches(e.name, r) for e in exp.tools):
                    rr = 1 / i
                    break
            if rr:
                break
        s.reciprocal_rank = rr

    # ---- selection ----------------------------------------------------------------------------
    called = {c.name for c in real}
    s.forbidden_called = any(name_matches(f, c) for f in task.forbid for c in called)
    if exp.tools:
        got = [e for e in exp.tools if any(name_matches(e.name, c) for c in called)]
        s.expected_recall = len(got) / len(exp.tools)
        if called:
            good = {c for c in called if any(name_matches(e.name, c) for e in exp.tools)}
            s.precision = len(good) / len(called)
        s.wrong_tool_calls = sum(
            1 for c in real if c.known and not any(name_matches(e.name, c.name) for e in exp.tools)
        )
    if exp.no_tools:
        s.irrelevance_ok = not real

    # ---- invocation ---------------------------------------------------------------------------
    s.real_calls = len(real)
    s.error_calls = sum(1 for c in real if not c.ok)
    s.had_error = s.error_calls > 0
    if real:
        s.first_call_ok = real[0].ok
        valid = 0
        for c in real:
            sch = schemas.get(c.name)
            if sch is None:
                continue
            try:
                jsonschema.validate(c.args, sch)
                valid += 1
            except jsonschema.ValidationError:
                pass
        known = [c for c in real if c.name in schemas]
        s.schema_valid_rate = valid / len(known) if known else None
        first_err = next((i for i, c in enumerate(real) if not c.ok), None)
        if first_err is not None:
            s.recovered = any(c.ok for c in real[first_err + 1 :])
    if exp.tools:
        matched = [
            e
            for e in exp.tools
            if any(name_matches(e.name, c.name) and c.ok and args_match(e, c.args) for c in real)
        ]
        s.arg_accuracy = len(matched) / len(exp.tools)

    # ---- orchestration ------------------------------------------------------------------------
    sig = Counter((c.name, repr(sorted(c.args.items()))) for c in real)
    s.redundant_calls = sum(n - 1 for n in sig.values() if n > 1)
    budget = exp.max_calls if exp.max_calls is not None else len(exp.tools) + 2
    s.over_budget = len(real) > budget
    if exp.ordered and len(exp.tools) > 1:
        idx = []
        for e in exp.tools:
            pos = next(
                (i for i, c in enumerate(real) if name_matches(e.name, c.name) and c.ok), None
            )
            idx.append(pos)
        found = [i for i in idx if i is not None]
        s.order_ok = len(found) == len(idx) and found == sorted(found)

    # ---- outcome ------------------------------------------------------------------------------
    checks: list[bool] = []
    ans = trace.final_answer.lower()
    if exp.answer_contains:
        checks.append(all(x.lower() in ans for x in exp.answer_contains))
    if exp.answer_regex:
        checks.append(re.search(exp.answer_regex, trace.final_answer, re.I | re.S) is not None)
    if exp.check:
        try:
            checks.append(_run_check(exp.check, trace, task))
        except Exception:
            checks.append(False)
    s.answer_ok = all(checks) if checks else None

    # ---- attribution (earliest broken stage wins) ---------------------------------------------
    def fail(stage: str, why: str) -> RunScore:
        s.failure_stage, s.failure_reason = stage, why
        return s

    if trace.error and trace.error != "max_turns":
        return fail("crash", trace.error)
    if exp.no_tools:
        if real:
            return fail("selection", f"called {len(real)} tool(s) on a task that needed none")
    else:
        missing = [e for e in exp.tools if not any(name_matches(e.name, c) for c in called)]
        if trace.mode == "search":
            unsurfaced = [e for e in missing if not any(name_matches(e.name, r) for r in surfaced)]
            if unsurfaced:
                why = (
                    "never searched"
                    if not trace.searches
                    else f"search never surfaced {unsurfaced[0].name}"
                )
                return fail("discovery", why)
        if missing and s.hallucinated_calls and trace.mode != "search":
            return fail("discovery", "called non-existent tool names")
        if missing:
            return fail("selection", f"never called {missing[0].name}")
        if s.forbidden_called:
            return fail("selection", "called a forbidden tool")
        if s.arg_accuracy is not None and s.arg_accuracy < 1:
            return fail("invocation", "no successful call with matching args")
    if trace.error == "max_turns":
        return fail("orchestration", "hit max_turns")
    if s.order_ok is False:
        return fail("orchestration", "tools called out of order")
    if s.answer_ok is False:
        return fail("outcome", "answer/end-state check failed")
    s.success = True
    return s


# ---------------------------------------------------------------------------------------------------
class GroupStats(BaseModel):
    agent_id: str
    mode: str
    runs: int
    tasks: int
    k: int
    pass_at_1: float
    pass_at_k: float
    pass_hat_k: float
    funnel: dict[str, int] = Field(default_factory=dict)
    recall_at_k: float | None = None
    mrr: float | None = None
    skipped_search_rate: float | None = None
    hallucination_rate: float = 0.0  # runs with >=1 non-existent tool name
    selection_accuracy: float | None = None  # mean expected_recall on tool tasks
    precision: float | None = None
    irrelevance_accuracy: float | None = None
    schema_valid_rate: float | None = None
    arg_accuracy: float | None = None
    call_error_rate: float | None = None
    first_call_ok_rate: float | None = None
    recovery_rate: float | None = None
    avg_calls: float = 0.0
    redundant_per_run: float = 0.0
    over_budget_rate: float = 0.0
    avg_input_tokens: float = 0.0
    avg_schema_tokens: float = 0.0
    tokens_per_success: float | None = None
    cost_per_success: float | None = None
    p50_latency_ms: float = 0.0


def _mean(xs: list[Any]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(scores: list[RunScore]) -> list[GroupStats]:
    groups: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for s in scores:
        groups[(s.agent_id, s.mode)].append(s)
    out = []
    for (agent, mode), rs in groups.items():
        by_task: dict[str, list[RunScore]] = defaultdict(list)
        for r in rs:
            by_task[r.task_id].append(r)
        k = max(len(v) for v in by_task.values())
        tool_runs = [r for r in rs if not r.no_tools_task]
        succ = [r for r in rs if r.success]
        total_calls = sum(r.real_calls for r in rs)
        errs = [r for r in rs if r.had_error]
        tokens = sum(r.input_tokens + r.output_tokens for r in rs)
        costs = [r.cost_usd for r in rs if r.cost_usd is not None]
        out.append(
            GroupStats(
                agent_id=agent,
                mode=mode,
                runs=len(rs),
                tasks=len(by_task),
                k=k,
                pass_at_1=_mean([r.success for r in rs]) or 0.0,
                pass_at_k=_mean([any(x.success for x in v) for v in by_task.values()]) or 0.0,
                pass_hat_k=_mean([all(x.success for x in v) for v in by_task.values()]) or 0.0,
                funnel={st: sum(1 for r in rs if r.failure_stage == st) for st in STAGES},
                recall_at_k=_mean([r.recall_at_k for r in rs]),
                mrr=_mean([r.reciprocal_rank for r in rs]),
                skipped_search_rate=_mean([r.skipped_search for r in tool_runs])
                if mode == "search"
                else None,
                hallucination_rate=_mean([r.hallucinated_calls > 0 for r in rs]) or 0.0,
                selection_accuracy=_mean([r.expected_recall for r in tool_runs]),
                precision=_mean([r.precision for r in tool_runs]),
                irrelevance_accuracy=_mean([r.irrelevance_ok for r in rs if r.no_tools_task]),
                schema_valid_rate=_mean([r.schema_valid_rate for r in rs]),
                arg_accuracy=_mean([r.arg_accuracy for r in tool_runs]),
                call_error_rate=(sum(r.error_calls for r in rs) / total_calls)
                if total_calls
                else None,
                first_call_ok_rate=_mean([r.first_call_ok for r in rs]),
                recovery_rate=_mean([r.recovered for r in errs]),
                avg_calls=_mean([r.real_calls for r in rs]) or 0.0,
                redundant_per_run=_mean([r.redundant_calls for r in rs]) or 0.0,
                over_budget_rate=_mean([r.over_budget for r in rs]) or 0.0,
                avg_input_tokens=_mean([r.input_tokens for r in rs]) or 0.0,
                avg_schema_tokens=_mean([r.schema_tokens for r in rs]) or 0.0,
                tokens_per_success=(tokens / len(succ)) if succ and tokens else None,
                cost_per_success=(sum(costs) / len(succ)) if succ and costs else None,
                p50_latency_ms=statistics.median([r.latency_ms for r in rs]) if rs else 0.0,
            )
        )
    return sorted(out, key=lambda g: (-g.pass_at_1, g.agent_id, g.mode))
