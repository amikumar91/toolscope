"""Run agents x exposure modes x tasks x repeats against a live MCP pool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import IO, TypeVar

from pydantic import BaseModel

from .agent import AgentFn, load_ref, run_builtin, run_custom
from .config import AgentConfig, ExposureMode, Suite, Task
from .llm import make_llm
from .mcp_pool import MCPPool
from .metrics import GroupStats, RunScore, aggregate, score
from .search import BM25Retriever
from .trace import SEARCH_TOOL, Trace

M = TypeVar("M", bound=BaseModel)

CONFIG_ERROR = "config_error"  # prefix of Trace.error when an agent could not be built at all


def _redact_model(suite: Suite, m: M) -> M:
    return type(m).model_validate(suite.redact_obj(m.model_dump(mode="json")))


class _JsonlSink:
    """Appends each finished job's (redacted) trace and score to traces.jsonl / scores.jsonl, flushing
    after every line, so a killed run still leaves one valid line per completed job."""

    def __init__(self, suite: Suite, out_dir: str | Path):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.suite = suite
        self.files: dict[str, IO[str]] = {
            n: (out / f"{n}.jsonl").open("w", encoding="utf-8") for n in ("traces", "scores")
        }

    def write(self, trace: Trace, sc: RunScore) -> None:
        for name, m in (("traces", trace), ("scores", sc)):
            f = self.files[name]
            f.write(_redact_model(self.suite, m).model_dump_json() + "\n")
            f.flush()

    def close(self) -> None:
        for f in self.files.values():
            f.close()


@dataclass
class RunResult:
    suite: Suite
    traces: list[Trace]
    scores: list[RunScore]
    stats: list[GroupStats]
    tool_count: int
    catalogue_tokens: int
    # agent id -> error, for agents that could not be built (bad `ref`, missing model, ...); every job
    # of such an agent is recorded as a `crash` run
    config_errors: dict[str, str] = field(default_factory=dict)

    def redacted(self) -> RunResult:
        """A copy with suite secrets scrubbed from every string field (structurally, so JSON stays valid)."""

        def clean(models: list[M]) -> list[M]:
            return [_redact_model(self.suite, m) for m in models]

        return replace(
            self,
            traces=clean(self.traces),
            scores=clean(self.scores),
            stats=clean(self.stats),
            config_errors=self.suite.redact_obj(self.config_errors),
        )

    def save(self, out_dir: str | Path) -> Path:
        """Write traces.jsonl, scores.jsonl and summary.json, with secrets redacted before serialising."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        red = self.redacted()
        (out / "traces.jsonl").write_text(
            "".join(t.model_dump_json() + "\n" for t in red.traces), encoding="utf-8"
        )
        (out / "scores.jsonl").write_text(
            "".join(s.model_dump_json() + "\n" for s in red.scores), encoding="utf-8"
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "suite": self.suite.name,
                    "tool_count": self.tool_count,
                    "catalogue_tokens": self.catalogue_tokens,
                    "groups": [g.model_dump() for g in red.stats],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out


async def run_suite(
    suite: Suite,
    on_progress: Callable[[RunScore], None] | None = None,
    agent_fns: dict[str, AgentFn] | None = None,
    server_log: str | None = None,
    out_dir: str | Path | None = None,
) -> RunResult:
    """`agent_fns` lets Python callers pass agent callables directly instead of `ref:` strings.

    Every job is isolated: an exception in one job (or an agent that cannot be built at all) is recorded
    as a `crash` run instead of aborting the others. With `out_dir`, traces.jsonl and scores.jsonl are
    written incrementally as jobs finish (`RunResult.save` rewrites them in sorted order at the end).
    """
    from .tokens import count

    agent_fns = dict(agent_fns or {})
    sink = _JsonlSink(suite, out_dir) if out_dir is not None else None
    try:
        async with MCPPool(
            suite.servers, base_dir=str(suite.base_dir), errlog_path=server_log, redact=suite.redact
        ) as pool:
            schemas = {q: s.input_schema for q, s in pool.tools.items()}
            catalogue_tokens = count([s.openai() for s in pool.tools.values()])
            retriever = BM25Retriever(list(pool.tools.values()))
            sem = asyncio.Semaphore(max(1, suite.concurrency))
            traces: list[Trace] = []
            scores: list[RunScore] = []

            # build each agent once up front, so a config error is reported once per agent
            config_errors: dict[str, str] = {}
            for cfg in suite.agents:
                try:
                    if cfg.kind == "python":
                        agent_fns[cfg.id] = agent_fns.get(cfg.id) or load_ref(
                            cfg.ref or "", str(suite.base_dir)
                        )
                    else:
                        make_llm(cfg.kind, cfg.model, cfg.temperature, cfg.params)
                except Exception as e:
                    config_errors[cfg.id] = f"{type(e).__name__}: {e}"

            async def run_job(cfg: AgentConfig, mode: ExposureMode, task: Task, rep: int) -> Trace:
                def crashed(err: str) -> Trace:
                    return Trace(
                        task_id=task.id,
                        agent_id=cfg.id,
                        mode=mode,
                        repeat=rep,
                        error=err,
                        exposed_tools=list(pool.tools),
                    )

                if cfg.id in config_errors:
                    return crashed(f"{CONFIG_ERROR}: {config_errors[cfg.id]}")
                try:
                    if cfg.kind == "python":
                        return await run_custom(
                            pool,
                            task,
                            cfg,
                            agent_fns[cfg.id],
                            mode,
                            rep,
                            suite.exposure.search_top_k,
                            retriever,
                        )
                    llm = make_llm(cfg.kind, cfg.model, cfg.temperature, cfg.params)
                    return await run_builtin(
                        pool, task, cfg, llm, mode, rep, suite.exposure.search_top_k, retriever
                    )
                except Exception as e:
                    return crashed(f"{type(e).__name__}: {e}")

            async def one(cfg: AgentConfig, mode: ExposureMode, task: Task, rep: int) -> None:
                async with sem:
                    tr = await run_job(cfg, mode, task, rep)
                    try:
                        sc = score(tr, task, schemas)
                    except Exception as e:
                        sc = RunScore(
                            task_id=task.id,
                            agent_id=cfg.id,
                            mode=mode,
                            repeat=rep,
                            success=False,
                            failure_stage="crash",
                            failure_reason=f"scoring failed: {type(e).__name__}: {e}",
                        )
                    traces.append(tr)
                    scores.append(sc)
                    if sink:
                        sink.write(tr, sc)
                    if on_progress:
                        on_progress(sc)

            jobs = [
                one(cfg, mode, task, rep)
                for cfg in suite.agents
                for mode in (cfg.modes or suite.exposure.modes)
                for task in suite.tasks
                for rep in range(suite.repeats)
            ]
            await asyncio.gather(*jobs)
            tool_count = len(pool.tools)
    finally:
        if sink:
            sink.close()

    order: dict[tuple[str, str, str], int] = {
        (a.id, m, t.id): i
        for i, (a, m, t) in enumerate(
            (a, m, t)
            for a in suite.agents
            for m in (a.modes or suite.exposure.modes)
            for t in suite.tasks
        )
    }

    def key(x: Trace | RunScore) -> tuple[int, int]:
        return (order.get((x.agent_id, x.mode, x.task_id), 0), x.repeat)

    traces.sort(key=key)
    scores.sort(key=key)
    return RunResult(
        suite, traces, scores, aggregate(scores), tool_count, catalogue_tokens, config_errors
    )


def score_messages(
    messages: list[dict],
    task: Task,
    schemas: dict[str, dict] | None = None,
    agent_id: str = "external",
    search_tool: str | Collection[str] = SEARCH_TOOL,
) -> RunScore:
    """Offline scoring of an already-recorded OpenAI-format conversation (no MCP connection needed).

    `search_tool` names the conversation's search meta-tool, so its calls count as searches."""
    tr = Trace.from_openai_messages(
        messages,
        task_id=task.id,
        agent_id=agent_id,
        exposed_tools=list(schemas) if schemas else None,
        search_tool=search_tool,
    )
    return score(tr, task, schemas or {})
