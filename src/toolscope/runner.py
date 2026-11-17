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
from .metrics import INFRA, GroupStats, RunScore, aggregate, score
from .schema import SCHEMA_KEY, SCHEMA_VERSION, check_schema_version
from .search import BM25Retriever
from .trace import SEARCH_TOOL, Trace

M = TypeVar("M", bound=BaseModel)

CONFIG_ERROR = "config_error"  # prefix of Trace.error when an agent could not be built at all
RUN_TIMEOUT = "run_timeout"  # prefix of Trace.infra_error when a run exceeded Suite.run_timeout_s

JobKey = tuple[str, str, str, int]  # (agent_id, mode, task_id, repeat)


def _job_key(x: Trace | RunScore) -> JobKey:
    return (x.agent_id, x.mode, x.task_id, x.repeat)


def _read_jsonl(path: Path, model: type[M]) -> list[M]:
    """Records of a .jsonl output; a torn or invalid line (e.g. from a killed run) is skipped."""
    out: list[M] = []
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        check_schema_version(rec, f"{path.name} line {i}")
        try:
            out.append(model.model_validate(rec))
        except ValueError:
            continue
    return out


def load_results(out_dir: str | Path) -> tuple[list[Trace], list[RunScore]]:
    """Traces and scores already written to `out_dir` (for `run_suite(resume_from=...)`). Missing files
    yield empty lists; raises UnsupportedSchemaVersion for files from an incompatible toolscope."""
    out = Path(out_dir)
    return _read_jsonl(out / "traces.jsonl", Trace), _read_jsonl(out / "scores.jsonl", RunScore)


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
    resumed: int = 0  # runs taken from a previous, interrupted run instead of being run again

    @property
    def infra_errors(self) -> int:
        return sum(1 for s in self.scores if s.failure_stage == INFRA)

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
                    SCHEMA_KEY: SCHEMA_VERSION,
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
    resume_from: tuple[list[Trace], list[RunScore]] | None = None,
) -> RunResult:
    """`agent_fns` lets Python callers pass agent callables directly instead of `ref:` strings.

    Every job is isolated: an exception in one job (or an agent that cannot be built at all) is recorded
    as a `crash` run instead of aborting the others; infrastructure failures are recorded as `infra` runs.
    With `out_dir`, traces.jsonl and scores.jsonl are written incrementally as jobs finish
    (`RunResult.save` rewrites them in sorted order at the end).

    `resume_from` (see `load_results`) skips every job that already has both a trace and a score, except
    `infra` runs and agent config errors, which are run again.
    """
    from .tokens import count

    agent_fns = dict(agent_fns or {})
    done: dict[JobKey, tuple[Trace, RunScore]] = {}
    if resume_from:
        prev_traces = {_job_key(t): t for t in resume_from[0]}
        for sc in resume_from[1]:
            tr = prev_traces.get(_job_key(sc))
            if (
                tr is not None
                and sc.failure_stage != INFRA
                and not (tr.error or "").startswith(CONFIG_ERROR)
            ):
                done[_job_key(sc)] = (tr, sc)
    sink: _JsonlSink | None = None
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
                def crashed(err: str | None, infra: str | None = None) -> Trace:
                    return Trace(
                        task_id=task.id,
                        agent_id=cfg.id,
                        mode=mode,
                        repeat=rep,
                        error=err,
                        infra_error=infra,
                        exposed_tools=list(pool.tools),
                    )

                if cfg.id in config_errors:
                    return crashed(f"{CONFIG_ERROR}: {config_errors[cfg.id]}")

                async def go() -> Trace:
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

                try:
                    return await asyncio.wait_for(go(), suite.run_timeout_s)
                except asyncio.TimeoutError:
                    return crashed(None, f"{RUN_TIMEOUT}: run exceeded {suite.run_timeout_s:g}s")
                except Exception as e:
                    return crashed(f"{type(e).__name__}: {e}")

            async def one(cfg: AgentConfig, mode: ExposureMode, task: Task, rep: int) -> None:
                prev = done.get((cfg.id, mode, task.id, rep))
                if prev is not None:  # finished in the run being resumed (already in the sink)
                    traces.append(prev[0])
                    scores.append(prev[1])
                    if on_progress:
                        on_progress(prev[1])
                    return
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

            keys = [
                (cfg, mode, task, rep)
                for cfg in suite.agents
                for mode in (cfg.modes or suite.exposure.modes)
                for task in suite.tasks
                for rep in range(suite.repeats)
            ]
            # opened only once the servers are up, so a failed start never truncates resumable results;
            # runs kept from a resumed run are rewritten first (dropping stale and torn lines)
            if out_dir is not None:
                sink = _JsonlSink(suite, out_dir)
                for cfg, mode, task, rep in keys:
                    if (prev := done.get((cfg.id, mode, task.id, rep))) is not None:
                        sink.write(*prev)
            await asyncio.gather(*(one(*k) for k in keys))
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
    resumed = sum(1 for s in scores if _job_key(s) in done)
    return RunResult(
        suite,
        traces,
        scores,
        aggregate(scores),
        tool_count,
        catalogue_tokens,
        config_errors,
        resumed,
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
