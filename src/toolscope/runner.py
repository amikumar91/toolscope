"""Run agents x exposure modes x tasks x repeats against a live MCP pool."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent import AgentFn, load_ref, run_builtin, run_custom
from .config import AgentConfig, Suite, Task
from .llm import make_llm
from .mcp_pool import MCPPool
from .metrics import GroupStats, RunScore, aggregate, score
from .search import BM25Retriever
from .trace import Trace


@dataclass
class RunResult:
    suite: Suite
    traces: list[Trace]
    scores: list[RunScore]
    stats: list[GroupStats]
    tool_count: int
    catalogue_tokens: int

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        redact = self.suite.redact
        (out / "traces.jsonl").write_text(redact("\n".join(t.model_dump_json() for t in self.traces) + "\n"))
        (out / "scores.jsonl").write_text(redact("\n".join(s.model_dump_json() for s in self.scores) + "\n"))
        (out / "summary.json").write_text(redact(json.dumps({
            "suite": self.suite.name, "tool_count": self.tool_count, "catalogue_tokens": self.catalogue_tokens,
            "groups": [g.model_dump() for g in self.stats]}, indent=2)))
        return out


async def run_suite(suite: Suite, on_progress: Callable[[RunScore], None] | None = None,
                    agent_fns: dict[str, AgentFn] | None = None, server_log: str | None = None) -> RunResult:
    """`agent_fns` lets Python callers pass agent callables directly instead of `ref:` strings."""
    from .tokens import count

    agent_fns = dict(agent_fns or {})
    async with MCPPool(suite.servers, base_dir=str(suite.base_dir), errlog_path=server_log) as pool:
        schemas = {q: s.input_schema for q, s in pool.tools.items()}
        catalogue_tokens = count([s.openai() for s in pool.tools.values()])
        retriever = BM25Retriever(list(pool.tools.values()))
        sem = asyncio.Semaphore(max(1, suite.concurrency))
        traces: list[Trace] = []
        scores: list[RunScore] = []

        async def one(cfg: AgentConfig, mode: str, task: Task, rep: int):
            async with sem:
                if cfg.kind == "python":
                    fn = agent_fns.get(cfg.id) or load_ref(cfg.ref or "", str(suite.base_dir))
                    agent_fns[cfg.id] = fn
                    tr = await run_custom(pool, task, cfg, fn, mode, rep, suite.exposure.search_top_k, retriever)
                else:
                    llm = make_llm(cfg.kind, cfg.model, cfg.temperature, cfg.params)
                    tr = await run_builtin(pool, task, cfg, llm, mode, rep, suite.exposure.search_top_k, retriever)
                sc = score(tr, task, schemas)
                traces.append(tr)
                scores.append(sc)
                if on_progress:
                    on_progress(sc)

        jobs = [one(cfg, mode, task, rep)
                for cfg in suite.agents for mode in (cfg.modes or suite.exposure.modes)
                for task in suite.tasks for rep in range(suite.repeats)]
        await asyncio.gather(*jobs)
        tool_count = len(pool.tools)

    order = {(a.id, m, t.id): i for i, (a, m, t) in enumerate(
        (a, m, t) for a in suite.agents for m in (a.modes or suite.exposure.modes) for t in suite.tasks)}
    key = lambda x: (order.get((x.agent_id, x.mode, x.task_id), 0), x.repeat)
    traces.sort(key=key)
    scores.sort(key=key)
    return RunResult(suite, traces, scores, aggregate(scores), tool_count, catalogue_tokens)


def score_messages(messages: list[dict], task: Task, schemas: dict[str, dict] | None = None,
                   agent_id: str = "external") -> RunScore:
    """Offline scoring of an already-recorded OpenAI-format conversation (no MCP connection needed)."""
    tr = Trace.from_openai_messages(messages, task_id=task.id, agent_id=agent_id,
                                    exposed_tools=list(schemas) if schemas else None)
    return score(tr, task, schemas or {})
