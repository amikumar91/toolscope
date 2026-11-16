"""Job isolation and incremental output in run_suite (#31)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from conftest import DEMO_SUITE
from toolscope import Suite, run_suite
from toolscope import runner as runner_mod
from toolscope.cli import app
from toolscope.config import AgentConfig
from toolscope.metrics import RunScore
from toolscope.trace import Trace

BAD_REF = "no_such_module_toolscope_31:agent"


def demo_suite(*agents: AgentConfig) -> Suite:
    suite = Suite.load(DEMO_SUITE)
    suite.repeats = 1
    suite.agents = [a for a in suite.agents if a.id == "mock-good"] + list(agents)
    return suite


async def test_bad_ref_does_not_abort_run(tmp_path):
    bad = AgentConfig(id="broken", kind="python", ref=BAD_REF, modes=["all"])
    suite = demo_suite(bad)
    res = await run_suite(suite, out_dir=tmp_path)

    good = [s for s in res.scores if s.agent_id == "mock-good"]
    assert len(good) == 2 * len(suite.tasks) and all(s.success for s in good)

    broken = [s for s in res.scores if s.agent_id == "broken"]
    assert len(broken) == len(suite.tasks)
    assert all(s.failure_stage == "crash" for s in broken)
    assert all("ModuleNotFoundError" in s.failure_reason for s in broken)
    assert all(t.error.startswith("config_error:") for t in res.traces if t.agent_id == "broken")
    assert list(res.config_errors) == ["broken"]
    assert "ModuleNotFoundError" in res.config_errors["broken"]

    # incremental files hold one parseable line per job
    lines = (tmp_path / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(res.scores)
    for line in lines:
        RunScore.model_validate_json(line)


async def test_unexpected_job_exception_is_recorded_as_crash(monkeypatch):
    real = runner_mod.run_builtin

    async def flaky(pool, task, cfg, llm, mode, rep, *a):
        if mode == "search":
            raise RuntimeError("boom")
        return await real(pool, task, cfg, llm, mode, rep, *a)

    monkeypatch.setattr(runner_mod, "run_builtin", flaky)
    suite = demo_suite()
    res = await run_suite(suite)
    by_mode = {m: [s for s in res.scores if s.mode == m] for m in ("all", "search")}
    assert all(s.success for s in by_mode["all"])
    assert len(by_mode["search"]) == len(suite.tasks)
    assert all(
        s.failure_stage == "crash" and "RuntimeError: boom" in s.failure_reason
        for s in by_mode["search"]
    )
    assert res.config_errors == {}


async def test_results_are_flushed_as_each_job_finishes(tmp_path):
    seen: list[int] = []

    def check(_: RunScore) -> None:
        # the line for the job that just finished is already on disk and parseable
        for name, model in (("traces", Trace), ("scores", RunScore)):
            lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            for line in lines:
                model.model_validate_json(line)
        seen.append(len(lines))

    suite = demo_suite()
    await run_suite(suite, on_progress=check, out_dir=tmp_path)
    assert seen == list(range(1, 2 * len(suite.tasks) + 1))


def test_cli_reports_config_error_once_and_exits_nonzero(tmp_path):
    suite_file = tmp_path / "suite.yaml"
    text = DEMO_SUITE.read_text(encoding="utf-8")
    text = text.replace("tasks_file: tasks.yaml", f"tasks_file: {DEMO_SUITE.parent / 'tasks.yaml'}")
    text = text.replace('ref: "my_agent:agent"', f'ref: "{BAD_REF}"')
    text = text.replace("servers.py", str(DEMO_SUITE.parent / "servers.py").replace("\\", "/"))
    suite_file.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    r = CliRunner().invoke(
        app,
        ["run", str(suite_file), "-o", str(out), "-a", "mock-good", "-a", "my-agent", "-k", "1"],
    )
    assert r.exit_code == 1, r.output
    assert r.output.count("CONFIG ERROR") == 1 and "my-agent" in r.output
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    groups = {(g["agent_id"], g["mode"]): g for g in summary["groups"]}
    assert groups[("mock-good", "all")]["pass_at_1"] == 1.0
    assert groups[("my-agent", "all")]["pass_at_1"] == 0.0
    assert (out / "report.html").exists()
