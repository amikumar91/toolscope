import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import DEMO_SUITE
from toolscope import SCHEMA_VERSION
from toolscope.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    r = runner.invoke(
        app,
        [
            "run",
            str(DEMO_SUITE),
            "-o",
            str(out),
            "-a",
            "mock-good",
            "-k",
            "1",
            "--fail-under",
            "1.0",
        ],
    )
    assert r.exit_code == 0, r.output
    return out, r.output


def test_run_writes_all_outputs(results):
    out, output = results
    for f in ("report.html", "summary.json", "scores.jsonl", "traces.jsonl", "servers.log"):
        assert (out / f).exists(), f
    summary = json.loads((out / "summary.json").read_text())
    assert summary["tool_count"] == 10
    assert {(g["agent_id"], g["mode"]) for g in summary["groups"]} == {
        ("mock-good", "all"),
        ("mock-good", "search"),
    }
    assert "tool-use effectiveness" in output and "FAIL" not in output


def test_mode_filter(tmp_path):
    r = runner.invoke(
        app,
        ["run", str(DEMO_SUITE), "-o", str(tmp_path), "-a", "mock-good", "-k", "1", "-m", "search"],
    )
    assert r.exit_code == 0, r.output
    groups = json.loads((tmp_path / "summary.json").read_text())["groups"]
    assert [g["mode"] for g in groups] == ["search"]


def test_fail_under_exits_nonzero(tmp_path):
    r = runner.invoke(
        app,
        [
            "run",
            str(DEMO_SUITE),
            "-o",
            str(tmp_path),
            "-a",
            "mock-sloppy",
            "-k",
            "2",
            "-m",
            "all",
            "--fail-under",
            "1.0",
        ],
    )
    assert r.exit_code == 1
    assert "FAIL" in r.output and "mock-sloppy/all" in r.output


def test_report_rerenders(results):
    out, _ = results
    (out / "report.html").unlink()
    r = runner.invoke(app, ["report", str(out)])
    assert r.exit_code == 0, r.output
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "demo-shop-crm" in html and "mock-good" in html


def test_outputs_carry_schema_version(results):
    out, _ = results
    assert json.loads((out / "summary.json").read_text())["schema_version"] == SCHEMA_VERSION
    for f in ("scores.jsonl", "traces.jsonl"):
        recs = [json.loads(ln) for ln in (out / f).read_text(encoding="utf-8").splitlines() if ln]
        assert recs and all(r["schema_version"] == SCHEMA_VERSION for r in recs), f


def _copy_results(src, dst, edit):
    """Copy summary.json + scores.jsonl, applying `edit(record)` to every JSON record."""
    dst.mkdir()
    summary = json.loads((src / "summary.json").read_text())
    edit(summary)
    (dst / "summary.json").write_text(json.dumps(summary))
    lines = []
    for ln in (src / "scores.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        edit(rec)
        lines.append(json.dumps(rec) + "\n")
    (dst / "scores.jsonl").write_text("".join(lines) + "\n", encoding="utf-8")


def test_report_reads_results_without_schema_version(results, tmp_path):
    legacy = tmp_path / "legacy"
    _copy_results(results[0], legacy, lambda r: r.pop("schema_version"))
    assert "schema_version" not in (legacy / "scores.jsonl").read_text()
    r = runner.invoke(app, ["report", str(legacy)])
    assert r.exit_code == 0, r.output
    assert "mock-good" in (legacy / "report.html").read_text(encoding="utf-8")


def test_report_accepts_newer_minor_version(results, tmp_path):
    newer = tmp_path / "minor"
    _copy_results(results[0], newer, lambda r: r.update(schema_version="1.7", added_later=True))
    assert runner.invoke(app, ["report", str(newer)]).exit_code == 0


@pytest.mark.parametrize("where", ["summary", "scores"])
def test_report_rejects_unknown_major_version(results, tmp_path, where):
    future = tmp_path / "future"

    def edit(r):
        if ("groups" in r) == (where == "summary"):
            r["schema_version"] = "2"

    _copy_results(results[0], future, edit)
    r = runner.invoke(app, ["report", str(future)])
    assert r.exit_code == 1 and not (future / "report.html").exists()
    assert "schema_version '2'" in r.output and "upgrade toolscope" in r.output
    assert ("summary.json" if where == "summary" else "scores.jsonl line 1") in r.output


def test_tools_lists_catalogue():
    r = runner.invoke(app, ["tools", str(DEMO_SUITE)])
    assert r.exit_code == 0, r.output
    assert (
        "10 tools" in r.output
        and "shop__get_order_status" in r.output
        and "catalogue total" in r.output
    )


WRAPPER = """
import sys
sys.path.insert(0, {demo!r})
from servers import shop
print("booting with token " + sys.argv[1], file=sys.stderr, flush=True)
shop().run()
"""


def test_secrets_never_reach_outputs(tmp_path, monkeypatch):
    """Regression for #30: DEBUG=true / PORT=8080 used to corrupt JSON; args secrets and servers.log leaked."""
    secret = "tok-9f8e7d6c"
    monkeypatch.setenv("TS_SECRET", secret)
    monkeypatch.setenv("TS_DEBUG", "true")
    monkeypatch.setenv("TS_PORT", "8080")
    (tmp_path / "wrapper.py").write_text(WRAPPER.format(demo=str(DEMO_SUITE.parent)))
    (tmp_path / "suite.yaml").write_text(f"""
name: redaction
servers:
  - name: shop
    command: python
    args: [wrapper.py, "${{TS_SECRET}}"]
    env: {{DEBUG: "${{TS_DEBUG}}", PORT: "${{TS_PORT}}", API_KEY: "${{TS_SECRET}}"}}
agents:
  - {{id: good, kind: mock, model: mock/good}}
tasks:
  - id: status
    prompt: "What's the shipping status of order ORD-1042? My key is {secret}, port 8080, debug true."
    expect: {{tools: [{{name: get_order_status, args: {{order_id: ORD-1042}}}}]}}
""")
    out = tmp_path / "out"
    r = runner.invoke(
        app, ["run", str(tmp_path / "suite.yaml"), "-o", str(out), "-k", "1", "-m", "all"]
    )
    assert r.exit_code == 0, r.output
    assert "booting with token ***" in (out / "servers.log").read_text(encoding="utf-8")
    for f in out.iterdir():
        assert secret not in f.read_text(encoding="utf-8"), f.name
    for f in ("scores.jsonl", "traces.jsonl"):
        lines = [ln for ln in (out / f).read_text(encoding="utf-8").splitlines() if ln]
        assert lines and all(json.loads(ln) for ln in lines), f
    assert json.loads((out / "summary.json").read_text())["groups"][0]["runs"] == 1
    r = runner.invoke(app, ["report", str(out)])
    assert r.exit_code == 0, r.output


# ---- --resume and infra errors (#5) ----------------------------------------------------------
def test_resume_finishes_remaining_runs(results, tmp_path):
    src, _ = results
    out = tmp_path / "resumed"
    out.mkdir()
    for f in ("traces.jsonl", "scores.jsonl"):
        lines = (src / f).read_text(encoding="utf-8").splitlines()
        (out / f).write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    args = ["run", str(DEMO_SUITE), "-o", str(out), "-a", "mock-good", "-k", "1"]
    r = runner.invoke(app, [*args, "--resume", "--fail-under", "1.0"])
    assert r.exit_code == 0, r.output
    assert "resumed 5 completed run(s)" in r.output
    clean = json.loads((src / "summary.json").read_text())["groups"]
    got = json.loads((out / "summary.json").read_text())["groups"]
    drop = lambda gs: [{k: v for k, v in g.items() if k != "p50_latency_ms"} for g in gs]  # noqa: E731
    assert drop(got) == drop(clean)


def test_resume_refuses_unknown_schema(tmp_path):
    (tmp_path / "scores.jsonl").write_text('{"schema_version": "9"}\n', encoding="utf-8")
    r = runner.invoke(app, ["run", str(DEMO_SUITE), "-o", str(tmp_path), "--resume"])
    assert r.exit_code == 1 and "cannot resume" in r.output


def test_infra_errors_are_reported(tmp_path):
    flaky = Path(__file__).with_name("flaky_server.py")
    (tmp_path / "suite.yaml").write_text(
        f"""
servers: [{{name: flaky, command: {json.dumps(sys.executable)}, args: [{json.dumps(str(flaky))}], max_restarts: 0}}]
agents: [{{id: good, kind: mock, model: mock/good}}]
repeats: 1
tasks:
  - {{id: die, prompt: "please die", expect: {{tools: [{{name: die}}]}}}}
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    r = runner.invoke(app, ["run", str(tmp_path / "suite.yaml"), "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert "1 run(s) hit infrastructure errors" in " ".join(r.output.split())
    (g,) = json.loads((out / "summary.json").read_text())["groups"]
    assert (g["runs"], g["infra_errors"]) == (0, 1)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "1 infra-error runs excluded" in html and "+1 infra (excluded)" in html
