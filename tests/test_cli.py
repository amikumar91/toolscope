import json

import pytest
from typer.testing import CliRunner

from conftest import DEMO_SUITE
from toolscope.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    r = runner.invoke(app, ["run", str(DEMO_SUITE), "-o", str(out), "-a", "mock-good", "-k", "1",
                            "--fail-under", "1.0"])
    assert r.exit_code == 0, r.output
    return out, r.output


def test_run_writes_all_outputs(results):
    out, output = results
    for f in ("report.html", "summary.json", "scores.jsonl", "traces.jsonl", "servers.log"):
        assert (out / f).exists(), f
    summary = json.loads((out / "summary.json").read_text())
    assert summary["tool_count"] == 10
    assert {(g["agent_id"], g["mode"]) for g in summary["groups"]} == {("mock-good", "all"), ("mock-good", "search")}
    assert "tool-use effectiveness" in output and "FAIL" not in output


def test_mode_filter(tmp_path):
    r = runner.invoke(app, ["run", str(DEMO_SUITE), "-o", str(tmp_path), "-a", "mock-good", "-k", "1", "-m", "search"])
    assert r.exit_code == 0, r.output
    groups = json.loads((tmp_path / "summary.json").read_text())["groups"]
    assert [g["mode"] for g in groups] == ["search"]


def test_fail_under_exits_nonzero(tmp_path):
    r = runner.invoke(app, ["run", str(DEMO_SUITE), "-o", str(tmp_path), "-a", "mock-sloppy", "-k", "2",
                            "-m", "all", "--fail-under", "1.0"])
    assert r.exit_code == 1
    assert "FAIL" in r.output and "mock-sloppy/all" in r.output


def test_report_rerenders(results):
    out, _ = results
    (out / "report.html").unlink()
    r = runner.invoke(app, ["report", str(out)])
    assert r.exit_code == 0, r.output
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "demo-shop-crm" in html and "mock-good" in html


def test_tools_lists_catalogue():
    r = runner.invoke(app, ["tools", str(DEMO_SUITE)])
    assert r.exit_code == 0, r.output
    assert "10 tools" in r.output and "shop__get_order_status" in r.output and "catalogue total" in r.output
