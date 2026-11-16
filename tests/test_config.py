import json
from pathlib import Path

import pytest

from toolscope import Suite, Trace
from toolscope.config import interpolate
from toolscope.metrics import aggregate
from toolscope.runner import RunResult
from toolscope.trace import ToolCall

SUITE = """
name: s
servers:
  - name: github
    transport: http
    url: "${GH_URL:-https://api.example.com/mcp/}"
    headers: {Authorization: "Bearer ${TS_TOKEN}"}
  - name: local
    command: "${TS_PY:-python}"
    args: [servers.py, "${TS_ARG}"]
    env: {API_KEY: "${TS_TOKEN}", PRICE: "$$5"}
agents:
  - {id: a, kind: mock, model: mock/good, params: {api_key: "${TS_TOKEN}", nested: ["${TS_PY:-py3}"]}}
tasks:
  - {id: t, prompt: "costs $5, see ${TS_TOKEN}", expect: {answer_contains: ["${TS_TOKEN}"]}}
"""


def load(tmp_path: Path, text: str = SUITE) -> Suite:
    p = tmp_path / "suite.yaml"
    p.write_text(text)
    return Suite.load(p)


@pytest.fixture
def env(monkeypatch):
    for k in ("GH_URL", "TS_PY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TS_TOKEN", "s3cr3t-token")
    monkeypatch.setenv("TS_ARG", "shop")
    return monkeypatch


def test_set_vars_are_expanded(tmp_path, env):
    s = load(tmp_path)
    gh, local = s.servers
    assert gh.headers == {"Authorization": "Bearer s3cr3t-token"}
    assert local.args == ["servers.py", "shop"]
    assert local.env == {"API_KEY": "s3cr3t-token", "PRICE": "$5"}
    assert s.agents[0].params == {"api_key": "s3cr3t-token", "nested": ["py3"]}


def test_defaults_used_when_unset_or_empty(tmp_path, env):
    env.setenv("TS_PY", "")
    s = load(tmp_path)
    assert s.servers[0].url == "https://api.example.com/mcp/"
    assert s.servers[1].command == "python"
    env.setenv("GH_URL", "http://localhost:9000/mcp")
    assert load(tmp_path).servers[0].url == "http://localhost:9000/mcp"


def test_unset_var_without_default_raises(tmp_path, env):
    env.delenv("TS_TOKEN")
    with pytest.raises(ValueError, match="server 'github': env var TS_TOKEN is not set"):
        load(tmp_path)


def test_unset_var_in_agent_params_names_agent(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    text = """
servers: [{name: x, command: python}]
agents: [{id: bot, kind: mock, model: mock/good, params: {api_key: "${NOPE_KEY}"}}]
tasks: [{id: t, prompt: p}]
"""
    with pytest.raises(ValueError, match="agent 'bot': env var NOPE_KEY is not set"):
        load(tmp_path, text)


def test_escape_and_bare_dollar():
    assert interpolate("$${TS_TOKEN} costs $$5 and $HOME", "x") == "${TS_TOKEN} costs $5 and $HOME"


def test_tasks_are_left_untouched(tmp_path, env):
    t = load(tmp_path).tasks[0]
    assert t.prompt == "costs $5, see ${TS_TOKEN}"
    assert t.expect.answer_contains == ["${TS_TOKEN}"]


def test_secrets_scrubbed_from_outputs(tmp_path, env):
    s = load(tmp_path)
    leak = "s3cr3t-token"
    trace = Trace(task_id="t", agent_id="a", final_answer=f"token is {leak}", error=f"401 for Bearer {leak}",
                  calls=[ToolCall(name="local__x", ok=False, error=f'header "{leak}" rejected')])
    from toolscope import score
    sc = score(trace, s.tasks[0], {})
    out = RunResult(s, [trace], [sc], aggregate([sc]), 1, 10).save(tmp_path / "out")
    for f in ("traces.jsonl", "scores.jsonl", "summary.json"):
        assert leak not in (out / f).read_text()
    assert "***" in (out / "traces.jsonl").read_text()
    json.loads((out / "summary.json").read_text())  # still valid JSON
    assert s.redact(f"<td>{leak}</td>") == "<td>***</td>"
    # non-secret expansions (args, url, command) are not scrubbed
    assert s.redact("shop python") == "shop python"
