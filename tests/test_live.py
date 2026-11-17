"""Opt-in live provider tests (#4): `pytest -m live`. Skipped unless the provider is configured.

Runs the providers example suite (demo tasks + tricky-schema server) once per configured provider and
checks the plumbing, not model quality: no transport, auth or schema-rejection errors.
Configure with the env vars in examples/providers/.env.example (direct keys or a router).
"""

from __future__ import annotations

import os

import pytest

from conftest import DEMO_DIR
from toolscope.config import Suite
from toolscope.runner import run_suite

pytestmark = pytest.mark.live
SUITE = DEMO_DIR.parent / "providers" / "suite.yaml"
KEYS = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


def _configured(agent_id: str, model: str) -> bool:
    # a model routed as openai/... (OpenAI-format router) needs the OpenAI key, whatever the provider
    key = "OPENAI_API_KEY" if model.startswith("openai/") else KEYS[agent_id]
    return bool(os.environ.get(key))


@pytest.mark.parametrize("agent_id", list(KEYS))
async def test_provider_runs_cleanly(agent_id, tmp_path):
    pytest.importorskip("litellm")
    suite = Suite.load(SUITE)
    agent = next(a for a in suite.agents if a.id == agent_id)
    if not _configured(agent_id, agent.model or ""):
        pytest.skip(f"{agent_id}: no API key configured")
    suite.agents = [agent]
    suite.repeats = 1
    res = await run_suite(suite, out_dir=tmp_path)

    assert not res.config_errors
    bad = [
        (t.task_id, t.mode, t.infra_error or t.error)
        for t in res.traces
        if t.infra_error or (t.error and t.error != "max_turns")
    ]
    assert not bad, f"transport/schema/auth errors: {bad}"
    assert any(t.calls for t in res.traces), "model never called a tool"
    assert sum(s.success for s in res.scores) >= len(res.scores) // 2  # sanity, not a benchmark
