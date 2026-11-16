"""Use toolscope as a regression gate in your own test suite: fail the build when a tool-description
change or model upgrade drops consistency.

    pytest examples/integrations/pytest_regression.py
"""

from pathlib import Path

import pytest

from toolscope import Suite, run_suite

SUITE = Path(__file__).parents[1] / "demo" / "suite.yaml"


@pytest.mark.asyncio
async def test_tool_use_does_not_regress():
    suite = Suite.load(SUITE)
    suite.agents = [a for a in suite.agents if a.id == "mock-good"]  # point at your real agent id
    result = await run_suite(suite)
    for g in result.stats:
        assert g.pass_hat_k >= 0.9, (
            f"{g.agent_id}/{g.mode}: pass^k {g.pass_hat_k:.0%} (funnel {g.funnel})"
        )
        assert (g.recall_at_k or 1) >= 0.95, f"{g.agent_id}/{g.mode}: tool search recall dropped"
