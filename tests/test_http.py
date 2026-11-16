"""End-to-end over `transport: http` against a local streamable-HTTP server (no network beyond loopback)."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from toolscope import Suite, run_suite


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("http_server.py")), str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail("HTTP MCP server exited early")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("HTTP MCP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()


async def test_one_task_suite_over_http(tmp_path, http_url):
    p = tmp_path / "suite.yaml"
    p.write_text(f"""
name: http-suite
servers: [{{name: shop, transport: http, url: "{http_url}", headers: {{X-Test: "1"}}}}]
agents: [{{id: good, kind: mock, model: mock/good}}]
exposure: {{modes: [all, search]}}
repeats: 1
tasks:
  - id: order-status
    prompt: "What's the shipping status of order ORD-1042?"
    expect: {{tools: [{{name: get_order_status, args: {{order_id: ORD-1042}}}}], answer_contains: [shipped]}}
""")
    res = await run_suite(Suite.load(p))
    assert res.tool_count == 6
    assert all(g.pass_at_1 == 1.0 for g in res.stats), [s.failure_reason for s in res.scores]
    call = next(c for c in res.traces[0].real_calls)
    assert call.name == "shop__get_order_status" and "shipped" in call.result_preview
