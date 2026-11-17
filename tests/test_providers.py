"""Provider support (#4): schema sanitising, LiteLLM param handling, cost, model interpolation."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from conftest import FakePool, spec
from toolscope import Task
from toolscope.agent import run_builtin
from toolscope.config import AgentConfig, Suite
from toolscope.llm import LiteLLMClient, LLMResponse, resolve_dialect, sanitize_tools


def _tool(params: dict, name: str = "t") -> dict:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": params}}


def _gemini(params: dict) -> tuple[dict, list[str]]:
    (out,), changed = sanitize_tools([_tool(params)], "gemini")
    return out["function"]["parameters"], changed.get("t", [])


# ---- dialect ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini/gemini-4-hydrogen", "gemini"),
        ("vertex_ai/gemini-4-argon", "gemini"),
        ("openai/google/gemini-4-hydrogen", "gemini"),  # via an OpenAI-format router
        ("anthropic/claude-sonnet-5-5", "none"),
        ("openai/gpt-6.1-sol", "none"),
        (None, "none"),
    ],
)
def test_resolve_dialect_auto(model, expected):
    assert resolve_dialect("auto", model) == expected


def test_resolve_dialect_explicit_wins():
    assert resolve_dialect("none", "gemini/x") == "none"
    assert resolve_dialect("gemini", "openai/my-proxy-model") == "gemini"


# ---- gemini sanitiser ------------------------------------------------------------------------
def test_clean_schema_is_untouched():
    params = {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}
    tools = [_tool(params)]
    out, changed = sanitize_tools(tools, "gemini")
    assert out[0] is tools[0] and changed == {}


def test_dialect_none_is_a_no_op():
    tools = [_tool({"type": "object", "additionalProperties": False})]
    assert sanitize_tools(tools, "none") == (tools, {})


def test_drops_unsupported_keywords():
    out, notes = _gemini(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"type": "integer", "default": 3, "examples": [1]}},
        }
    )
    assert out == {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert {"$schema removed", "additionalProperties removed", "default removed"} <= set(notes)


def test_inlines_refs_and_removes_defs():
    out, notes = _gemini(
        {
            "type": "object",
            "$defs": {"Pri": {"type": "string", "enum": ["low", "high"]}},
            "properties": {"p": {"$ref": "#/$defs/Pri", "description": "priority"}},
        }
    )
    assert out == {
        "type": "object",
        "properties": {"p": {"type": "string", "enum": ["low", "high"], "description": "priority"}},
    }
    assert "$ref inlined" in notes and "$defs removed (refs inlined)" in notes


def test_recursive_ref_becomes_object():
    out, notes = _gemini(
        {
            "type": "object",
            "definitions": {
                "Node": {"type": "object", "properties": {"child": {"$ref": "#/definitions/Node"}}}
            },
            "properties": {"root": {"$ref": "#/definitions/Node"}},
        }
    )
    assert out["properties"]["root"]["properties"]["child"] == {"type": "object"}
    assert "$ref unresolvable or recursive: dropped" in notes


def test_nullable_forms():
    out, notes = _gemini(
        {
            "type": "object",
            "properties": {
                "a": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "A"},
                "b": {"type": ["integer", "null"]},
                "c": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
        }
    )
    p = out["properties"]
    assert p["a"] == {"type": "string", "title": "A", "nullable": True}
    assert p["b"] == {"type": "integer", "nullable": True}
    assert p["c"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert "null branch -> nullable" in notes and "type list -> type + nullable" in notes


def test_const_oneof_allof_bounds_format():
    out, notes = _gemini(
        {
            "type": "object",
            "properties": {
                "kind": {"const": "bug"},
                "x": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                "n": {"type": "integer", "exclusiveMinimum": 0},
                "url": {"type": "string", "format": "uri"},
                "when": {"type": "string", "format": "date-time"},
                "m": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"a": {"type": "string"}},
                            "required": ["a"],
                        },
                        {"properties": {"b": {"type": "string"}}},
                    ]
                },
                "tags": {"type": "array", "items": {"type": "string", "const": "x"}},
            },
        }
    )
    p = out["properties"]
    assert p["kind"] == {"enum": ["bug"]}
    assert p["x"] == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert p["n"] == {"type": "integer", "minimum": 0}
    assert p["url"] == {"type": "string"} and p["when"]["format"] == "date-time"
    assert p["m"]["type"] == "object" and set(p["m"]["properties"]) == {"a", "b"}
    assert p["m"]["required"] == ["a"]
    assert p["tags"]["items"] == {"type": "string", "enum": ["x"]}
    for n in ("const -> enum", "oneOf -> anyOf", "allOf merged", "exclusiveMinimum -> minimum"):
        assert n in notes


def test_sanitise_does_not_mutate_input():
    params = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
    _gemini(params)
    assert params["properties"]["a"] == {"type": "string", "default": "x"}


# ---- built-in loop records what was sanitised ------------------------------------------------
class CaptureLLM:
    def __init__(self):
        self.tools: list[list[dict]] = []

    async def complete(self, messages, tools, **_: Any) -> LLMResponse:
        self.tools.append(tools)
        return LLMResponse("done", [], 10, 5)


TRICKY = spec("shop", "make", "Make a thing", {"v": {"type": "string", "default": "x"}})
TASK = Task.model_validate({"id": "t", "prompt": "make it", "expect": {"no_tools": True}})


@pytest.mark.parametrize("mode", ["all", "search"])
async def test_run_builtin_sanitises_for_gemini_and_records_it(mode):
    cfg = AgentConfig(id="g", kind="litellm", model="gemini/gemini-4-hydrogen")
    llm = CaptureLLM()
    tr = await run_builtin(FakePool([TRICKY]), TASK, cfg, llm, mode, 0)
    if mode == "all":
        sent = next(t for t in llm.tools[0] if t["function"]["name"] == "shop__make")
        assert "default" not in sent["function"]["parameters"]["properties"]["v"]
        assert tr.schema_sanitized == {"shop__make": ["default removed"]}
    else:  # only search_tools was visible, and it is already clean
        assert tr.schema_sanitized == {}


async def test_run_builtin_leaves_other_providers_alone():
    for cfg in (
        AgentConfig(id="c", kind="litellm", model="anthropic/claude-sonnet-5-5"),
        AgentConfig(id="g", kind="litellm", model="gemini/x", schema_dialect="none"),
        AgentConfig(id="m", kind="mock", model="mock/good"),
    ):
        llm = CaptureLLM()
        tr = await run_builtin(FakePool([TRICKY]), TASK, cfg, llm, "all", 0)
        assert tr.schema_sanitized == {}
        assert llm.tools[0][0]["function"]["parameters"]["properties"]["v"]["default"] == "x"


# ---- LiteLLMClient (fake litellm module: CI has no litellm installed) ------------------------
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _resp(hidden_cost=None):
    tc = _Obj(id="c1", function=_Obj(name="f", arguments='{"a": 1}'))
    r = _Obj(
        choices=[_Obj(message=_Obj(content=None, tool_calls=[tc]))],
        usage=_Obj(prompt_tokens=7, completion_tokens=3),
    )
    r._hidden_params = {} if hidden_cost is None else {"response_cost": hidden_cost}
    return r


@pytest.fixture
def fake_litellm(monkeypatch):
    mod = types.ModuleType("litellm")
    mod.calls = []  # type: ignore[attr-defined]
    mod.next = _resp()  # type: ignore[attr-defined]
    mod.price = None  # type: ignore[attr-defined]

    async def acompletion(**kw):
        mod.calls.append(kw)  # type: ignore[attr-defined]
        return mod.next  # type: ignore[attr-defined]

    def completion_cost(completion_response):
        if mod.price is None:  # type: ignore[attr-defined]
            raise ValueError("This model isn't mapped yet")
        return mod.price  # type: ignore[attr-defined]

    mod.acompletion = acompletion  # type: ignore[attr-defined]
    mod.completion_cost = completion_cost  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


async def test_litellm_drops_empty_params_and_none_temperature(fake_litellm):
    c = LiteLLMClient("openai/x", None, api_base="", api_key=None, top_p=0.9)
    r = await c.complete([{"role": "user", "content": "hi"}], [], seed=3)
    kw = fake_litellm.calls[0]
    assert "api_base" not in kw and "api_key" not in kw and "temperature" not in kw
    assert "seed" not in kw and kw["top_p"] == 0.9 and kw["tools"] is None
    assert r.tool_calls[0].name == "f" and (r.input_tokens, r.output_tokens) == (7, 3)


async def test_litellm_sends_temperature_and_base(fake_litellm):
    c = LiteLLMClient("anthropic/x", 1.0, api_base="https://router.example")
    await c.complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    kw = fake_litellm.calls[0]
    assert kw["temperature"] == 1.0 and kw["api_base"] == "https://router.example"


@pytest.mark.parametrize(
    "hidden,price,expected",
    [(0.0012, None, 0.0012), (None, 0.5, 0.5), (None, None, None), (0.0, None, 0.0)],
)
async def test_litellm_cost(fake_litellm, hidden, price, expected):
    fake_litellm.next, fake_litellm.price = _resp(hidden), price
    r = await LiteLLMClient("openai/x").complete([{"role": "user", "content": "hi"}], [])
    assert r.cost_usd == expected  # unknown price -> None (shown as n/a), never a fake 0


# ---- config ------------------------------------------------------------------------------------
def _suite(tmp_path, agent_yaml: str):
    (tmp_path / "s.yaml").write_text(
        "servers: []\nagents:\n  - " + agent_yaml + "\ntasks: [{id: t, prompt: p}]\n"
    )
    return Suite.load(tmp_path / "s.yaml")


def test_model_is_interpolated_and_not_treated_as_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODEL", "openai/google/gemini-4-hydrogen")
    s = _suite(tmp_path, '{id: g, kind: litellm, model: "${MY_MODEL:-gemini/x}"}')
    assert s.agents[0].model == "openai/google/gemini-4-hydrogen"
    assert "gemini" in s.redact("model openai/google/gemini-4-hydrogen")
    monkeypatch.delenv("MY_MODEL")
    assert _suite(tmp_path, '{id: g, model: "${MY_MODEL:-gemini/x}"}').agents[0].model == "gemini/x"


def test_optional_base_url_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("ROUTER_BASE", raising=False)
    s = _suite(tmp_path, '{id: a, model: openai/x, params: {api_base: "${ROUTER_BASE:-}"}}')
    assert s.agents[0].params == {
        "api_base": ""
    }  # dropped by LiteLLMClient, so LiteLLM env applies


def test_temperature_null_and_dialect(tmp_path):
    a = _suite(
        tmp_path, "{id: a, model: openai/x, temperature: null, schema_dialect: gemini}"
    ).agents[0]
    assert a.temperature is None and a.schema_dialect == "gemini"
    with pytest.raises(ValueError):
        _suite(tmp_path, "{id: a, model: openai/x, schema_dialect: openapi}")


def test_providers_example_suite_loads():
    from conftest import DEMO_DIR

    s = Suite.load(DEMO_DIR.parent / "providers" / "suite.yaml")
    assert {a.id for a in s.agents} == {"claude", "openai", "gemini"}
    assert all(a.temperature and a.temperature > 0 for a in s.agents) and s.repeats >= 3
    assert any(t.id == "add-catalogue-item" for t in s.tasks)


async def test_schema_server_tools_are_sanitised_for_gemini():
    """The example server's real (pydantic-generated) schema: refs, nullable, const, uri format."""
    import json

    from conftest import DEMO_DIR
    from toolscope.mcp_pool import MCPPool

    s = Suite.load(DEMO_DIR.parent / "providers" / "suite.yaml")
    servers = [x for x in s.servers if x.name == "schema"]
    async with MCPPool(servers, s.base_dir) as pool:
        (out,), changed = sanitize_tools(
            [pool.tools["schema__add_catalogue_item"].openai()], "gemini"
        )
    text = json.dumps(out)
    for kw in ("$ref", "$defs", "const", "default", '"null"', "uri"):
        assert kw not in text
    assert (
        out["function"]["parameters"]["properties"]["item"]["properties"]["dimensions"]["type"]
        == "object"
    )
    assert {"$ref inlined", "const -> enum", "null branch -> nullable"} <= set(
        changed["schema__add_catalogue_item"]
    )


def test_provider_env_credentials_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-router-secret-123")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://router.internal.example")
    monkeypatch.setenv("OPENAI_API_VERSION", "2026-01-01")  # not a credential name: kept
    s = _suite(tmp_path, "{id: a, kind: litellm, model: openai/x}")
    msg = "401 from https://router.internal.example for key sk-router-secret-123 (2026-01-01)"
    assert s.redact(msg) == "401 from *** for key *** (2026-01-01)"
    mock_only = _suite(tmp_path, "{id: m, kind: mock}")  # no provider agents: nothing to scrub
    assert mock_only.redact(msg) == msg
