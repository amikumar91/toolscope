"""Suite configuration: which MCP servers, which models/agents, which tasks, which exposure modes."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from .trace import SEARCH_TOOL

# `$$` -> literal `$`; `${VAR}` / `${VAR:-default}` -> environment value. A bare `$VAR` is left alone.
_VAR_RE = re.compile(r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
ExposureMode = Literal["all", "search"]
SchemaDialect = Literal["auto", "gemini", "none"]  # see toolscope.llm.sanitize_tools

_SERVER_INTERP_KEYS = ("env", "headers", "url", "command", "args")
_SECRET_KEYS = (
    "env",
    "headers",
    "url",
    "args",
)  # values expanded here are scrubbed from every output file
_MIN_SECRET_LEN = 4  # don't scrub trivially short values, they'd mangle unrelated text
# booleans and plain numbers (DEBUG=true, PORT=8080) are config, not credentials; scrubbing them mangles text
_TRIVIAL_RE = re.compile(r"(?i)(true|false|yes|no|on|off|null|none|[-+]?\d+(\.\d+)?)")
REDACTED = "***"
# credentials LiteLLM reads from the environment by itself (OPENAI_API_KEY, ANTHROPIC_API_BASE, ...): they
# never pass through `${VAR}` interpolation, but provider errors can quote them, so they are scrubbed too
_PROVIDER_ENV_RE = re.compile(r"[A-Z0-9_]*_(API_KEY|API_BASE|BASE_URL|API_TOKEN|ACCESS_TOKEN)")


def _is_secret_value(v: str) -> bool:
    return len(v) >= _MIN_SECRET_LEN and not _TRIVIAL_RE.fullmatch(v.strip())


def provider_env_secrets() -> set[str]:
    """Values of provider credential env vars (see `_PROVIDER_ENV_RE`), for redaction."""
    return {
        v for k, v in os.environ.items() if _PROVIDER_ENV_RE.fullmatch(k) and _is_secret_value(v)
    }


def interpolate(value: Any, owner: str, secrets: set[str] | None = None) -> Any:
    """Expand `${VAR}` / `${VAR:-default}` and `$$` in every string inside `value` (recursing into dicts/lists).

    `owner` (e.g. "server 'github'") prefixes the error for an unset variable without a default. Expanded
    values are added to `secrets` when given, so callers can scrub them from outputs.
    """
    if isinstance(value, dict):
        return {k: interpolate(v, owner, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, owner, secrets) for v in value]
    if not isinstance(value, str):
        return value

    def sub(m: re.Match[str]) -> str:
        if m.group(0) == "$$":
            return "$"
        name, default = m.group(1), m.group(2)
        env = os.environ.get(name)
        if env:
            if secrets is not None and _is_secret_value(env):
                secrets.add(env)
            return env
        if default is not None:
            return default
        if env is None:
            raise ValueError(f"{owner}: env var {name} is not set")
        return ""  # set but empty, no default

    return _VAR_RE.sub(sub, value)


def _interpolate_suite(data: dict[str, Any], secrets: set[str]) -> None:
    """Expand env vars in server connection settings and agent params, in place. Tasks are never touched."""
    for srv in data.get("servers") or []:
        if not isinstance(srv, dict):
            continue
        owner = f"server '{srv.get('name', '?')}'"
        for key in _SERVER_INTERP_KEYS:
            if key in srv:
                srv[key] = interpolate(srv[key], owner, secrets if key in _SECRET_KEYS else None)
    for agent in data.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        owner = f"agent '{agent.get('id', '?')}'"
        if "params" in agent:
            agent["params"] = interpolate(agent["params"], owner, secrets)
        if isinstance(
            agent.get("model"), str
        ):  # e.g. "${CLAUDE_MODEL:-anthropic/...}"; not a secret
            agent["model"] = interpolate(agent["model"], owner)


class ServerConfig(BaseModel):
    """One MCP server. Tools are namespaced as `<name>__<tool>` so multi-server suites never collide."""

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    call_timeout_s: float | None = (
        60.0  # per tool call; a timed-out call returns an error to the model
    )
    max_restarts: int = 1  # times a dead server is restarted before its calls fail as `infra`

    @model_validator(mode="after")
    def _check(self) -> ServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"server '{self.name}': stdio transport needs `command`")
        if self.transport == "http" and not self.url:
            raise ValueError(f"server '{self.name}': http transport needs `url`")
        if "__" in self.name:
            raise ValueError(f"server '{self.name}': name must not contain '__'")
        return self


class AgentConfig(BaseModel):
    """What drives the tools.

    kind=litellm  -> built-in tool loop; `model` is any LiteLLM model string
                     (e.g. anthropic/claude-sonnet-5-5, openai/gpt-6.1-sol, gemini/gemini-4-hydrogen,
                     ollama/qwen3). Keys and base URLs come from LiteLLM's env vars or `params`.
    kind=mock     -> offline deterministic model for demos/CI (`model`: mock/good | mock/sloppy).
    kind=python   -> your own agent: `ref` = "package.module:callable". See toolscope.agent.CustomAgent.
    """

    id: str
    kind: Literal["litellm", "mock", "python"] = "litellm"
    model: str | None = None
    ref: str | None = None
    temperature: float | None = 0.0  # null = don't send (provider default)
    max_turns: int = 8
    params: dict[str, Any] = Field(default_factory=dict)  # passed through to litellm / your agent
    # built-in loop: rewrite tool schemas for providers that reject some JSON Schema keywords.
    # auto = "gemini" when the model string names Gemini; changes are recorded in trace.schema_sanitized
    schema_dialect: SchemaDialect = "auto"
    modes: list[ExposureMode] | None = None  # override suite exposure modes for this agent
    # kind=python returning a history: name of the agent's own search meta-tool, if not `search_tools`
    search_tool: str = SEARCH_TOOL
    # built-in loop (litellm/mock): per model call timeout, and retries with exponential backoff + jitter
    # on rate limits, 5xx, timeouts and connection errors. Still failing -> the run is `infra`
    timeout_s: float | None = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 1.0  # base delay; attempt n waits uniform(0, base * 2**(n-1)), max 30s


class ExpectedCall(BaseModel):
    name: str  # bare tool name or "server__tool"
    args: dict[str, Any] = Field(default_factory=dict)  # subset match; values may be "re:<regex>"


class Expectation(BaseModel):
    tools: list[ExpectedCall] = Field(default_factory=list)
    ordered: bool = False  # expected tools must appear in this order
    no_tools: bool = False  # irrelevance task: correct behaviour is calling nothing
    answer_contains: list[str] = Field(default_factory=list)  # case-insensitive substrings
    answer_regex: str | None = None
    check: str | None = None  # "module:function(trace, task) -> bool" for end-state checks
    max_calls: int | None = None  # efficiency budget (defaults to len(tools) + 2)


class Task(BaseModel):
    id: str
    prompt: str
    system: str | None = None
    expect: Expectation = Field(default_factory=Expectation)
    forbid: list[str] = Field(
        default_factory=list
    )  # tools that must never be called (destructive, decoys)
    tags: list[str] = Field(default_factory=list)


class ExposureConfig(BaseModel):
    """How tools reach the model.

    all    -> every schema in context every turn (classic MCP; the 'bloat' baseline)
    search -> only a `search_tools` meta-tool up front; schemas are loaded on demand (tool-search / deferred loading)
    """

    modes: list[ExposureMode] = Field(default_factory=lambda: list[ExposureMode](["all"]))
    search_top_k: int = 5


class Suite(BaseModel):
    name: str = "toolscope-suite"
    servers: list[ServerConfig]
    agents: list[AgentConfig]
    tasks: list[Task] = Field(default_factory=list)
    tasks_file: str | None = None
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    repeats: int = 3  # k for pass@k / pass^k
    concurrency: int = 4
    run_timeout_s: float | None = (
        600.0  # per run (one agent x mode x task x repeat); exceeded -> `infra`
    )
    base_dir: Path = Path(".")
    _secrets: set[str] = PrivateAttr(default_factory=set)

    def redact(self, text: str) -> str:
        """Replace every env-var value expanded into server env/headers/url/args or agent params with `***`.

        For plain text (log lines, a single string value). Never run it over serialised JSON/HTML: use
        `redact_obj` on the data before serialising, so JSON syntax and non-string values are never touched.
        """
        # also the JSON-escaped spelling, for tool results that embed the secret in JSON text
        forms = {f for s in self._secrets for f in (s, json.dumps(s)[1:-1])}
        for s in sorted(forms, key=len, reverse=True):
            text = text.replace(s, REDACTED)
        return text

    def redact_obj(self, obj: Any) -> Any:
        """Return a copy of `obj` (e.g. a `model_dump(mode="json")`) with secrets scrubbed from string keys and values."""
        if not self._secrets:
            return obj
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {self.redact_obj(k): self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.redact_obj(v) for v in obj]
        return obj

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        secrets: set[str] = set()
        if isinstance(data, dict):
            _interpolate_suite(data, secrets)
        data["base_dir"] = path.parent.resolve()
        suite = cls.model_validate(data)
        if any(a.kind == "litellm" for a in suite.agents):
            secrets |= provider_env_secrets()
        suite._secrets = secrets
        if suite.tasks_file:
            tpath = (suite.base_dir / suite.tasks_file).resolve()
            raw = yaml.safe_load(tpath.read_text()) or []
            raw = raw.get("tasks", raw) if isinstance(raw, dict) else raw
            suite.tasks = suite.tasks + [Task.model_validate(t) for t in raw]
        if not suite.tasks:
            raise ValueError("suite has no tasks (set `tasks:` or `tasks_file:`)")
        return suite
