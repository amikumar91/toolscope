"""Suite configuration: which MCP servers, which models/agents, which tasks, which exposure modes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    """One MCP server. Tools are namespaced as `<name>__<tool>` so multi-server suites never collide."""

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    @model_validator(mode="after")
    def _check(self) -> "ServerConfig":
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
                     (e.g. anthropic/claude-sonnet-4-5, openai/gpt-4.1, ollama/qwen3).
    kind=mock     -> offline deterministic model for demos/CI (`model`: mock/good | mock/sloppy).
    kind=python   -> your own agent: `ref` = "package.module:callable". See toolscope.agent.CustomAgent.
    """

    id: str
    kind: Literal["litellm", "mock", "python"] = "litellm"
    model: str | None = None
    ref: str | None = None
    temperature: float = 0.0
    max_turns: int = 8
    params: dict[str, Any] = Field(default_factory=dict)  # passed through to litellm / your agent
    modes: list[Literal["all", "search"]] | None = None   # override suite exposure modes for this agent


class ExpectedCall(BaseModel):
    name: str                                   # bare tool name or "server__tool"
    args: dict[str, Any] = Field(default_factory=dict)  # subset match; values may be "re:<regex>"


class Expectation(BaseModel):
    tools: list[ExpectedCall] = Field(default_factory=list)
    ordered: bool = False                       # expected tools must appear in this order
    no_tools: bool = False                      # irrelevance task: correct behaviour is calling nothing
    answer_contains: list[str] = Field(default_factory=list)  # case-insensitive substrings
    answer_regex: str | None = None
    check: str | None = None                    # "module:function(trace, task) -> bool" for end-state checks
    max_calls: int | None = None                # efficiency budget (defaults to len(tools) + 2)


class Task(BaseModel):
    id: str
    prompt: str
    system: str | None = None
    expect: Expectation = Field(default_factory=Expectation)
    forbid: list[str] = Field(default_factory=list)  # tools that must never be called (destructive, decoys)
    tags: list[str] = Field(default_factory=list)


class ExposureConfig(BaseModel):
    """How tools reach the model.

    all    -> every schema in context every turn (classic MCP; the 'bloat' baseline)
    search -> only a `search_tools` meta-tool up front; schemas are loaded on demand (tool-search / deferred loading)
    """

    modes: list[Literal["all", "search"]] = Field(default_factory=lambda: ["all"])
    search_top_k: int = 5


class Suite(BaseModel):
    name: str = "toolscope-suite"
    servers: list[ServerConfig]
    agents: list[AgentConfig]
    tasks: list[Task] = Field(default_factory=list)
    tasks_file: str | None = None
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    repeats: int = 3                            # k for pass@k / pass^k
    concurrency: int = 4
    base_dir: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "Suite":
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        data["base_dir"] = path.parent.resolve()
        suite = cls.model_validate(data)
        if suite.tasks_file:
            tpath = (suite.base_dir / suite.tasks_file).resolve()
            raw = yaml.safe_load(tpath.read_text()) or []
            raw = raw.get("tasks", raw) if isinstance(raw, dict) else raw
            suite.tasks = suite.tasks + [Task.model_validate(t) for t in raw]
        if not suite.tasks:
            raise ValueError("suite has no tasks (set `tasks:` or `tasks_file:`)")
        return suite
