"""toolscope: does the model actually find and use your tools?"""

from .agent import AgentContext, AgentError
from .config import AgentConfig, Expectation, ExpectedCall, ServerConfig, Suite, Task
from .metrics import GroupStats, RunScore, aggregate, score
from .runner import RunResult, run_suite, score_messages
from .schema import SCHEMA_VERSION
from .trace import Trace

__all__ = [
    "AgentContext",
    "AgentError",
    "AgentConfig",
    "ExpectedCall",
    "Expectation",
    "ServerConfig",
    "Suite",
    "Task",
    "GroupStats",
    "RunScore",
    "aggregate",
    "score",
    "RunResult",
    "run_suite",
    "score_messages",
    "Trace",
    "SCHEMA_VERSION",
]
__version__ = "0.1.0"
