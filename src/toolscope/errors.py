"""Infrastructure failures: rate limits, timeouts, network and auth errors, dead MCP servers.

A run that hits one of these is attributed to the `infra` stage and excluded from every metric, so a flaky
provider or a crashed server never shows up as the model being worse.
"""

from __future__ import annotations

import asyncio
import random

# class names (anywhere in the MRO) of provider / HTTP client errors worth retrying: LiteLLM, OpenAI,
# Anthropic and httpx all use these spellings
_RETRYABLE_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "Timeout",
    "TimeoutException",
    "ServiceUnavailableError",
    "InternalServerError",
    "BadGatewayError",
    "OverloadedError",
    "TransportError",
    "NetworkError",
}
_AUTH_NAMES = {"AuthenticationError", "PermissionDeniedError"}  # infra, but retrying won't help
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 529}
_AUTH_STATUS = {401, 403}


class InfraError(Exception):
    """An infrastructure failure. Raised by toolscope (dead server, exhausted retries) or by your agent."""


def _status(e: BaseException) -> int | None:
    for obj in (e, getattr(e, "response", None)):
        for attr in ("status_code", "status"):
            v = getattr(obj, attr, None)
            if isinstance(v, int):
                return v
    return None


def _names(e: BaseException) -> set[str]:
    return {c.__name__ for c in type(e).__mro__}


def is_retryable(e: BaseException) -> bool:
    """Transient: rate limit, 5xx, timeout or connection failure."""
    if isinstance(e, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    return bool(_names(e) & _RETRYABLE_NAMES) or _status(e) in _RETRYABLE_STATUS


def is_infra_error(e: BaseException) -> bool:
    """Not the model's or agent's fault: transient errors, auth failures, or an explicit InfraError."""
    if isinstance(e, InfraError) or is_retryable(e):
        return True
    return bool(_names(e) & _AUTH_NAMES) or _status(e) in _AUTH_STATUS


def describe(e: BaseException) -> str:
    msg = str(e)
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def backoff_s(attempt: int, base: float, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter: uniform in [0, min(cap, base * 2**(attempt-1))]."""
    return random.uniform(0, min(cap, base * 2 ** max(attempt - 1, 0))) if base > 0 else 0.0
