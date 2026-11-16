"""Token estimates for schema cost. Uses tiktoken (cl100k) when installed, else chars/4."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _enc():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    enc = _enc()
    return len(enc.encode(text)) if enc else max(1, len(text) // 4)
