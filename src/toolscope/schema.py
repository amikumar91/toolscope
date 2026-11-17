"""Versioning of toolscope's output files (summary.json, scores.jsonl, traces.jsonl, tool_stats.json).

Every output carries `schema_version` ("<major>" or "<major>.<minor>"). Adding optional fields is
backward compatible and keeps the version; removing, renaming or changing the meaning of a field bumps
the major version. Readers treat files without the field as version "1" and refuse unknown majors.
"""

from __future__ import annotations

from typing import Any

# 1.1: `infra` failure stage + Trace.infra_error/retries, RunScore.retries, GroupStats.infra_errors/retries
SCHEMA_VERSION = "1.1"
SCHEMA_KEY = "schema_version"
LEGACY_VERSION = "1"  # files written before versioning


class UnsupportedSchemaVersion(ValueError):
    """An output file was written by a newer toolscope with an incompatible schema."""


def _major(version: Any) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except ValueError:
        raise UnsupportedSchemaVersion(f"invalid {SCHEMA_KEY} {version!r}") from None


def check_schema_version(record: dict[str, Any], source: str = "output") -> str:
    """Return the record's schema version (default "1") or raise if its major version is unknown."""
    version = str(record.get(SCHEMA_KEY, LEGACY_VERSION))
    if _major(version) != _major(SCHEMA_VERSION):
        raise UnsupportedSchemaVersion(
            f"{source} has {SCHEMA_KEY} {version!r}, but this toolscope reads major version "
            f"{_major(SCHEMA_VERSION)}; upgrade toolscope to read it"
        )
    return version
