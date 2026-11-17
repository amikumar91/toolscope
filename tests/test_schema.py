import pytest

from toolscope.schema import SCHEMA_VERSION, UnsupportedSchemaVersion, check_schema_version


def test_missing_version_is_legacy_1():
    assert check_schema_version({}) == "1" and SCHEMA_VERSION.startswith("1.")


@pytest.mark.parametrize("v", ["1", "1.0", "1.12", 1])
def test_same_major_is_accepted(v):
    assert check_schema_version({"schema_version": v}) == str(v)


@pytest.mark.parametrize("v", ["2", "0", "3.1"])
def test_other_major_is_rejected(v):
    with pytest.raises(UnsupportedSchemaVersion, match="x.json has schema_version"):
        check_schema_version({"schema_version": v}, "x.json")


def test_garbage_version_is_rejected():
    with pytest.raises(UnsupportedSchemaVersion, match="invalid schema_version 'abc'"):
        check_schema_version({"schema_version": "abc"})
