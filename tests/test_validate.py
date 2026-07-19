"""Validation rules: not_null, unique, range, regex, allowed, dtype and
cross-field checks."""

import pandas as pd

from cleankit import validate
from cleankit.validate import ValidationReport


def _report(df, rules) -> ValidationReport:
    return validate(df, rules)


def test_not_null():
    df = pd.DataFrame({"x": ["1", None, "3"]})
    rep = _report(df, {"columns": {"x": {"not_null": True}}})
    assert not rep.ok
    assert rep.failures_by_rule() == {"x.not_null": 1}
    assert rep.failures[0].row == 1


def test_unique():
    df = pd.DataFrame({"id": ["a", "b", "b", "c"]})
    rep = _report(df, {"columns": {"id": {"unique": True}}})
    # Both rows holding "b" are reported.
    assert rep.failures_by_rule() == {"id.unique": 2}


def test_range():
    df = pd.DataFrame({"age": ["25", "200", "-3", "x"]})
    rep = _report(df, {"columns": {"age": {"range": {"min": 0, "max": 120}}}})
    # 200 too high, -3 too low, "x" not numeric.
    assert rep.failures_by_rule()["age.range"] == 3


def test_regex():
    df = pd.DataFrame({"email": ["a@b.com", "bad-email", "c@d.org"]})
    rules = {"columns": {"email": {"regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"}}}
    rep = _report(df, rules)
    assert rep.failures_by_rule() == {"email.regex": 1}
    assert rep.failures[0].value == "bad-email"


def test_allowed():
    df = pd.DataFrame({"status": ["active", "weird", "active", "gone"]})
    rep = _report(df, {"columns": {"status": {"allowed": ["active", "inactive"]}}})
    assert rep.failures_by_rule() == {"status.allowed": 2}


def test_dtype():
    df = pd.DataFrame({"n": ["1", "2.5", "abc"]})
    rep = _report(df, {"columns": {"n": {"dtype": "integer"}}})
    # "2.5" is not integer, "abc" is not numeric.
    assert rep.failures_by_rule()["n.dtype"] == 2


def test_missing_column_reported():
    df = pd.DataFrame({"a": ["1"]})
    rep = _report(df, {"columns": {"ghost": {"not_null": True}}})
    assert not rep.ok
    assert rep.failures[0].rule == "ghost.exists"


def test_cross_field_check_fails_and_passes():
    df = pd.DataFrame(
        {
            "start": ["2023-01-01", "2023-05-01", "2023-01-01"],
            "end": ["2023-02-01", "2023-04-01", "2023-03-01"],
        }
    )
    rules = {
        "checks": [
            {"name": "start_before_end", "left": "start", "op": "<=", "right": "end"}
        ]
    }
    rep = _report(df, rules)
    # Only the middle row violates start <= end.
    assert rep.failures_by_rule() == {"start_before_end": 1}
    assert rep.failures[0].row == 1


def test_cross_field_against_literal():
    df = pd.DataFrame({"qty": [5, 0, 3]})
    rules = {"checks": [{"name": "positive_qty", "left": "qty", "op": ">", "right": 0}]}
    rep = _report(df, rules)
    assert rep.failures_by_rule() == {"positive_qty": 1}


def test_clean_data_passes():
    df = pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "email": ["a@b.com", "c@d.com", "e@f.com"],
            "age": ["20", "30", "40"],
        }
    )
    rules = {
        "columns": {
            "id": {"not_null": True, "unique": True},
            "email": {"regex": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
            "age": {"range": {"min": 0, "max": 120}},
        }
    }
    rep = _report(df, rules)
    assert rep.ok
    assert rep.n_failures == 0


def test_report_markdown_and_dict():
    df = pd.DataFrame({"x": [None]})
    rep = _report(df, {"columns": {"x": {"not_null": True}}})
    md = rep.to_markdown()
    assert "Validation report" in md and "FAIL" in md
    d = rep.as_dict()
    assert d["ok"] is False and d["n_failures"] == 1
