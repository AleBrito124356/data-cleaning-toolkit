"""Type inference and profiling."""

import pandas as pd

from cleankit import profile_dataframe
from cleankit.clean import parse_boolean, parse_number, to_snake_case
from cleankit.profile import profile_column


def _types(df):
    report = profile_dataframe(df)
    return {c.name: c.inferred_type for c in report.columns}


def test_infer_basic_types():
    df = pd.DataFrame(
        {
            "count": ["1", "2", "3", "1000"],
            "price": ["1.5", "2.25", "3,000.50", "9.99"],
            "active": ["yes", "no", "yes", "no"],
            "signup": ["2023-01-05", "2023-02-06", "2023-03-07", "2023-04-08"],
            "email": ["a@b.com", "c@d.org", "x@y.net", "z@w.io"],
        }
    )
    types = _types(df)
    assert types["count"] == "integer"
    assert types["price"] == "float"
    assert types["active"] == "boolean"
    assert types["signup"] == "datetime"
    assert types["email"] == "email"


def test_infer_categorical_and_text():
    df = pd.DataFrame(
        {
            "grade": ["A", "B", "A", "C", "B", "A", "C", "B"],
            "note": [f"free text number {i} is unique" for i in range(8)],
        }
    )
    types = _types(df)
    assert types["grade"] == "categorical"
    assert types["note"] == "text"


def test_missingness_and_cardinality():
    df = pd.DataFrame({"x": ["1", "2", None, "N/A", "2", ""]})
    prof = profile_column(df["x"])
    # None, "N/A" and "" are all treated as missing.
    assert prof.n_missing == 3
    assert prof.n_present == 3
    assert prof.n_unique == 2  # "1" and "2"


def test_outliers_and_issues_flagged():
    df = pd.DataFrame({"v": [str(x) for x in [10, 11, 12, 13, 12, 11, 5000]]})
    prof = profile_column(df["v"])
    assert prof.inferred_type == "integer"
    assert prof.n_outliers >= 1
    assert any("outlier" in issue.lower() for issue in prof.issues)


def test_constant_and_whitespace_issue():
    df = pd.DataFrame({"c": ["  same ", "same", "same"]})
    prof = profile_column(df["c"])
    assert any("whitespace" in issue.lower() for issue in prof.issues)


def test_report_renders_markdown_and_html():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["x@y.com", "z@w.com"]})
    report = profile_dataframe(df)
    md = report.to_markdown()
    html = report.to_html()
    assert "# Data profile" in md
    assert "<table" in html and "</html>" in html
    assert "a" in md and "b" in md


def test_parse_number_formats():
    assert parse_number("1,234.56") == 1234.56
    assert parse_number("1.234,56") == 1234.56
    assert parse_number("1 234,50") == 1234.50
    assert parse_number("$2,000") == 2000.0
    assert parse_number("(1234)") == -1234.0
    assert parse_number("50%") == 0.5
    assert parse_number("n/a") is None
    assert parse_number("abc") is None


def test_parse_boolean_spellings():
    for truthy in ["yes", "Y", "true", "1", "si", "Sí", "on"]:
        assert parse_boolean(truthy) is True
    for falsy in ["no", "N", "false", "0", "off"]:
        assert parse_boolean(falsy) is False
    assert parse_boolean("maybe") is None


def test_snake_case():
    assert to_snake_case("Customer ID") == "customer_id"
    assert to_snake_case("Balance (USD)") == "balance_usd"
    assert to_snake_case("firstName") == "first_name"
    assert to_snake_case("HTTPStatus") == "http_status"
    assert to_snake_case("2020_total") == "col_2020_total"
    assert to_snake_case("  país  ") == "pais"
