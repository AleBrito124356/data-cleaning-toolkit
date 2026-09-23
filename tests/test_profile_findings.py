"""The profiler surfaces the mess the README promises, on the bundled dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cleankit import ProfileReport, profile_dataframe
from cleankit.profile import cluster_values, date_fingerprint, profile_column

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> ProfileReport:
    df = pd.read_csv(
        ROOT / "data" / "messy_sample.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    return profile_dataframe(df)


def _issues(report: ProfileReport, col: str) -> str:
    return "\n".join(report.column(col).issues)


def test_types_and_roles(report):
    got = {c.name: (c.inferred_type, c.semantic) for c in report.columns}
    assert got["Customer ID"] == ("integer", "identifier")
    assert got["Full Name"] == ("text", "person_name")
    assert got["Email Address"] == ("email", None)
    assert got["Phone"] == ("phone", None)
    assert got["Country"] == ("categorical", "country")
    assert got["Signup Date"][0] == "datetime"
    assert got["Is Active"][0] == "boolean"
    assert got["Address"] == ("text", "address")


def test_mixed_date_formats_on_signup_date(report):
    text = _issues(report, "Signup Date")
    assert "Mixed date formats detected: YYYY-MM-DD (10), DD/MM/YYYY (6), Mon DD YYYY (2), DD-MM-YYYY (1)." in text
    assert "Day-first dates present (e.g. 13/02/2023)" in text
    assert "3 ambiguous value(s) such as 05/02/2023" in text


def test_separators_and_ambiguity_on_balance(report):
    text = _issues(report, "Balance (USD)")
    assert "Numbers stored as text with separators in 13 value(s)" in text
    assert "Currency symbols in 1 value(s) (e.g. '$980')" in text
    assert "Mixed decimal marks" in text
    assert "Ambiguous '.' in 1 value(s) ('1.000' -> 1000): read as thousands" in text
    assert report.column("Balance (USD)").stats["min"] == -1200.0


def test_boolean_spellings_on_is_active(report):
    text = _issues(report, "Is Active")
    assert text.startswith("Inconsistent boolean spellings: yes, true, Y, 1, si, TRUE (true) / no, 0 (false).")


def test_panama_variants_on_country(report):
    text = _issues(report, "Country")
    assert "5 spellings of 'Panamá': Panamá (5), Panama (2), panama (1), Rep. de Panamá (1), panamá (1)." in text
    assert "2 spellings of 'Colombia'" in text


def test_repeated_ids_on_customer_id(report):
    text = _issues(report, "Customer ID")
    assert "1002, 1005 with differing rows" in text
    assert "1001 as exact duplicate rows" in text
    assert report.column("Customer ID").details["conflicting_keys"] == ["1002", "1005"]


def test_contact_findings(report):
    assert "Inconsistent casing in 8 name(s)" in _issues(report, "Full Name")
    assert "2 value(s) are whitespace-only" in _issues(report, "Full Name")
    email = _issues(report, "Email Address")
    assert "2 value(s) do not look like valid email addresses" in email
    assert "mistyped provider domain" in email
    assert "12 number(s) have no '+' country code" in _issues(report, "Phone")


def test_clustering_does_not_merge_different_values():
    counts = {"Austria": 3, "Australia": 2, "Gambia": 1, "Zambia": 1, "Category 10": 1, "Category 11": 1}
    clusters = cluster_values(counts)
    assert all(len(c["variants"]) == 1 for c in clusters)
    merged = cluster_values({"Colombia": 3, "Columbia": 1, "COLOMBIA": 1})
    assert len(merged) == 1 and merged[0]["canonical"] == "Colombia"


def test_date_fingerprints():
    assert date_fingerprint("Jan 15 2023") == "Mon DD YYYY"
    assert date_fingerprint("2024/03/15") == "YYYY/MM/DD"
    assert date_fingerprint("15-06-2023") == "DD-MM-YYYY"
    assert date_fingerprint("02/13/2023") == "MM/DD/YYYY"
    assert date_fingerprint("31/02/2023") is None
    assert date_fingerprint("15 de enero de 2023") == "DD Mon YYYY"


def test_phone_column_detected_from_values_without_header_hint():
    s = pd.Series(["+507 6123-4567", "(507) 6222 3333", "6000 1111", "+1 305 555 0199"], name="contacto")
    assert profile_column(s).inferred_type == "phone"


def test_markdown_escapes_pipes():
    rep = profile_dataframe(pd.DataFrame({"a|b": ["x|y", "z"]}))
    row = next(line for line in rep.to_markdown().splitlines() if "a\\|b" in line)
    # Unescaped pipes delimit exactly the 8 declared columns.
    unescaped = row.replace("\\|", "")
    assert unescaped.count("|") == 9
    assert "x\\|y" in row


def test_json_round_trip(report):
    data = json.loads(report.to_json())
    again = ProfileReport.from_dict(data)
    assert again.as_dict() == report.as_dict()
    assert data["columns"][0]["semantic"] == "identifier"


def test_html_is_self_contained_and_dark_mode_aware(report):
    page = report.to_html()
    assert page.startswith("<!doctype html>")
    assert "prefers-color-scheme: dark" in page
    assert 'data-theme="dark"' in page
    assert "http://" not in page and "https://" not in page
    assert page.count("<tr>") == len(report.columns) + 1
    assert "Findings" in page
