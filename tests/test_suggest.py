"""`suggest`: profile -> starter steps.yaml + rules.yaml, end to end."""

import io
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cleankit import load_pipeline, load_rules, run_pipeline, suggest, validate

ROOT = Path(__file__).resolve().parents[1]


def _raw() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "messy_sample.csv", dtype=str, keep_default_na=False, na_values=[""]
    )


@pytest.fixture(scope="module")
def sugg():
    return suggest(_raw(), default_region="PA", source="data/messy_sample.csv")


def test_generated_yaml_round_trips(sugg):
    assert yaml.safe_load(sugg.steps_yaml()) == sugg.pipeline
    assert yaml.safe_load(sugg.rules_yaml()) == sugg.rules


def test_steps_cover_the_profile(sugg):
    ops = [s["op"] for s in sugg.pipeline["steps"]]
    assert ops == [
        "standardize_column_names",
        "normalize_whitespace",
        "handle_missing",
        "coerce_types",
        "standardize_categoricals",
        "standardize_names",
        "standardize_emails",
        "standardize_phones",
        "clean_addresses",
        "handle_outliers",
        "deduplicate",
    ]
    by_op = {s["op"]: s for s in sugg.pipeline["steps"]}
    assert by_op["coerce_types"]["mapping"] == {
        "customer_id": "integer",
        "signup_date": "datetime",
        "last_login": "datetime",
        "balance_usd": "float",
        "is_active": "boolean",
        "age": "integer",
    }
    cats = by_op["standardize_categoricals"]
    assert cats["column"] == "country"
    assert cats["canonical"][0] == "Panamá"
    assert cats["extra_aliases"] == {"Rep. de Panamá": "Panamá"}
    assert by_op["standardize_phones"]["region_column"] == "country"
    assert by_op["deduplicate"] == {"op": "deduplicate", "subset": ["customer_id"], "keep": "most_complete"}
    assert by_op["handle_outliers"]["action"] == "flag"


def test_judgement_calls_are_commented_out(sugg):
    text = sugg.steps_yaml()
    assert "  # - op: handle_missing\n  #   strategy: fill\n  #   columns: [age]" in text
    assert all(s["op"] != "handle_missing" or s["strategy"] != "fill" for s in sugg.pipeline["steps"])


def test_rules_target_snake_case_output(sugg):
    cols = sugg.rules["columns"]
    assert cols["customer_id"] == {"not_null": True, "unique": True, "dtype": "integer"}
    assert cols["phone"]["regex"].startswith("^\\+")
    assert cols["is_active"] == {"allowed": ["True", "False"]}
    assert cols["age"]["range"] == {"min": 0, "max": 99}
    assert cols["country"]["allowed"] == sugg.pipeline["steps"][4]["canonical"]
    assert sugg.rules["checks"] == [
        {"name": "signup_date_not_after_last_login", "left": "signup_date", "op": "<=", "right": "last_login"}
    ]


def test_suggested_pipeline_and_rules_end_to_end(sugg, tmp_path):
    steps_path = tmp_path / "steps.yaml"
    rules_path = tmp_path / "rules.yaml"
    steps_path.write_text(sugg.steps_yaml(), encoding="utf-8")
    rules_path.write_text(sugg.rules_yaml(), encoding="utf-8")

    result = run_pipeline(_raw(), load_pipeline(steps_path))
    buf = io.StringIO()
    result.df.to_csv(buf, index=False)
    buf.seek(0)
    cleaned = pd.read_csv(buf, dtype=str, keep_default_na=False, na_values=[""])
    report = validate(cleaned, load_rules(rules_path))

    # Only genuinely malformed values fail: two emails without '@', the two
    # absurd ages and the 9,999,999 balance the profiler flagged.
    assert report.failures_by_rule() == {
        "email_address.regex": 2,
        "age.range": 2,
        "balance_usd.range": 1,
    }
    bad_emails = sorted(f.value for f in report.failures if f.rule == "email_address.regex")
    assert bad_emails == ["ana_lopezexample.com", "valentina.rojasgmail.com"]
    assert sorted(f.value for f in report.failures if f.rule == "age.range") == ["150", "200"]


def test_suggest_rejects_unknown_region():
    with pytest.raises(ValueError, match="unknown phone region"):
        suggest(_raw(), default_region="XX")


def test_suggest_on_a_clean_frame_is_minimal():
    df = pd.DataFrame({"id": ["1", "2", "3", "4"], "score": ["10", "12", "11", "13"]})
    s = suggest(df)
    ops = [st["op"] for st in s.pipeline["steps"]]
    assert "deduplicate" not in ops and "handle_outliers" not in ops
    assert s.rules["columns"]["id"]["unique"] is True


def test_suggest_on_a_header_only_frame_round_trips():
    s = suggest(pd.DataFrame(columns=["A", "B"]))
    assert yaml.safe_load(s.rules_yaml()) == {"columns": {}, "checks": []}
    assert yaml.safe_load(s.steps_yaml()) == s.pipeline
