"""Pipeline execution, reproducibility, and audit trail."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cleankit import load_pipeline, run_pipeline
from cleankit.pipeline import frame_hash


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Customer ID": ["1", "1", "2", "3"],
            "Full Name": ["  Ana ", "  Ana ", "Bob", "Cid"],
            "Country": ["Panamá", "panama", "Colombia", "colombia"],
            "Age": ["30", "30", "200", "40"],
        }
    )


def _config() -> dict:
    return {
        "name": "test",
        "steps": [
            {"op": "standardize_column_names"},
            {"op": "normalize_whitespace"},
            {"op": "coerce_types", "mapping": {"customer_id": "integer", "age": "integer"}},
            {
                "op": "standardize_categoricals",
                "column": "country",
                "canonical": ["Panama", "Colombia"],
                "threshold": 0.75,
            },
            {"op": "handle_outliers", "columns": ["age"], "method": "iqr", "action": "cap"},
            {"op": "deduplicate", "subset": ["customer_id"], "keep": "first"},
        ],
    }


def test_pipeline_runs_and_summarizes():
    result = run_pipeline(_sample(), _config())
    assert result.before["rows"] == 4
    assert result.after["rows"] == 3  # one exact duplicate id removed
    assert list(result.df.columns) == ["customer_id", "full_name", "country", "age"]
    assert set(result.df["country"]) == {"Panama", "Colombia"}


def test_pipeline_is_reproducible():
    cfg = _config()
    r1 = run_pipeline(_sample(), cfg)
    r2 = run_pipeline(_sample(), cfg)
    assert r1.output_hash == r2.output_hash
    assert_frame_equal(r1.df, r2.df)


def test_frame_hash_is_index_independent():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 3]}, index=[10, 20, 30])
    assert frame_hash(a) == frame_hash(b)


def test_frame_hash_changes_with_data():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 4]})
    assert frame_hash(a) != frame_hash(b)


def test_audit_log_covers_all_steps():
    result = run_pipeline(_sample(), _config())
    ops = [e["op"] for e in result.audit_log]
    assert ops == [
        "standardize_column_names",
        "normalize_whitespace",
        "coerce_types",
        "coerce_types",  # one entry per coerced column
        "standardize_categoricals",
        "handle_outliers",
        "deduplicate",
    ]
    md = result.to_markdown()
    assert "Audit log" in md and "Before / after" in md


def test_load_pipeline_rejects_unknown_op(tmp_path):
    bad = tmp_path / "steps.yaml"
    bad.write_text("name: x\nsteps:\n  - op: not_a_real_op\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline(str(bad))


def test_load_pipeline_requires_steps(tmp_path):
    bad = tmp_path / "steps.yaml"
    bad.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline(str(bad))


def test_load_pipeline_roundtrip(tmp_path):
    good = tmp_path / "steps.yaml"
    good.write_text(
        "name: demo\nsteps:\n  - op: standardize_column_names\n", encoding="utf-8"
    )
    cfg = load_pipeline(str(good))
    assert cfg["name"] == "demo"
    result = run_pipeline(_sample(), cfg)
    assert result.name == "demo"
