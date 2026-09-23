"""Regression tests for data-corrupting bugs found in the 0.1.0 audit.

Each test pins down one reproduced failure (wrong ISO date comparisons,
blank names passing not_null, Int64 crashes, 1.000 read as 1.0, phones
always "valid", mangled addresses, silently accepted pipeline typos).
"""

from pathlib import Path

import pandas as pd
import pytest

from cleankit import Cleaner, load_pipeline, parse_dates, parse_number, run_pipeline, validate
from cleankit.clean import _is_missing, parse_number_column, snake_case_columns
from cleankit.contacts import clean_address, to_e164

ROOT = Path(__file__).resolve().parents[1]


def _cross(start, end, op="<="):
    df = pd.DataFrame({"start": start, "end": end})
    return validate(df, {"checks": [{"name": "c", "left": "start", "op": op, "right": "end"}]})


# -- validator: ISO dates were parsed day-first ----------------------------------


def test_iso_cross_field_no_false_positive():
    rep = _cross(["2023-01-05", "2023-03-10"], ["2023-02-01", "2023-04-02"])
    assert rep.ok, [f.message for f in rep.failures]


def test_iso_cross_field_no_false_negative():
    # 2 May is after 10 April: must fail.
    rep = _cross(["2023-05-02", "2023-01-01"], ["2023-04-10", "2023-01-03"])
    assert rep.failures_by_rule() == {"c": 1}
    assert rep.failures[0].row == 0
    assert "2023-05-02 vs 2023-04-10" in rep.failures[0].message


def test_cross_field_mixed_formats_use_dayfirst_only_where_needed():
    rep = _cross(["13/02/2023", "2023-01-05"], ["2023-02-14", "2023-01-04"])
    assert rep.failures_by_rule() == {"c": 1}
    assert rep.failures[0].row == 1


def test_cross_field_compares_numeric_text_as_numbers():
    df = pd.DataFrame({"lo": ["9", "10"], "hi": ["10", "9"]})
    rep = validate(df, {"checks": [{"name": "c", "left": "lo", "op": "<", "right": "hi"}]})
    assert [f.row for f in rep.failures] == [1]


def test_cross_field_date_literal():
    df = pd.DataFrame({"d": ["2023-05-02", "2022-12-31"]})
    rep = validate(df, {"checks": [{"name": "c", "left": "d", "op": ">=", "right": "2023-01-01"}]})
    assert [f.row for f in rep.failures] == [1]


def test_year_first_slash_dates_are_never_read_day_first():
    # pandas' dayfirst=True reads 2024/03/05 as 3 May.
    out = parse_dates(pd.Series(["2024/03/05", "13/02/2023"]))
    assert out.tolist() == [pd.Timestamp("2024-03-05"), pd.Timestamp("2023-02-13")]


def test_numbers_are_not_dates():
    assert parse_dates(pd.Series(["5", "1000", "abc"])).isna().all()


# -- whitespace-only cells defeated not_null -------------------------------------


def test_whitespace_only_cells_become_null_and_fail_not_null():
    c = Cleaner(pd.DataFrame({"full_name": ["Ana", "   ", "\xa0"]})).normalize_whitespace()
    assert c.df["full_name"].iloc[0] == "Ana"
    assert c.df["full_name"].iloc[1:].isna().all()
    assert c.log[-1]["blank_to_null"] == 2
    rep = validate(c.df, {"columns": {"full_name": {"not_null": True}}})
    assert rep.failures_by_rule() == {"full_name.not_null": 2}


def test_validator_treats_blank_strings_as_null():
    df = pd.DataFrame({"x": ["a", "   ", ""], "u": ["k", " ", "  "]})
    rep = validate(df, {"columns": {"x": {"not_null": True}, "u": {"unique": True}}})
    assert rep.failures_by_rule() == {"x.not_null": 2}
    assert rep.failures[0].message == "value is blank"


# -- integers: fractional fills crashed, caps turned Int64 into floats -----------


def test_fill_median_on_integer_column_rounds_instead_of_crashing():
    df = pd.DataFrame({"age": ["30", "35", None, "40", "41"]})  # median 37.5
    c = Cleaner(df).coerce_types({"age": "integer"}).handle_missing("fill", ["age"], "median")
    assert str(c.df["age"].dtype) == "Int64"
    assert c.df["age"].tolist() == [30, 35, 38, 40, 41]
    assert "rounded to 38" in c.log[-1]["message"]


def test_bundled_steps_survive_fractional_median(tmp_path):
    raw = pd.read_csv(ROOT / "data" / "messy_sample.csv", dtype=str, keep_default_na=False, na_values=[""])
    raw.loc[raw["Customer ID"] == "1011", "Age"] = "35"  # median becomes fractional
    result = run_pipeline(raw, load_pipeline(ROOT / "steps.yaml"))
    assert str(result.df["age"].dtype) == "Int64"
    assert result.df["age"].notna().all()


def test_cap_keeps_integer_dtype_and_whole_numbers():
    df = pd.DataFrame({"age": ["30", "31", "29", "35", "33", "200"]})
    c = Cleaner(df).coerce_types({"age": "integer"}).handle_outliers(["age"], action="cap")
    assert str(c.df["age"].dtype) == "Int64"
    assert c.df["age"].tolist() == [30, 31, 29, 35, 33, 40]
    assert c.log[-1]["fences"] == {"age": [24.0, 40.0]}


def test_cap_keeps_float_dtype():
    df = pd.DataFrame({"v": ["1.5", "2.5", "2.0", "3.0", "100.0"]})
    c = Cleaner(df).coerce_types({"v": "float"}).handle_outliers(["v"], action="cap")
    assert str(c.df["v"].dtype) == "Float64"


def test_integer_coercion_audits_rounding():
    c = Cleaner(pd.DataFrame({"n": ["2.5", "3.7", "10"]})).coerce_types({"n": "integer"})
    assert c.df["n"].tolist() == [3, 4, 10]  # half away from zero
    entry = c.log[-1]
    assert entry["rounded"] == 2
    assert entry["fraction_examples"][0] == {"row": 0, "value": "2.5", "to": 3}
    assert "2 fractional value(s) were rounded" in entry["message"]


def test_integer_coercion_on_fraction_nullify_and_error():
    df = pd.DataFrame({"n": ["2.5", "3", "x"]})
    c = Cleaner(df).coerce_types({"n": "integer"}, on_fraction="nullify")
    assert c.df["n"].isna().tolist() == [True, False, True]
    assert c.log[-1]["nullified"] == 1
    assert c.log[-1]["unparsed"] == 1  # 'x' is counted separately
    with pytest.raises(ValueError, match="non-integer"):
        Cleaner(df).coerce_types({"n": "integer"}, on_fraction="error")
    with pytest.raises(ValueError, match="YAML"):
        Cleaner(df).coerce_types({"n": "integer"}, on_fraction=None)


# -- column names ----------------------------------------------------------------


def test_snake_case_columns_never_collide():
    assert snake_case_columns(["Name", "name", "name_1"]) == ["name", "name_2", "name_1"]
    cols = snake_case_columns(["a", "a_1", "A", "a 1", "A-1"])
    assert len(set(cols)) == len(cols)
    out = Cleaner(pd.DataFrame([[1, 2, 3]], columns=["Name", "name", "name_1"])).standardize_column_names()
    assert list(out.df.columns) == ["name", "name_2", "name_1"]


# -- numbers: '1.000' next to '2.500,50' became 1.0 ------------------------------


def test_eu_thousands_dot_resolved_from_column_context():
    nums, fmt, ambiguous = parse_number_column(["1,250.00", "2.500,50", "1.000", "3,000"])
    assert nums == [1250.0, 2500.5, 1000.0, 3000.0]
    assert {a["value"]: a["read_as"] for a in ambiguous} == {"1.000": 1000.0, "3,000": 3000.0}


def test_three_decimal_column_keeps_decimal_reading():
    nums, _, ambiguous = parse_number_column(["0.125", "1.500", "2.375"])
    assert nums == [0.125, 1.5, 2.375]
    assert ambiguous[0]["reason"].startswith("'.' marks 3-digit decimals")


def test_comma_decimal_column_reads_dot_as_thousands():
    nums, _, _ = parse_number_column(["1,5", "2,75", "1.000"])
    assert nums == [1.5, 2.75, 1000.0]


def test_parse_number_decimal_hint_and_strictness():
    assert parse_number("1.000") == 1.0
    assert parse_number("1.000", decimal=",") == 1000.0
    assert parse_number("2,5", decimal=",") == 2.5
    assert parse_number("1.234.567") == 1234567.0
    assert parse_number("USD 1,200") == 1200.0
    assert parse_number("1e3") == 1000.0
    assert parse_number("12 years") is None
    assert parse_number(pd.NA) is None


def test_coerce_float_logs_ambiguous_values():
    c = Cleaner(pd.DataFrame({"b": ["2.500,50", "1.000"]})).coerce_types({"b": "float"})
    assert c.df["b"].tolist() == [2500.5, 1000.0]
    assert c.log[-1]["ambiguous"][0]["value"] == "1.000"


# -- phones were valid for any length, addresses were mangled --------------------


def test_phone_validity_uses_the_numbering_plan():
    r = to_e164("3105551234", "PA")
    assert r.valid is False and r.value == "+5073105551234"
    assert to_e164("3105551234", "CO").value == "+573105551234"
    assert to_e164("+507 6123 456").valid is False  # 7 digits starting with 6
    assert to_e164("+507 200-1234").valid is True  # 7-digit Panama landline
    with pytest.raises(ValueError, match="unknown phone region"):
        to_e164("6123-4567", "ZZ")


def test_clean_address_keeps_no_and_ph_comma_and_unit_codes():
    assert clean_address("Calle 5 No. 12, PH, Torre") == "Calle 5 No. 12, PH, Torre"
    assert clean_address("PH Ocean, Apto 12B") == "PH Ocean, Apartamento 12B"
    assert clean_address("Reforma 100, CDMX") == "Reforma 100, CDMX"
    assert clean_address("CALLE 50, CIUDAD DE PANAMA") == "Calle 50, Ciudad de Panama"


# -- pipeline validation was shallow ---------------------------------------------


@pytest.mark.parametrize(
    "body, message",
    [
        ("  - op: handle_missing\n    colums: [age]\n", "Did you mean 'columns'"),
        ("  - op: handle_missing\n    strategy: fil\n", "Did you mean 'fill'"),
        ("  - op: handle_outliers\n    columns: [a]\n    action: capp\n", "Did you mean 'cap'"),
        ("  - op: handle_outliers\n    columns: [a]\n    method: iqrr\n", "Did you mean 'iqr'"),
        ("  - op: deduplicate\n    keep: most\n", "choose one of"),
        ("  - op: coerce_types\n    mapping: {a: intger}\n", "Did you mean 'integer'"),
        ("  - op: standardize_email\n", "Did you mean 'standardize_emails'"),
        ("  - op: standardize_categoricals\n    column: c\n", "missing required parameter 'canonical'"),
        ("  - op: handle_outliers\n    columns: [1, 2]\n", "must be a column name"),
    ],
)
def test_load_pipeline_rejects_bad_steps(tmp_path, body, message):
    path = tmp_path / "steps.yaml"
    path.write_text("name: x\nsteps:\n" + body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_pipeline(str(path))


def test_cleaner_methods_reject_bad_enums_directly():
    c = Cleaner(pd.DataFrame({"a": ["1"]}))
    with pytest.raises(ValueError, match="strategy"):
        c.handle_missing(strategy="fil")
    with pytest.raises(ValueError, match="action"):
        c.handle_outliers(["a"], action="capp")


def test_single_column_name_is_accepted_where_a_list_is_expected():
    df = pd.DataFrame({"age": ["10", "11", "12", "13", "1000"]})
    c = Cleaner(df).handle_outliers("age", action="flag")
    assert c.df["age_is_outlier"].tolist() == [False, False, False, False, True]


def test_is_missing_understands_pandas_na():
    assert _is_missing(pd.NA) and _is_missing(pd.NaT) and _is_missing("  n/a ")
    assert not _is_missing("0")
