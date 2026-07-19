"""Cleaning operations: names, whitespace, coercion, categoricals, missing,
duplicates and outliers."""

import pandas as pd

from cleankit import Cleaner


def test_standardize_column_names():
    df = pd.DataFrame({"Customer ID": [1], "Balance (USD)": [2], "firstName": [3]})
    out = Cleaner(df).standardize_column_names().df
    assert list(out.columns) == ["customer_id", "balance_usd", "first_name"]


def test_standardize_column_names_dedupes_collisions():
    df = pd.DataFrame([[1, 2]], columns=["Name", "name"])
    out = Cleaner(df).standardize_column_names().df
    assert list(out.columns) == ["name", "name_1"]


def test_normalize_whitespace():
    df = pd.DataFrame({"n": ["  Juan   Perez ", "Ana\tLopez", "ok"]})
    c = Cleaner(df).normalize_whitespace()
    assert c.df["n"].tolist() == ["Juan Perez", "Ana Lopez", "ok"]
    assert c.log[-1]["cells_changed"] == 2


def test_coerce_numbers_with_separators():
    df = pd.DataFrame({"amount": ["1,000", "2,500", "3.000,50", "bad"]})
    c = Cleaner(df).coerce_types({"amount": "float"})
    vals = c.df["amount"].tolist()
    assert vals[0] == 1000.0
    assert vals[1] == 2500.0
    assert vals[2] == 3000.50
    assert pd.isna(vals[3])
    assert c.log[-1]["unparsed"] == 1


def test_coerce_integer():
    df = pd.DataFrame({"age": ["34", "29", "1,000"]})
    c = Cleaner(df).coerce_types({"age": "integer"})
    assert c.df["age"].tolist() == [34, 29, 1000]


def test_coerce_boolean():
    df = pd.DataFrame({"active": ["yes", "no", "1", "0", "si", "maybe"]})
    c = Cleaner(df).coerce_types({"active": "boolean"})
    out = c.df["active"].tolist()
    assert out[:5] == [True, False, True, False, True]
    assert pd.isna(out[5])


def test_coerce_datetime_dayfirst_detection():
    df = pd.DataFrame({"d": ["13/02/2023", "05/01/2023", "20/03/2023"]})
    c = Cleaner(df).coerce_types({"d": "datetime"})
    assert c.df["d"].iloc[0] == pd.Timestamp("2023-02-13")
    assert c.df["d"].iloc[2] == pd.Timestamp("2023-03-20")


def test_coerce_datetime_monthfirst_default():
    df = pd.DataFrame({"d": ["01/05/2023", "02/06/2023", "03/07/2023"]})
    c = Cleaner(df).coerce_types({"d": "datetime"})
    # No component exceeds 12, so month-first (US) is assumed.
    assert c.df["d"].iloc[0] == pd.Timestamp("2023-01-05")


def test_standardize_categoricals_fuzzy():
    df = pd.DataFrame(
        {"country": ["Panamá", "panama", "PANAMA", "Colombia", "colombia", "Xyz"]}
    )
    c = Cleaner(df).standardize_categoricals(
        "country", canonical=["Panama", "Colombia"], threshold=0.75
    )
    out = c.df["country"].tolist()
    assert out[:5] == ["Panama", "Panama", "Panama", "Colombia", "Colombia"]
    assert out[5] == "Xyz"  # too far from any canonical value
    assert "Xyz" in c.log[-1]["unmatched"]


def test_handle_missing_fill_median():
    df = pd.DataFrame({"v": ["1", "2", None, "4"]})
    c = Cleaner(df).handle_missing(strategy="fill", columns=["v"], value="median")
    assert c.df["v"].tolist() == ["1", "2", 2.0, "4"]


def test_handle_missing_drop_rows():
    df = pd.DataFrame({"a": ["1", None, "3"], "b": ["x", "y", None]})
    c = Cleaner(df).handle_missing(strategy="drop_rows", columns=["a", "b"])
    assert len(c.df) == 1
    assert c.df["a"].tolist() == ["1"]


def test_deduplicate_exact():
    df = pd.DataFrame({"id": [1, 1, 2, 3, 3], "v": ["a", "a", "b", "c", "c"]})
    c = Cleaner(df).deduplicate()
    assert len(c.df) == 3
    assert c.log[-1]["duplicates"] == 2


def test_deduplicate_on_subset():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "different", "b"]})
    c = Cleaner(df).deduplicate(subset=["id"], keep="first")
    assert c.df["id"].tolist() == [1, 2]
    assert c.df["v"].tolist() == ["a", "b"]


def test_deduplicate_fuzzy():
    df = pd.DataFrame(
        {
            "name": ["Jonathan Smith", "Jonathon Smith", "Maria Gomez"],
            "city": ["NYC", "NYC", "PTY"],
        }
    )
    c = Cleaner(df).deduplicate(fuzzy=True, fuzzy_keys=["name"], threshold=0.85)
    names = c.df["name"].tolist()
    assert "Maria Gomez" in names
    assert len(names) == 2  # the two Jonathans collapse to one


def test_deduplicate_flag_does_not_drop():
    df = pd.DataFrame({"id": [1, 1, 2]})
    c = Cleaner(df).deduplicate(subset=["id"], flag=True)
    assert len(c.df) == 3
    assert c.df["is_duplicate"].tolist() == [False, True, False]


def test_handle_outliers_cap():
    df = pd.DataFrame({"v": ["10", "11", "12", "13", "1000"]})
    c = Cleaner(df).handle_outliers(["v"], method="iqr", action="cap")
    assert c.df["v"].iloc[-1] == 16.0  # capped to the IQR upper fence
    assert c.df["v"].iloc[0] == 10.0


def test_handle_outliers_flag():
    df = pd.DataFrame({"v": ["10", "11", "12", "13", "1000"]})
    c = Cleaner(df).handle_outliers(["v"], method="iqr", action="flag")
    assert c.df["v_is_outlier"].tolist() == [False, False, False, False, True]


def test_audit_log_records_every_step():
    df = pd.DataFrame({"Name ": ["  a ", "b"], "n": ["1", "2"]})
    c = (
        Cleaner(df)
        .standardize_column_names()
        .normalize_whitespace()
        .coerce_types({"n": "integer"})
    )
    ops = [e["op"] for e in c.log]
    assert ops == ["standardize_column_names", "normalize_whitespace", "coerce_types"]
    assert all("message" in e for e in c.log)
