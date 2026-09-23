"""Shared ISO-safe date parsing."""

import datetime as dt

import pandas as pd

from cleankit.dates import detect_dayfirst, looks_like_date_text, parse_date, parse_dates


def test_mixed_column_parses_each_value_in_its_own_format():
    values = ["2023-01-05", "13/02/2023", "05/02/2023", "Jan 15 2023", "15-06-2023", "2024/03/15"]
    out = parse_dates(pd.Series(values))
    assert [d.strftime("%Y-%m-%d") for d in out] == [
        "2023-01-05", "2023-02-13", "2023-02-05", "2023-01-15", "2023-06-15", "2024-03-15",
    ]


def test_month_first_is_kept_without_evidence():
    assert detect_dayfirst(["01/05/2023", "02/06/2023"]) is False
    assert parse_dates(pd.Series(["01/05/2023"])).iloc[0] == pd.Timestamp("2023-01-05")
    assert detect_dayfirst(["02/13/2023", "13/02/2023"]) is True  # tie goes day-first


def test_timezones_datetimes_and_garbage():
    out = parse_dates(pd.Series(["2023-01-05T10:00:00Z", "2023-01-05T12:00:00+02:00", None, "soon"]))
    assert out.iloc[0] == out.iloc[1] == pd.Timestamp("2023-01-05 10:00")
    assert out.iloc[2:].isna().all()
    objs = parse_dates(pd.Series([dt.date(2023, 1, 5), pd.Timestamp("2023-02-01")], dtype=object))
    assert objs.tolist() == [pd.Timestamp("2023-01-05"), pd.Timestamp("2023-02-01")]


def test_date_text_detection_and_scalar_parse():
    assert looks_like_date_text("January 5, 2023")
    assert not looks_like_date_text("20230105")
    assert not looks_like_date_text("Calle 50")
    assert parse_date("2023-12-31") == pd.Timestamp("2023-12-31")
    assert pd.isna(parse_date("not a date"))
