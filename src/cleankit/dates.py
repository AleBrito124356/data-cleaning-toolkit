"""ISO-safe date parsing shared by the cleaner, the validator and the profiler.

Messy exports mix ``2023-01-05``, ``13/02/2023``, ``Jan 15 2023`` and
``2024/03/15`` in one column. Handing such a column to ``pd.to_datetime`` in one
call either locks every row onto the first row's format or, with
``dayfirst=True``, silently reads year-first values as year-*day*-month
(``2024/03/05`` becomes 5 March... or 3 May, depending on the path taken).

The rules used here are deliberately simple and predictable:

* year-first values (``YYYY-MM-DD``, ``YYYY/MM/DD``, ``YYYY.MM.DD``, with an
  optional time part) are always parsed year-month-day, never day-first;
* the remaining numeric dates (``13/02/2023``) are parsed day-first or
  month-first according to :func:`detect_dayfirst`, which votes on the values
  that are unambiguous (a first component above 12 means day-first);
* only values that *look* like dates are parsed at all, so ``"5"`` or
  ``"1000"`` never turn into "the 5th of this month".
"""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

__all__ = [
    "detect_dayfirst",
    "looks_like_date_text",
    "parse_dates",
    "parse_date",
    "dayfirst_evidence",
]

# YYYY-MM-DD (optionally followed by a time), strict enough for format="ISO8601".
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
# Any year-first numeric date: 2024/3/5, 2024.03.05, 2024-3-5 10:00.
_YEAR_FIRST_RE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[T ].*)?$")
# Day-or-month-first numeric date: 13/02/2023, 5-6-23, 01.02.2023 (+ time).
_DM_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})(?:[T ].*)?$")
_MONTH_WORDS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|"
    "november|december"
)
_MONTH_NAME_RE = re.compile(rf"\b({_MONTH_WORDS})\b\.?", re.IGNORECASE)


def looks_like_date_text(value: Any) -> bool:
    """True when ``value`` is a string shaped like a calendar date.

    Numbers such as ``"5"`` or ``"20230105"`` are rejected on purpose: they
    are far more often quantities or identifiers than dates.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < 6 or not any(ch.isdigit() for ch in text):
        return False
    if _YEAR_FIRST_RE.match(text) or _DM_RE.match(text):
        return True
    # "Jan 15 2023", "15 Jan 2023", "January 5, 2023": a month word plus a year.
    return bool(_MONTH_NAME_RE.search(text)) and bool(re.search(r"\d{4}", text))


def dayfirst_evidence(values: Iterable[Any]) -> tuple[int, int, int]:
    """Count (day_first, month_first, ambiguous) among ``dd/mm/yyyy``-style values."""
    day_first = month_first = ambiguous = 0
    for v in values:
        if not isinstance(v, str):
            continue
        m = _DM_RE.match(v.strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 >= b:
            day_first += 1
        elif b > 12 >= a:
            month_first += 1
        elif a <= 12 and b <= 12:
            ambiguous += 1
    return day_first, month_first, ambiguous


def detect_dayfirst(values: Iterable[Any], sample: int = 5000) -> bool:
    """Vote on day-first vs month-first using only the unambiguous values.

    Returns ``True`` when at least one value can only be day-first and
    day-first votes are not outnumbered. With no evidence either way the
    month-first (US) reading is kept, matching pandas' default.
    """
    head = []
    for i, v in enumerate(values):
        if i >= sample:
            break
        head.append(v)
    day_first, month_first, _ = dayfirst_evidence(head)
    return day_first > 0 and day_first >= month_first


def _is_datetime_scalar(v: Any) -> bool:
    return isinstance(v, (pd.Timestamp, datetime, date, np.datetime64))


def _to_naive(parsed: pd.Series) -> pd.Series:
    """Drop timezone info (converting to UTC first) so columns stay comparable."""
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def parse_dates(values: Any, dayfirst: bool | None = None) -> pd.Series:
    """Parse a column of mixed-format dates element by element.

    ``values`` may be a Series or any iterable. The result is a
    ``datetime64[ns]`` Series aligned with the input index; values that are
    missing, not date-shaped, or not parseable become ``NaT``.
    Timezone-aware values are converted to UTC and made naive.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        out = _to_naive(series)
        return out.astype("datetime64[ns]")

    obj = series.astype("object")
    if dayfirst is None:
        dayfirst = detect_dayfirst(v.strip() for v in obj.tolist() if isinstance(v, str))
    present = obj[obj.notna()]
    try:
        uniques = pd.Series(present.unique(), dtype="object")
    except TypeError:  # unhashable values: parse element by element
        return _parse_distinct(obj, dayfirst)
    if len(uniques) == len(obj):
        return _parse_distinct(obj, dayfirst)
    parsed = _parse_distinct(uniques, dayfirst)
    lookup = dict(zip(uniques.tolist(), parsed.tolist()))
    out = obj.map(lambda v: lookup.get(v, pd.NaT) if v is not None else pd.NaT)
    return pd.to_datetime(out, errors="coerce").astype("datetime64[ns]")


def _parse_distinct(obj: pd.Series, dayfirst: bool) -> pd.Series:
    stripped = obj.map(lambda v: v.strip() if isinstance(v, str) else v)
    result = pd.Series(pd.NaT, index=obj.index, dtype="datetime64[ns]")

    is_ts = stripped.map(_is_datetime_scalar)
    iso = stripped.map(lambda v: isinstance(v, str) and bool(_ISO_RE.match(v)))
    year_first = stripped.map(
        lambda v: isinstance(v, str) and bool(_YEAR_FIRST_RE.match(v))
    ) & ~iso
    other = stripped.map(looks_like_date_text) & ~iso & ~year_first

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_ts.any():
            result.loc[is_ts] = _to_naive(
                pd.to_datetime(stripped[is_ts], errors="coerce", utc=True)
            )
        if iso.any():
            result.loc[iso] = _to_naive(
                pd.to_datetime(
                    stripped[iso], errors="coerce", format="ISO8601", utc=True
                )
            )
        if year_first.any():
            result.loc[year_first] = pd.to_datetime(
                stripped[year_first],
                errors="coerce",
                format="mixed",
                dayfirst=False,
                yearfirst=True,
            )
        if other.any():
            result.loc[other] = pd.to_datetime(
                stripped[other], errors="coerce", format="mixed", dayfirst=dayfirst
            )
    return result


def parse_date(value: Any, dayfirst: bool = False) -> pd.Timestamp | Any:
    """Parse one value with the same rules as :func:`parse_dates` (``NaT`` if not a date)."""
    return parse_dates(pd.Series([value], dtype="object"), dayfirst=dayfirst).iloc[0]
