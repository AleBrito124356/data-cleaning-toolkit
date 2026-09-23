"""Composable, logged cleaning operations.

Every operation on :class:`Cleaner` mutates a working copy of the DataFrame and
appends a structured entry to an audit log describing exactly what changed
(columns touched, rows/cells affected, before/after row counts, examples of
the values it had to interpret, and a short human-readable message). Methods
return ``self`` so they chain.

    result = (
        Cleaner(df)
        .standardize_column_names()
        .normalize_whitespace()
        .coerce_types({"signup_date": "datetime", "balance": "float"})
        .standardize_emails(["email"])
        .deduplicate(subset=["email"], keep="most_complete")
    )
    clean_df = result.df
    for entry in result.log:
        print(entry["message"])

Row numbers in the audit log are the DataFrame's index labels. The Cleaner
never renumbers rows, so for a frame read from a CSV they stay the 0-based
position of the record in the input file (CSV line = row + 2).
"""

from __future__ import annotations

import difflib
import functools
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import contacts as _contacts
from .dates import dayfirst_evidence, detect_dayfirst, parse_dates
from .text import canonical_key as _text_key

# ---------------------------------------------------------------------------
# Option vocabularies (also used by pipeline validation)
# ---------------------------------------------------------------------------

TYPE_TARGETS = {
    "integer": "integer", "int": "integer",
    "float": "float", "number": "float",
    "boolean": "boolean", "bool": "boolean",
    "datetime": "datetime", "date": "datetime",
    "string": "string", "str": "string", "text": "string",
}
MISSING_STRATEGIES = ("flag", "fill", "drop_rows", "drop_columns")
MISSING_HOW = ("any", "all")
OUTLIER_METHODS = ("iqr", "zscore")
OUTLIER_ACTIONS = ("flag", "cap")
KEEP_OPTIONS = ("first", "last", "most_complete")
ON_FRACTION = ("round", "nullify", "error")
INVALID_ACTIONS = ("keep", "null", "flag")


def check_choice(param: str, value: Any, choices: Iterable[str]) -> str:
    """Raise a helpful ``ValueError`` when ``value`` is not one of ``choices``."""
    choices = list(choices)
    if isinstance(value, str) and value in choices:
        return value
    hint = ""
    if isinstance(value, str):
        close = difflib.get_close_matches(value, choices, n=1, cutoff=0.6)
        if close:
            hint = f" Did you mean '{close[0]}'?"
    raise ValueError(
        f"{param}={value!r} is not valid; choose one of: {', '.join(choices)}.{hint}"
    )


def _as_list(value: Any) -> list[str] | None:
    """Accept a single column name where a list is expected (YAML ``columns: age``)."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


# ---------------------------------------------------------------------------
# Scalar parsers (importable and independently testable)
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "t", "yes", "y", "1", "si", "sí", "s", "on", "x", "✓"}
_BOOL_FALSE = {"false", "f", "no", "n", "0", "off", "no.", "n/a"}

_NULL_TOKENS = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "nil",
    "-",
    "--",
    "?",
    "unknown",
    "missing",
    "#n/a",
}

# Whitespace-like code points to fold into a plain space: ASCII control
# whitespace, the non-breaking space, the Unicode space range 2000-200A,
# the zero-width space, narrow/ideographic spaces, and the BOM/ZWNBSP.
_WHITESPACE_CODEPOINTS = frozenset(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B + 1))
    + [0x202F, 0x205F, 0x3000, 0xFEFF]
)
_WS_TRANSLATION = {cp: " " for cp in _WHITESPACE_CODEPOINTS}


def _is_missing(value: Any) -> bool:
    """None/NaN/NA/NaT, blank strings, and textual placeholders such as 'N/A'."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NULL_TOKENS
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def _missing_series(series: pd.Series) -> pd.Series:
    """Vectorised ``_is_missing`` for a whole column."""
    na = series.isna()
    if not _is_text_dtype(series):
        return na
    obj = series.astype(object)
    tokens = {
        u for u in obj[~na].unique() if isinstance(u, str) and u.strip().lower() in _NULL_TOKENS
    }
    return na | obj.isin(tokens) if tokens else na


def _missing_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: _missing_series(df[c]) for c in df.columns}, index=df.index)


def _map_unique(series: pd.Series, func: Callable[[Any], Any]) -> pd.Series:
    """``series.map(func)`` evaluating ``func`` once per distinct value."""
    obj = series.astype(object)
    cache: dict[Any, Any] = {}
    out = []
    for v in obj.tolist():
        try:
            out.append(cache[v])
        except KeyError:
            cache[v] = r = func(v)
            out.append(r)
        except TypeError:  # unhashable
            out.append(func(v))
    return pd.Series(out, index=series.index, dtype=object)


def to_snake_case(name: str) -> str:
    """Convert an arbitrary header to ``snake_case``.

    Handles camelCase, spaces, punctuation, accented characters, and headers
    that begin with a digit.
    """
    text = str(name).strip()
    # Strip accents.
    text = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    # Split camelCase / PascalCase boundaries.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    # Non-alphanumeric to underscore.
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = text.strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        text = "column"
    if text[0].isdigit():
        text = "col_" + text
    return text


def snake_case_columns(columns: Iterable[Any]) -> list[str]:
    """snake_case a list of headers and guarantee the result has no duplicates.

    The first header that produces a name keeps it; later collisions get the
    smallest ``_<n>`` suffix that is not already used *and* is not the natural
    name of another header, so ``["Name", "name", "name_1"]`` becomes
    ``["name", "name_2", "name_1"]``.
    """
    bases = [to_snake_case(c) for c in columns]
    natural = set(bases)
    taken: set[str] = set()
    out: list[str] = []
    for base in bases:
        name = base
        if name in taken:
            k = 1
            while f"{base}_{k}" in taken or f"{base}_{k}" in natural:
                k += 1
            name = f"{base}_{k}"
        taken.add(name)
        out.append(name)
    return out


def normalize_text(value: Any, form: str = "NFKC") -> Any:
    """Unicode-normalise, replace exotic whitespace, trim, and collapse spaces.

    A value made only of whitespace (spaces, tabs, non-breaking spaces...)
    becomes ``None``: it carries no information and must not pass a
    ``not_null`` rule.
    """
    if not isinstance(value, str):
        return value
    s = unicodedata.normalize(form, value).translate(_WS_TRANSLATION)
    s = re.sub(r" {2,}", " ", s).strip()
    return s if s else None


# -- numbers -------------------------------------------------------------------

_CURRENCY_RE = re.compile(
    r"(?:us\$|r\$|usd|eur|mxn|cop|pab|pen|clp|ars|crc|gtq|dop|brl|b/\.|s/\.?|bs\.?|[$€£¥])",
    re.IGNORECASE,
)
_EXP_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$")
_GROUPED_RE = {
    ".": re.compile(r"^\d{1,3}(?:\.\d{3})+$"),
    ",": re.compile(r"^\d{1,3}(?:,\d{3})+$"),
}


_SPACING_RE = re.compile(r"[\s'’_]")
_DIGITS_SEPS_RE = re.compile(r"[0-9.,]+")


def _numeric_core(value: Any) -> tuple[str, bool, bool] | None:
    """Strip currency, sign, percent and spacing: ``'($1,234.5)'`` -> ``('1,234.5', True, False)``."""
    return _numeric_core_str(str(value))


@functools.lru_cache(maxsize=1 << 16)
def _numeric_core_str(value: str) -> tuple[str, bool, bool] | None:
    s = value.strip()
    if s == "" or s.lower() in _NULL_TOKENS:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    percent = s.endswith("%")
    if percent:
        s = s[:-1]
    s = _CURRENCY_RE.sub("", s)
    s = _SPACING_RE.sub("", s)
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("-"):
        negative = not negative if s.count("-") == 1 else negative
        s = s[1:]
    if s in {"", ".", ","} or not _DIGITS_SEPS_RE.fullmatch(s):
        return None
    return s, negative, percent


def _auto_separators(s: str) -> str:
    has_comma, has_dot = "," in s, "." in s
    if has_comma and has_dot:
        # The right-most separator is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            return s.replace(".", "").replace(",", ".")
        return s.replace(",", "")
    if has_comma:
        # Comma is decimal if it appears once with 1-2 trailing digits,
        # otherwise it is a thousands separator.
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            return parts[0] + "." + parts[1]
        return s.replace(",", "")
    if has_dot and s.count(".") > 1 and _GROUPED_RE["."].match(s):
        return s.replace(".", "")  # 1.234.567
    return s


def parse_number(value: Any, decimal: str | None = None) -> float | None:
    """Parse a number that may carry thousands/decimal separators or currency.

    Understands ``1,234.56`` (US), ``1.234,56`` (EU), ``1 234,56``,
    ``$1,234``, ``USD 1,234``, ``(1234)`` for negatives, trailing ``%`` and
    ``1e5``. Text with other letters (``12 years``) is not a number.

    ``decimal`` resolves the genuinely ambiguous cases: with ``decimal=","``
    the value ``1.000`` is one thousand and ``2,5`` is two and a half; with
    ``decimal="."`` ``1.000`` is one. Without a hint a lone comma followed by
    three digits is a thousands separator and a lone dot is a decimal point.
    """
    if value is None or isinstance(value, bool):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return None if np.isnan(f) else f
    return _parse_number_str(str(value), decimal)


@functools.lru_cache(maxsize=1 << 16)
def _parse_number_str(value: str, decimal: str | None) -> float | None:
    raw = value.strip()
    if _EXP_RE.match(raw):
        return float(raw)
    core = _numeric_core_str(raw)
    if core is None:
        return None
    s, negative, percent = core

    has_comma, has_dot = "," in s, "." in s
    if decimal == "," and not (has_comma and has_dot and s.rfind(".") > s.rfind(",")) and s.count(",") <= 1:
        s = s.replace(".", "").replace(",", ".")
    elif decimal == "." and not (has_comma and has_dot and s.rfind(",") > s.rfind(".")) and s.count(".") <= 1:
        s = s.replace(",", "")
    else:
        s = _auto_separators(s)

    try:
        num = float(s)
    except ValueError:
        return None
    if percent:
        num /= 100.0
    return -num if negative else num


def ambiguous_separator(value: Any) -> str | None:
    """Return ``'.'`` or ``','`` if ``value`` is ``1.000``-style ambiguous, else None.

    Ambiguous means: exactly one separator, followed by exactly three digits,
    preceded by one to three digits (not ``0``), so it could be a thousands
    group or a three-decimal fraction.
    """
    if not isinstance(value, str):
        return None
    return _ambiguous_separator_str(value)


@functools.lru_cache(maxsize=1 << 16)
def _ambiguous_separator_str(value: str) -> str | None:
    core = _numeric_core_str(value)
    if core is None:
        return None
    s = core[0]
    for sep, other in ((".", ","), (",", ".")):
        if other in s or s.count(sep) != 1:
            continue
        int_part, frac = s.split(sep)
        if len(frac) == 3 and int_part.isdigit() and 1 <= len(int_part) <= 3 and int_part != "0":
            return sep
    return None


@dataclass
class NumberFormat:
    """Evidence about how a column uses ``.`` and ``,``.

    ``thousands[sep]`` counts values where ``sep`` can only be a grouping
    separator, ``decimal[sep]`` values where it can only be a decimal mark,
    and ``decimal3[sep]`` values where it is a decimal mark followed by
    exactly three digits (``0.125``).
    """

    thousands: dict[str, int] = field(default_factory=lambda: {".": 0, ",": 0})
    decimal: dict[str, int] = field(default_factory=lambda: {".": 0, ",": 0})
    decimal3: dict[str, int] = field(default_factory=lambda: {".": 0, ",": 0})
    examples: dict[str, str] = field(default_factory=dict)
    ambiguous: list[str] = field(default_factory=list)

    def _note(self, key: str, value: str) -> None:
        self.examples.setdefault(key, value)

    def resolve(self, sep: str) -> tuple[str, str]:
        """How to read an ambiguous value using ``sep``: ('thousands'|'decimal', reason)."""
        other = "," if sep == "." else "."
        if self.decimal3[sep] and not self.thousands[sep]:
            ex = self.examples.get(f"decimal3{sep}", "")
            return "decimal", f"'{sep}' marks 3-digit decimals elsewhere in this column (e.g. {ex})"
        if self.thousands[sep] and not self.decimal3[sep]:
            ex = self.examples.get(f"thousands{sep}", "")
            return "thousands", f"'{sep}' groups thousands elsewhere in this column (e.g. {ex})"
        if self.decimal[other] and not self.decimal[sep] and not self.decimal3[sep]:
            ex = self.examples.get(f"decimal{other}", "")
            return "thousands", f"'{other}' is the decimal mark elsewhere in this column (e.g. {ex})"
        if sep == ",":
            return "thousands", "no evidence in the column; default reading"
        return "decimal", "no evidence in the column; default reading"

    @property
    def mixed_decimal_marks(self) -> bool:
        return bool(self.decimal["."] and self.decimal[","])


def infer_number_format(values: Iterable[Any]) -> NumberFormat:
    """Collect separator evidence from the unambiguous values of a column."""
    fmt = NumberFormat()
    counts = Counter(v for v in values if isinstance(v, str))
    for v, n in counts.items():
        core = _numeric_core(v)
        if core is None:
            continue
        s = core[0]
        v = v.strip()
        has_comma, has_dot = "," in s, "." in s
        if has_comma and has_dot:
            left, right = (".", ",") if s.rfind(",") > s.rfind(".") else (",", ".")
            fmt.thousands[left] += n
            fmt._note(f"thousands{left}", v)
            frac = s.rsplit(right, 1)[1]
            key = "decimal3" if len(frac) == 3 else "decimal"
            getattr(fmt, key)[right] += n
            fmt._note(f"{key}{right}", v)
            continue
        sep = "," if has_comma else "." if has_dot else None
        if sep is None:
            continue
        if s.count(sep) > 1:
            if _GROUPED_RE[sep].match(s):
                fmt.thousands[sep] += n
                fmt._note(f"thousands{sep}", v)
            continue
        if ambiguous_separator(v):
            fmt.ambiguous.extend([v] * n)
            continue
        int_part, frac = s.split(sep)
        key = "decimal3" if len(frac) == 3 else "decimal"
        getattr(fmt, key)[sep] += n
        fmt._note(f"{key}{sep}", v)
    return fmt


def parse_number_column(
    values: Iterable[Any],
) -> tuple[list[float | None], NumberFormat, list[dict[str, Any]]]:
    """Parse a whole column, resolving ``1.000``-style values from column context.

    Returns the parsed numbers, the separator evidence, and one record per
    ambiguous value: ``{"value", "read_as", "reason"}``.
    """
    values = list(values)
    fmt = infer_number_format(values)
    decisions = {sep: fmt.resolve(sep) for sep in (".", ",")}
    out: list[float | None] = []
    ambiguous: list[dict[str, Any]] = []
    cache: dict[str, tuple[float | None, str | None]] = {}
    for v in values:
        if isinstance(v, str):
            hit = cache.get(v)
            if hit is None:
                sep = ambiguous_separator(v)
                if sep is None:
                    hit = (parse_number(v), None)
                else:
                    role, reason = decisions[sep]
                    hint = sep if role == "decimal" else ("," if sep == "." else ".")
                    hit = (parse_number(v, decimal=hint), reason)
                cache[v] = hit
            num, reason = hit
            out.append(num)
            if reason is not None:
                ambiguous.append({"value": v.strip(), "read_as": num, "reason": reason})
        else:
            out.append(parse_number(v))
    return out, fmt, ambiguous


def _round_half_away(x: float) -> int:
    """Round half away from zero (2.5 -> 3, -2.5 -> -3), unlike Python's round()."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def parse_boolean(value: Any) -> bool | None:
    """Parse a boolean from many spellings; ``None`` if not recognisable."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if _is_missing(value):
        return None
    s = str(value).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def _canonical_key(value: Any) -> str:
    """A comparison key: accent-free, lowercased, punctuation-collapsed.

    Textual placeholders ('N/A', 'unknown') count as empty.
    """
    if _is_missing(value):
        return ""
    return _text_key(value)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_text_dtype(series: pd.Series) -> bool:
    """True for object and any string-like column (incl. pandas 3.0 ``str``)."""
    if series.dtype == object:
        return True
    try:
        return bool(pd.api.types.is_string_dtype(series.dtype))
    except (TypeError, ValueError):
        return str(series.dtype) in ("string", "str")


def _is_int_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)


def _jsonable_label(label: Any) -> Any:
    if isinstance(label, (np.integer,)):
        return int(label)
    if isinstance(label, (int, str)):
        return label
    return str(label)


def _short(value: Any, width: int = 60) -> Any:
    if value is None or (not isinstance(value, str) and _is_missing(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    s = str(value)
    return s if len(s) <= width else s[: width - 1] + "…"


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

# canonical keys only contain these characters, so a 37-slot count vector is
# an exact multiset representation (used for the quick_ratio upper bound).
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 "
_CHAR_INDEX = {ch: i for i, ch in enumerate(_ALPHABET)}


def _char_counts(s: str) -> np.ndarray:
    vec = np.zeros(len(_ALPHABET), dtype=np.int32)
    for ch in s:
        vec[_CHAR_INDEX.get(ch, len(_ALPHABET) - 1)] += 1
    return vec


def fuzzy_matches(
    signatures: Sequence[str],
    threshold: float,
    order: Sequence[int],
    blocks: Sequence[Any] | None = None,
) -> dict[int, tuple[int, float]]:
    """Greedy fuzzy clustering with exact pruning.

    Rows are visited in ``order`` (positions into ``signatures``). A row whose
    signature scores ``SequenceMatcher(None, sig, kept).ratio() >= threshold``
    against any previously kept row of the same block is a duplicate;
    otherwise it becomes a kept row. Returns ``{dup_position: (kept_position,
    score)}`` with the best-scoring kept row.

    The result is identical to comparing every pair with ``ratio()``: pairs
    are only skipped when an exact upper bound of ``ratio()`` (the length
    bound behind ``real_quick_ratio`` and the character-multiset bound behind
    ``quick_ratio``) is already below the threshold.
    """
    if blocks is None:
        blocks = [s[:1] for s in signatures]
    members: dict[Any, list[int]] = {}
    for pos in order:
        members.setdefault(blocks[pos], []).append(pos)

    result: dict[int, tuple[int, float]] = {}
    for rows in members.values():
        cap = len(rows)
        kept_counts = np.zeros((cap, len(_ALPHABET)), dtype=np.int32)
        kept_len = np.zeros(cap, dtype=np.float64)
        kept_pos: list[int] = []
        kept_sm: list[SequenceMatcher] = []
        n_kept = 0
        for pos in rows:
            s = signatures[pos]
            if not s:
                continue
            la = len(s)
            best: tuple[int, float] | None = None
            if n_kept:
                counts = _char_counts(s)
                totals = kept_len[:n_kept] + la
                # Length bound (== real_quick_ratio) and multiset bound (== quick_ratio).
                length_ub = 2.0 * np.minimum(kept_len[:n_kept], la) / totals
                cand = np.nonzero(length_ub >= threshold)[0]
                if cand.size:
                    inter = np.minimum(kept_counts[cand], counts).sum(axis=1)
                    quick_ub = 2.0 * inter / totals[cand]
                    cand = cand[quick_ub >= threshold]
                for k in cand:
                    sm = kept_sm[k]
                    sm.set_seq1(s)
                    score = sm.ratio()
                    if score >= threshold and (best is None or score > best[1]):
                        best = (kept_pos[k], score)
            if best is not None:
                result[pos] = best
            else:
                kept_counts[n_kept] = _char_counts(s)
                kept_len[n_kept] = la
                kept_pos.append(pos)
                kept_sm.append(SequenceMatcher(None, "", s))
                n_kept += 1
    return result


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


class Cleaner:
    """A DataFrame wrapper that records an audit log of every change."""

    def __init__(self, df: pd.DataFrame, log: list[dict[str, Any]] | None = None):
        self.df = df.copy()
        if not self.df.index.is_unique:
            self.df = self.df.reset_index(drop=True)
        self.log: list[dict[str, Any]] = log if log is not None else []
        # Duplicate pairs with the full values of both rows, for review files.
        self.review: list[dict[str, Any]] = []

    # -- logging ------------------------------------------------------------
    def _record(self, op: str, message: str, **details: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"op": op, "message": message}
        entry.update(details)
        self.log.append(entry)
        return entry

    def messages(self) -> list[str]:
        return [e["message"] for e in self.log]

    def _skip_missing_column(self, op: str, col: str) -> None:
        self._record(op, f"Skipped '{col}': column not found.", columns=[col], skipped=True)

    # -- column names -------------------------------------------------------
    def standardize_column_names(self) -> "Cleaner":
        new_cols = snake_case_columns(self.df.columns)
        renames = {
            str(old): new for old, new in zip(self.df.columns, new_cols) if str(old) != new
        }
        self.df.columns = new_cols
        self._record(
            "standardize_column_names",
            f"Renamed {len(renames)} column(s) to snake_case.",
            renames=renames,
            columns=list(renames.keys()),
        )
        return self

    # -- whitespace / unicode ----------------------------------------------
    def normalize_whitespace(
        self, columns: Sequence[str] | None = None, form: str = "NFKC"
    ) -> "Cleaner":
        cols = self._object_columns(_as_list(columns))
        changed = 0
        blanked = 0
        affected: list[str] = []
        blank_rows: dict[str, list[Any]] = {}
        for col in cols:
            before = self.df[col]
            obj = before.astype(object)
            after = _map_unique(obj, lambda v: normalize_text(v, form))
            n = 0
            rows_blank: list[Any] = []
            for label, a, b in zip(obj.index, obj.tolist(), after.tolist()):
                if isinstance(a, str) and not isinstance(b, str):
                    rows_blank.append(_jsonable_label(label))
                    n += 1
                elif isinstance(a, str) and a != b:
                    n += 1
            if n:
                changed += n
                affected.append(col)
                try:
                    self.df[col] = after.astype(before.dtype) if before.dtype != object else after
                except (TypeError, ValueError):
                    self.df[col] = after
            if rows_blank:
                blanked += len(rows_blank)
                blank_rows[col] = rows_blank[:20]
        msg = (
            f"Normalised whitespace/unicode in {changed} cell(s) "
            f"across {len(affected)} column(s)."
        )
        if blanked:
            msg += f" {blanked} whitespace-only cell(s) became null."
        self._record(
            "normalize_whitespace",
            msg,
            columns=affected,
            cells_changed=changed,
            blank_to_null=blanked,
            blank_rows=blank_rows,
        )
        return self

    # -- type coercion ------------------------------------------------------
    def coerce_types(
        self, mapping: dict[str, str] | None = None, on_fraction: str = "round"
    ) -> "Cleaner":
        """Coerce columns to target types.

        ``mapping`` maps a column name to one of ``integer``, ``float``,
        ``boolean``, ``datetime``, ``string``. If ``mapping`` is ``None`` the
        type is inferred per column via the profiler.

        ``on_fraction`` decides what an ``integer`` column does with a value
        such as ``2.5``: ``round`` (half away from zero, the default),
        ``nullify`` (becomes null) or ``error`` (raise). Either way the
        affected values are counted in the log.
        """
        if on_fraction is None:
            raise ValueError(
                "on_fraction is null; YAML reads a bare 'null' as nothing. "
                "Use one of: round, nullify, error."
            )
        check_choice("on_fraction", on_fraction, ON_FRACTION)
        if mapping is None:
            mapping = self._infer_type_mapping()
        for col, target in mapping.items():
            if col not in self.df.columns:
                self._skip_missing_column("coerce_types", col)
                continue
            self._coerce_one(col, target, on_fraction)
        return self

    def _coerce_one(self, col: str, target: str, on_fraction: str = "round") -> None:
        series = self.df[col]
        canonical = TYPE_TARGETS.get(str(target).lower())
        if canonical is None:
            self._record(
                "coerce_types",
                f"Unknown target type '{target}' for '{col}'; left unchanged.",
                columns=[col],
                skipped=True,
            )
            return
        missing_before = _missing_series(series)
        details: dict[str, Any] = {}
        notes: list[str] = []

        if canonical in ("integer", "float"):
            nums, fmt, ambiguous = parse_number_column(series.tolist())
            if ambiguous:
                details["ambiguous"] = ambiguous[:20]
                ex = ambiguous[0]
                notes.append(
                    f"{len(ambiguous)} ambiguous value(s) read from column context "
                    f"(e.g. '{ex['value']}' -> {_fmt_num(ex['read_as'])})."
                )
            if canonical == "integer":
                new: list[Any] = []
                fractional: list[dict[str, Any]] = []
                for label, raw, num in zip(series.index, series.tolist(), nums):
                    if num is None:
                        new.append(pd.NA)
                    elif float(num).is_integer():
                        new.append(int(num))
                    else:
                        to = _round_half_away(num) if on_fraction == "round" else None
                        fractional.append(
                            {"row": _jsonable_label(label), "value": _short(raw), "to": to}
                        )
                        new.append(pd.NA if to is None else to)
                if fractional and on_fraction == "error":
                    ex = ", ".join(repr(f["value"]) for f in fractional[:3])
                    raise ValueError(
                        f"coerce_types: '{col}' has {len(fractional)} non-integer "
                        f"value(s) ({ex}); set on_fraction to round or nullify"
                    )
                self.df[col] = pd.array(new, dtype="Int64")
                details["rounded" if on_fraction == "round" else "nullified"] = len(fractional)
                if fractional:
                    details["fraction_examples"] = fractional[:20]
                    ex = fractional[0]
                    if on_fraction == "round":
                        notes.append(
                            f"{len(fractional)} fractional value(s) were rounded "
                            f"(e.g. '{ex['value']}' -> {ex['to']})."
                        )
                    else:
                        notes.append(
                            f"{len(fractional)} fractional value(s) became null "
                            f"(e.g. '{ex['value']}')."
                        )
            else:
                self.df[col] = pd.array(nums, dtype="Float64")
        elif canonical == "boolean":
            self.df[col] = pd.array([parse_boolean(v) for v in series], dtype="boolean")
        elif canonical == "datetime":
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                dayfirst = False
            else:
                dayfirst = detect_dayfirst(series.dropna().astype(str).tolist())
                _, _, ambiguous_dates = dayfirst_evidence(series.dropna().astype(str).tolist())
                details["dayfirst"] = dayfirst
                if dayfirst and ambiguous_dates:
                    details["ambiguous_dates"] = ambiguous_dates
                    notes.append(
                        f"Day-first detected; {ambiguous_dates} ambiguous value(s) "
                        "such as 05/02/2023 were read day-first."
                    )
            self.df[col] = parse_dates(series, dayfirst=dayfirst)
        else:  # string
            self.df[col] = pd.array(
                [pd.NA if _is_missing(v) else str(v) for v in series], dtype="string"
            )

        now_null = pd.isna(self.df[col]).to_numpy()
        unparsed_mask = now_null & ~missing_before.to_numpy()
        n_nullified = details.get("nullified", 0)
        unparsed = int(unparsed_mask.sum()) - n_nullified
        msg = f"Coerced '{col}' to {target}."
        if unparsed:
            examples = [
                _short(v)
                for v, bad in zip(series.tolist(), unparsed_mask)
                if bad and not (isinstance(v, str) and ambiguous_separator(v))
            ][:5]
            details["unparsed_examples"] = examples
            msg += f" {unparsed} value(s) could not be parsed and became null"
            msg += f" (e.g. {examples[0]!r})." if examples else "."
        if notes:
            msg += " " + " ".join(notes)
        self._record(
            "coerce_types",
            msg,
            columns=[col],
            target=target,
            unparsed=max(0, unparsed),
            **details,
        )

    # -- categoricals -------------------------------------------------------
    def standardize_categoricals(
        self,
        column: str,
        canonical: Sequence[str],
        threshold: float = 0.8,
        extra_aliases: dict[str, str] | None = None,
    ) -> "Cleaner":
        """Fuzzy-map each value in ``column`` to the closest canonical label."""
        if column not in self.df.columns:
            self._skip_missing_column("standardize_categoricals", column)
            return self
        canonical = _as_list(canonical) or []

        canon_keys = {c: _canonical_key(c) for c in canonical}
        alias_keys = {_canonical_key(k): v for k, v in (extra_aliases or {}).items()}
        mapping: dict[str, str] = {}
        unmatched: list[str] = []
        cache: dict[str, str | None] = {}
        counts: Counter = Counter()

        def resolve(raw: Any) -> Any:
            if _is_missing(raw):
                return raw
            key = _canonical_key(raw)
            if key in cache:
                mapped = cache[key]
                if mapped is not None and str(raw) != mapped:
                    counts[str(raw)] += 1
                    mapping[str(raw)] = mapped
                return mapped if mapped is not None else raw
            if key in alias_keys:
                cache[key] = alias_keys[key]
                if str(raw) != alias_keys[key]:
                    mapping[str(raw)] = alias_keys[key]
                    counts[str(raw)] += 1
                return alias_keys[key]
            best_label, best_score = None, 0.0
            for label, ckey in canon_keys.items():
                score = 1.0 if key == ckey else _similarity(key, ckey)
                if score > best_score:
                    best_label, best_score = label, score
            if best_label is not None and best_score >= threshold:
                cache[key] = best_label
                if str(raw) != best_label:
                    mapping[str(raw)] = best_label
                    counts[str(raw)] += 1
                return best_label
            cache[key] = None
            if str(raw) not in unmatched:
                unmatched.append(str(raw))
            return raw

        self.df[column] = self.df[column].astype(object).map(resolve)
        cells = int(sum(counts.values()))
        self._record(
            "standardize_categoricals",
            f"Standardised '{column}': mapped {len(mapping)} variant(s) to "
            f"{len(canonical)} canonical value(s) in {cells} cell(s); "
            f"{len(unmatched)} left unmatched.",
            columns=[column],
            mapping=mapping,
            unmatched=unmatched,
            cells_changed=cells,
            threshold=threshold,
        )
        return self

    # -- missing values -----------------------------------------------------
    def handle_missing(
        self,
        strategy: str = "flag",
        columns: Sequence[str] | None = None,
        value: Any = None,
        how: str = "any",
    ) -> "Cleaner":
        """Handle missing values.

        strategy:
            ``drop_rows``     drop rows with missing values in ``columns``
                              (``how="any"``: any of them; ``"all"``: all)
            ``drop_columns``  drop columns whose values are all missing
            ``fill``          fill with ``value`` or a keyword: ``mean``,
                              ``median``, ``mode``, ``ffill``, ``bfill``, or a
                              literal constant. Typed columns stay typed: an
                              integer column gets a rounded fill value (and
                              the log says so).
            ``flag``          add a boolean ``has_missing`` column (default)
        """
        check_choice("strategy", strategy, MISSING_STRATEGIES)
        check_choice("how", how, MISSING_HOW)
        cols = _as_list(columns) or list(self.df.columns)
        cols = [c for c in cols if c in self.df.columns]
        missing = _missing_frame(self.df[cols]) if cols else pd.DataFrame(index=self.df.index)

        if strategy == "drop_rows":
            before = len(self.df)
            mask = missing.all(axis=1) if how == "all" else missing.any(axis=1)
            if not cols:
                mask = pd.Series(False, index=self.df.index)
            dropped_rows = [_jsonable_label(i) for i in self.df.index[mask]]
            self.df = self.df[~mask]
            dropped = before - len(self.df)
            self._record(
                "handle_missing",
                f"Dropped {dropped} row(s) with missing values in "
                f"{'all' if how == 'all' else 'any'} of {len(cols)} column(s).",
                columns=cols,
                how=how,
                rows_before=before,
                rows_after=len(self.df),
                dropped_rows=dropped_rows[:100],
            )
            return self

        if strategy == "drop_columns":
            to_drop = [c for c in cols if missing[c].all()]
            self.df = self.df.drop(columns=to_drop)
            self._record(
                "handle_missing",
                f"Dropped {len(to_drop)} fully-empty column(s).",
                columns=to_drop,
            )
            return self

        if strategy == "fill":
            if value is None:
                raise ValueError(
                    "handle_missing: strategy 'fill' needs a value "
                    "(mean, median, mode, ffill, bfill or a constant)"
                )
            filled_total = 0
            touched: list[str] = []
            fills: dict[str, Any] = {}
            notes: list[str] = []
            for col in cols:
                mask = missing[col]
                n = int(mask.sum())
                if not n:
                    continue
                fill_value, note = self._resolve_fill(col, value)
                if fill_value is None:
                    continue
                if note:
                    notes.append(note)
                if isinstance(fill_value, str) and fill_value in ("__ffill__", "__bfill__"):
                    series = self.df[col].mask(mask)
                    self.df[col] = series.ffill() if fill_value == "__ffill__" else series.bfill()
                    n = n - int(pd.isna(self.df[col]).sum())
                else:
                    # A string column cannot hold a non-string fill value
                    # (pandas 3.0 'str' dtype is strict); widen to object.
                    if not isinstance(fill_value, str) and _is_text_dtype(self.df[col]):
                        self.df[col] = self.df[col].astype(object)
                    self.df.loc[mask, col] = fill_value
                    fills[col] = _short(fill_value)
                filled_total += n
                touched.append(col)
            msg = (
                f"Filled {filled_total} missing value(s) across "
                f"{len(touched)} column(s) using '{value}'."
            )
            if notes:
                msg += " " + " ".join(notes)
            self._record(
                "handle_missing",
                msg,
                columns=touched,
                cells_changed=filled_total,
                fill=str(value),
                fill_values=fills,
            )
            return self

        # Default: flag.
        flag_col = "has_missing"
        self.df[flag_col] = missing.any(axis=1) if cols else False
        n = int(self.df[flag_col].sum())
        self._record(
            "handle_missing",
            f"Flagged {n} row(s) with missing values in new column '{flag_col}'.",
            columns=cols,
            flag_column=flag_col,
        )
        return self

    def _numeric_values(self, col: str) -> pd.Series:
        """The column as floats (NaN where not numeric), text parsed in context."""
        series = self.df[col]
        if pd.api.types.is_bool_dtype(series.dtype):
            return pd.Series(np.nan, index=series.index, dtype="float64")
        if pd.api.types.is_numeric_dtype(series.dtype):
            return series.astype("Float64").astype("float64")
        nums, _, _ = parse_number_column(series.tolist())
        return pd.Series(
            [np.nan if n is None else n for n in nums], index=series.index, dtype="float64"
        )

    def _resolve_fill(self, col: str, value: Any) -> tuple[Any, str]:
        """Turn a fill keyword into a concrete value that fits the column's dtype."""
        if value == "ffill":
            return "__ffill__", ""
        if value == "bfill":
            return "__bfill__", ""
        series = self.df[col]
        present = series[~_missing_series(series)]
        is_int = _is_int_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series.dtype)
        is_dt = pd.api.types.is_datetime64_any_dtype(series.dtype)

        if value in ("mean", "median"):
            if is_bool:
                raise ValueError(
                    f"handle_missing: cannot fill boolean column '{col}' with the "
                    f"{value}; use 'mode' or a constant"
                )
            if is_dt:
                if present.empty:
                    return None, ""
                return (present.mean() if value == "mean" else present.median()), ""
            nums = self._numeric_values(col).dropna()
            if not len(nums):
                return None, ""
            result = float(nums.mean() if value == "mean" else nums.median())
        elif value == "mode":
            if present.empty:
                return None, ""
            return present.mode().iloc[0], ""
        else:
            result = value

        if is_int:
            num = parse_number(result) if not isinstance(result, (int, float)) else float(result)
            if num is None:
                raise ValueError(
                    f"handle_missing: cannot fill integer column '{col}' with {result!r}"
                )
            if not float(num).is_integer():
                rounded = _round_half_away(num)
                return rounded, (
                    f"'{col}' is an integer column: fill value {_fmt_num(num)} "
                    f"was rounded to {rounded}."
                )
            return int(num), ""
        if is_bool:
            b = parse_boolean(result)
            if b is None:
                raise ValueError(
                    f"handle_missing: cannot fill boolean column '{col}' with {result!r}"
                )
            return b, ""
        if pd.api.types.is_float_dtype(series.dtype):
            num = parse_number(result) if not isinstance(result, (int, float)) else float(result)
            if num is None:
                raise ValueError(
                    f"handle_missing: cannot fill numeric column '{col}' with {result!r}"
                )
            return num, ""
        if is_dt and not isinstance(result, pd.Timestamp):
            ts = parse_dates(pd.Series([result], dtype=object)).iloc[0]
            if pd.isna(ts):
                raise ValueError(
                    f"handle_missing: cannot fill date column '{col}' with {result!r}"
                )
            return ts, ""
        return result, ""

    # -- duplicates ---------------------------------------------------------
    def deduplicate(
        self,
        subset: Sequence[str] | None = None,
        keep: str = "first",
        fuzzy: bool = False,
        fuzzy_keys: Sequence[str] | None = None,
        threshold: float = 0.9,
        flag: bool = False,
        block_on: Sequence[str] | None = None,
    ) -> "Cleaner":
        """Remove or flag duplicate rows.

        Exact duplicates are matched on ``subset`` (or all columns). When
        ``fuzzy`` is true, rows whose ``fuzzy_keys`` are similar above
        ``threshold`` (``difflib`` ratio) are treated as duplicates of the
        kept occurrence. Fuzzy comparisons only happen inside a block: rows
        sharing the first character of the key and, if given, the values of
        the ``block_on`` columns (e.g. country or email domain).

        ``keep`` chooses the survivor of each group: ``first``, ``last``, or
        ``most_complete`` (fewest missing values; ties go to the first).
        Every removed or flagged row is logged as a pair with the row it
        duplicates, the match type and the score.
        """
        check_choice("keep", keep, KEEP_OPTIONS)
        if not (0 < float(threshold) <= 1):
            raise ValueError(f"threshold={threshold!r} must be in (0, 1]")
        subset = _as_list(subset)
        block_on = _as_list(block_on)
        for col in (subset or []) + (block_on or []):
            if col not in self.df.columns:
                raise ValueError(f"deduplicate: column '{col}' not found")

        n = len(self.df)
        labels = list(self.df.index)
        order = self._keep_order(keep)
        rank = np.empty(n, dtype=np.int64)
        rank[np.asarray(order, dtype=np.int64)] = np.arange(n)

        # Exact duplicates: within each group the lowest-ranked row survives.
        pairs: dict[int, dict[str, Any]] = {}
        if n:
            sub = self.df[subset] if subset else self.df
            gid = sub.astype(object).groupby(list(sub.columns), dropna=False, sort=False).ngroup().to_numpy()
            frame = pd.DataFrame({"g": gid, "rank": rank})
            winner_rank = frame.groupby("g")["rank"].transform("min").to_numpy()
            pos_by_rank = np.asarray(order, dtype=np.int64)
            for pos in np.nonzero(rank != winner_rank)[0]:
                kept = int(pos_by_rank[winner_rank[pos]])
                pairs[int(pos)] = {"match": "exact", "kept": kept, "score": 1.0}

        n_exact = len(pairs)
        n_fuzzy = 0
        if fuzzy and n:
            keys = [k for k in (_as_list(fuzzy_keys) or subset or list(self.df.columns)) if k in self.df.columns]
            if keys:
                sigs = self._signatures(keys)
                survivors = [p for p in order if p not in pairs]
                blocks = self._blocks(sigs, block_on)
                for pos, (kept, score) in fuzzy_matches(sigs, threshold, survivors, blocks).items():
                    pairs[pos] = {
                        "match": "fuzzy",
                        "kept": kept,
                        "score": round(float(score), 4),
                        "key": sigs[pos],
                        "kept_key": sigs[kept],
                    }
                n_fuzzy = len(pairs) - n_exact

        dup_pos = sorted(pairs)
        dup_mask = np.zeros(n, dtype=bool)
        dup_mask[dup_pos] = True
        log_pairs = []
        for pos in dup_pos:
            p = pairs[pos]
            entry = {
                "row": _jsonable_label(labels[pos]),
                "kept_row": _jsonable_label(labels[p["kept"]]),
                "match": p["match"],
                "score": p["score"],
            }
            if "key" in p:
                entry["key"] = p["key"]
                entry["kept_key"] = p["kept_key"]
            log_pairs.append(entry)
            self.review.append(
                {
                    **entry,
                    "values": {str(c): _short(v, 200) for c, v in self.df.iloc[pos].items()},
                    "kept_values": {
                        str(c): _short(v, 200) for c, v in self.df.iloc[p["kept"]].items()
                    },
                }
            )
        n_dupes = len(dup_pos)
        how = f"{n_exact} exact on {subset if subset else 'all columns'}"
        if fuzzy:
            how += f", {n_fuzzy} fuzzy at threshold {threshold}"
        common = dict(
            columns=subset or [],
            duplicates=n_dupes,
            exact=n_exact,
            fuzzy=n_fuzzy,
            keep=keep,
            pairs=log_pairs,
        )
        if flag:
            self.df["is_duplicate"] = dup_mask
            dup_of = [pd.NA] * n
            for pos in dup_pos:
                dup_of[pos] = _jsonable_label(labels[pairs[pos]["kept"]])
            try:
                self.df["duplicate_of"] = pd.array(dup_of, dtype="Int64")
            except (TypeError, ValueError):
                self.df["duplicate_of"] = pd.array(dup_of, dtype=object)
            self._record(
                "deduplicate",
                f"Flagged {n_dupes} duplicate row(s) ({how}) in new columns "
                "'is_duplicate' and 'duplicate_of'.",
                flagged=True,
                **common,
            )
        else:
            before = len(self.df)
            self.df = self.df[~dup_mask]
            self._record(
                "deduplicate",
                f"Removed {n_dupes} duplicate row(s) ({how}), keeping the "
                f"{keep.replace('_', ' ')} of each group.",
                rows_before=before,
                rows_after=len(self.df),
                **common,
            )
        return self

    def _keep_order(self, keep: str) -> list[int]:
        n = len(self.df)
        if keep == "last":
            return list(range(n - 1, -1, -1))
        if keep == "most_complete":
            n_missing = _missing_frame(self.df).sum(axis=1).to_numpy()
            return sorted(range(n), key=lambda i: (int(n_missing[i]), i))
        return list(range(n))

    def _signatures(self, keys: Sequence[str]) -> list[str]:
        cols = [self.df[k].astype(object).tolist() for k in keys]
        return [" ".join(_canonical_key(v) for v in row) for row in zip(*cols)]

    def _blocks(self, sigs: Sequence[str], block_on: Sequence[str] | None) -> list[Any]:
        if not block_on:
            return [s[:1] for s in sigs]
        extra = [self.df[c].astype(object).map(_canonical_key).tolist() for c in block_on]
        return [(s[:1],) + tuple(vals) for s, vals in zip(sigs, zip(*extra))]

    def _fuzzy_duplicate_mask(
        self, keys: Sequence[str], threshold: float, keep: str
    ) -> pd.Series:
        """Boolean mask of fuzzy duplicates (the pre-0.2 helper's contract)."""
        keys = [k for k in keys if k in self.df.columns]
        mask = pd.Series(False, index=self.df.index)
        if not keys:
            return mask
        sigs = self._signatures(keys)
        order = self._keep_order(keep)
        for pos in fuzzy_matches(sigs, threshold, order):
            mask.iloc[pos] = True
        return mask

    # -- outliers -----------------------------------------------------------
    def handle_outliers(
        self,
        columns: Sequence[str],
        method: str = "iqr",
        action: str = "flag",
        factor: float = 1.5,
        z: float = 3.0,
    ) -> "Cleaner":
        """Detect outliers per column and either cap or flag them.

        method: ``iqr`` (Tukey fences) or ``zscore``.
        action: ``cap`` (winsorize to the bound) or ``flag`` (add a boolean
        column ``<col>_is_outlier``). Capping keeps the column's dtype: an
        integer column is capped to the whole numbers inside the fences.
        """
        check_choice("method", method, OUTLIER_METHODS)
        check_choice("action", action, OUTLIER_ACTIONS)
        touched: list[str] = []
        total = 0
        fences: dict[str, list[float]] = {}
        examples: dict[str, list[Any]] = {}
        for col in _as_list(columns) or []:
            if col not in self.df.columns:
                continue
            nums = self._numeric_values(col)
            valid = nums.dropna()
            if len(valid) < 4:
                continue
            if method == "zscore":
                mean, std = valid.mean(), valid.std(ddof=0)
                if std == 0:
                    continue
                low, high = mean - z * std, mean + z * std
            else:  # iqr
                q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                low, high = q1 - factor * iqr, q3 + factor * iqr

            series = self.df[col]
            if _is_int_dtype(series):
                low, high = math.ceil(low), math.floor(high)
            outlier_mask = ((nums < low) | (nums > high)).fillna(False)
            k = int(outlier_mask.sum())
            if not k:
                continue
            total += k
            touched.append(col)
            fences[col] = [float(low), float(high)]
            examples[col] = [
                {"row": _jsonable_label(i), "value": _short(series[i])}
                for i in series.index[outlier_mask.to_numpy()][:10]
            ]
            if action == "cap":
                if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
                    self.df[col] = series.clip(lower=low, upper=high)
                else:
                    capped = nums.clip(lower=low, upper=high)
                    self.df[col] = capped.where(~nums.isna(), series)
            else:
                self.df[f"{col}_is_outlier"] = outlier_mask.to_numpy()
        self._record(
            "handle_outliers",
            f"{'Capped' if action == 'cap' else 'Flagged'} {total} outlier(s) "
            f"across {len(touched)} column(s) via {method}.",
            columns=touched,
            outliers=total,
            method=method,
            action=action,
            fences=fences,
            examples=examples,
        )
        return self

    # -- contacts -----------------------------------------------------------
    def _contact_columns(self, op: str, columns: Any) -> list[str]:
        cols = _as_list(columns)
        if not cols:
            raise ValueError(f"{op}: 'columns' must name at least one column")
        present = []
        for col in cols:
            if col in self.df.columns:
                present.append(col)
            else:
                self._skip_missing_column(op, col)
        return present

    def _apply_invalid(self, col: str, new: list[Any], invalid_rows: list[int], originals: list[Any], invalid: str) -> None:
        if invalid == "null":
            for pos in invalid_rows:
                new[pos] = None
        elif invalid == "flag":
            flags = np.zeros(len(new), dtype=bool)
            flags[invalid_rows] = True
            self.df[f"{col}_is_invalid"] = flags
        self.df[col] = pd.Series(new, index=self.df.index, dtype=object)

    def standardize_emails(
        self, columns: Sequence[str], fix_typos: bool = True, invalid: str = "keep"
    ) -> "Cleaner":
        """Lower-case, trim and repair provider typos (``gmial.con``) in email columns.

        ``invalid`` decides what happens to addresses that are still malformed:
        ``keep`` (leave the cleaned text), ``null`` (blank it) or ``flag``
        (add ``<col>_is_invalid``). Suggestions for addresses that are only
        missing their ``@`` are logged, never applied.
        """
        check_choice("invalid", invalid, INVALID_ACTIONS)
        for col in self._contact_columns("standardize_emails", columns):
            originals = self.df[col].astype(object).tolist()
            labels = list(self.df.index)
            new: list[Any] = []
            invalid_rows: list[int] = []
            bad: list[dict[str, Any]] = []
            changed = typos = 0
            cache: dict[str, Any] = {}
            for pos, raw in enumerate(originals):
                if _is_missing(raw):
                    new.append(None if isinstance(raw, str) else raw)
                    continue
                res = cache.get(raw) or cache.setdefault(raw, _contacts.standardize_email(raw, fix_typos))
                new.append(res.value)
                if res.value != raw:
                    changed += 1
                if res.note == "fixed domain typo":
                    typos += 1
                if not res.valid:
                    invalid_rows.append(pos)
                    item = {"row": _jsonable_label(labels[pos]), "value": _short(raw)}
                    if res.suggestion:
                        item["suggestion"] = res.suggestion
                    bad.append(item)
            self._apply_invalid(col, new, invalid_rows, originals, invalid)
            msg = (
                f"Standardised emails in '{col}': {changed} cell(s) changed, "
                f"{typos} domain typo(s) fixed, {len(bad)} invalid"
            )
            msg += f" ({'kept' if invalid == 'keep' else 'set to null' if invalid == 'null' else 'flagged'})." if bad else "."
            self._record(
                "standardize_emails",
                msg,
                columns=[col],
                cells_changed=changed,
                typos_fixed=typos,
                invalid=len(bad),
                invalid_examples=bad[:20],
                on_invalid=invalid,
            )
        return self

    def standardize_phones(
        self,
        columns: Sequence[str],
        default_region: str = "PA",
        region_column: str | None = None,
        invalid: str = "keep",
    ) -> "Cleaner":
        """Rewrite phone numbers as E.164 (``+50761234567``).

        Each number is read in the region of its own row when
        ``region_column`` names a country column (``Panamá``, ``Colombia``,
        ``México``, ``USA`` ... map to PA, CO, MX, US), else in
        ``default_region``. A number that does not fit the row's numbering
        plan is retried with the default region and then as an international
        number typed without ``+``; every such fallback is logged.

        ``invalid`` handles numbers no reading accepts: ``keep`` leaves the
        original text untouched, ``null`` blanks it, ``flag`` keeps it and
        adds ``<col>_is_invalid``.
        """
        check_choice("invalid", invalid, INVALID_ACTIONS)
        region = _contacts._check_region(default_region)
        region_values: list[Any] | None = None
        region_note = ""
        if region_column:
            if region_column in self.df.columns:
                region_values = self.df[region_column].astype(object).tolist()
            else:
                region_note = f" Region column '{region_column}' not found; used {region}."
        country_cache: dict[Any, str | None] = {}
        phone_cache: dict[tuple[Any, str], tuple[Any, str | None]] = {}

        for col in self._contact_columns("standardize_phones", columns):
            originals = self.df[col].astype(object).tolist()
            labels = list(self.df.index)
            new: list[Any] = []
            invalid_rows: list[int] = []
            bad: list[dict[str, Any]] = []
            fallbacks: list[dict[str, Any]] = []
            regions_used: Counter = Counter()
            changed = 0
            for pos, raw in enumerate(originals):
                if _is_missing(raw):
                    new.append(None if isinstance(raw, str) else raw)
                    continue
                row_region = region
                if region_values is not None:
                    country = region_values[pos]
                    if country not in country_cache:
                        country_cache[country] = _contacts.region_for_country(country)
                    row_region = country_cache[country] or region
                key = (raw, row_region)
                if key not in phone_cache:
                    phone_cache[key] = self._read_phone(raw, row_region, region)
                res, via = phone_cache[key]
                if res.valid:
                    regions_used[res.region] += 1
                    new.append(res.value)
                    if res.value != raw:
                        changed += 1
                    if via:
                        fallbacks.append(
                            {
                                "row": _jsonable_label(labels[pos]),
                                "value": _short(raw),
                                "row_region": row_region,
                                "read_as": res.value,
                                "via": via,
                            }
                        )
                else:
                    new.append(raw)
                    invalid_rows.append(pos)
                    bad.append(
                        {
                            "row": _jsonable_label(labels[pos]),
                            "value": _short(raw),
                            "region": row_region,
                            "note": res.note,
                        }
                    )
            self._apply_invalid(col, new, invalid_rows, originals, invalid)
            msg = (
                f"Standardised phones in '{col}' to E.164: {changed} cell(s) changed, "
                f"{len(bad)} invalid"
            )
            if bad:
                msg += f" ({'kept as-is' if invalid == 'keep' else 'set to null' if invalid == 'null' else 'flagged'})"
            msg += f", {len(fallbacks)} read outside the row's region." if fallbacks else "."
            msg += region_note
            self._record(
                "standardize_phones",
                msg,
                columns=[col],
                cells_changed=changed,
                invalid=len(bad),
                invalid_examples=bad[:20],
                fallbacks=fallbacks[:50],
                regions=dict(sorted(regions_used.items())),
                default_region=region,
                region_column=region_column,
                on_invalid=invalid,
            )
        return self

    @staticmethod
    def _read_phone(raw: Any, row_region: str, region: str) -> tuple[Any, str | None]:
        """Row region first, then the default region, then 'international without +'."""
        res = _contacts.to_e164(raw, row_region)
        via = None
        if not res.valid and row_region != region:
            alt = _contacts.to_e164(raw, region)
            if alt.valid:
                res, via = alt, f"default region {region}"
        if not res.valid:
            bare = _contacts.to_e164_bare_international(raw)
            if bare is not None:
                res, via = bare, "international number without '+'"
        return res, via

    def standardize_names(self, columns: Sequence[str]) -> "Cleaner":
        """Title-case person names, keeping particles (de, del, la) and Mc/O' forms."""
        for col in self._contact_columns("standardize_names", columns):
            before = self.df[col].astype(object)
            after = _map_unique(before, lambda v: v if (not isinstance(v, str) and _is_missing(v)) else _contacts.standardize_name(v))
            changed = [
                {"row": _jsonable_label(i), "from": _short(a), "to": _short(b)}
                for i, a, b in zip(before.index, before.tolist(), after.tolist())
                if isinstance(a, str) and a != b
            ]
            self.df[col] = after
            self._record(
                "standardize_names",
                f"Standardised names in '{col}': {len(changed)} cell(s) changed.",
                columns=[col],
                cells_changed=len(changed),
                examples=changed[:20],
            )
        return self

    def clean_addresses(self, columns: Sequence[str]) -> "Cleaner":
        """Tidy spacing/punctuation and expand street abbreviations (Ave -> Avenida)."""
        for col in self._contact_columns("clean_addresses", columns):
            before = self.df[col].astype(object)
            after = _map_unique(before, lambda v: v if (not isinstance(v, str) and _is_missing(v)) else _contacts.clean_address(v))
            changed = [
                {"row": _jsonable_label(i), "from": _short(a), "to": _short(b)}
                for i, a, b in zip(before.index, before.tolist(), after.tolist())
                if isinstance(a, str) and a != b
            ]
            self.df[col] = after
            self._record(
                "clean_addresses",
                f"Cleaned addresses in '{col}': {len(changed)} cell(s) changed.",
                columns=[col],
                cells_changed=len(changed),
                examples=changed[:20],
            )
        return self

    # -- helpers ------------------------------------------------------------
    def _object_columns(self, columns: Sequence[str] | None) -> list[str]:
        if columns is not None:
            return [c for c in columns if c in self.df.columns]
        return [c for c in self.df.columns if _is_text_dtype(self.df[c])]

    def _infer_type_mapping(self) -> dict[str, str]:
        from .profile import profile_column

        mapping: dict[str, str] = {}
        for col in self.df.columns:
            t = profile_column(self.df[col]).inferred_type
            if t in {"integer", "float", "boolean", "datetime"}:
                mapping[col] = t
        return mapping

    def apply(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        message: str = "custom step",
    ) -> "Cleaner":
        """Escape hatch: apply an arbitrary transform and log it."""
        before = self.df.shape
        self.df = func(self.df)
        self._record(
            "apply",
            f"{message} (shape {before} -> {self.df.shape}).",
            shape_before=list(before),
            shape_after=list(self.df.shape),
        )
        return self


def _fmt_num(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return f"{v:g}" if isinstance(v, float) else str(v)


def clean(df: pd.DataFrame) -> Cleaner:
    """Convenience constructor: ``clean(df).standardize_column_names()...``."""
    return Cleaner(df)
