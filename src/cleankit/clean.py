"""Composable, logged cleaning operations.

Every operation on :class:`Cleaner` mutates a working copy of the DataFrame and
appends a structured entry to an audit log describing exactly what changed
(columns touched, rows/cells affected, before/after row counts, and a short
human-readable message). Methods return ``self`` so they chain.

    result = (
        Cleaner(df)
        .standardize_column_names()
        .normalize_whitespace()
        .coerce_types({"signup_date": "datetime", "balance": "float"})
        .deduplicate(subset=["email"])
    )
    clean_df = result.df
    for entry in result.log:
        print(entry["message"])
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _NULL_TOKENS:
        return True
    return False


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


def normalize_text(value: Any, form: str = "NFKC") -> Any:
    """Unicode-normalise, replace exotic whitespace, trim, and collapse spaces."""
    if _is_missing(value):
        return value
    s = unicodedata.normalize(form, str(value))
    s = "".join(
        " " if ord(ch) in _WHITESPACE_CODEPOINTS else ch for ch in s
    )
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


def parse_number(value: Any) -> float | None:
    """Parse a number that may carry thousands/decimal separators or currency.

    Understands ``1,234.56`` (US), ``1.234,56`` (EU), ``1 234,56``,
    ``$1,234``, ``(1234)`` for negatives, and trailing ``%``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if (isinstance(value, float) and np.isnan(value)) else float(value)

    s = str(value).strip()
    if s == "" or s.lower() in _NULL_TOKENS:
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    percent = s.endswith("%")

    # Keep digits, separators and sign.
    s = re.sub(r"[^0-9,.\-\s']", "", s)
    s = re.sub(r"\s", "", s).replace("'", "")
    if s in {"", "-", ".", ","}:
        return None
    if s.startswith("-"):
        negative = True
        s = s[1:]

    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        # The right-most separator is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        # Comma is decimal if it appears once with 1-2 trailing digits,
        # otherwise it is a thousands separator.
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = parts[0] + "." + parts[1]
        else:
            s = s.replace(",", "")
    # Only dots: assume standard decimal notation.

    try:
        num = float(s)
    except ValueError:
        return None
    if percent:
        num /= 100.0
    return -num if negative else num


def parse_boolean(value: Any) -> bool | None:
    """Parse a boolean from many spellings; ``None`` if not recognisable."""
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return None
    s = str(value).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    return None


def _strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def _canonical_key(value: Any) -> str:
    """A comparison key: accent-free, lowercased, punctuation-collapsed."""
    if _is_missing(value):
        return ""
    s = _strip_accents(str(value)).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s).strip()
    return s


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


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


class Cleaner:
    """A DataFrame wrapper that records an audit log of every change."""

    def __init__(self, df: pd.DataFrame, log: list[dict[str, Any]] | None = None):
        self.df = df.copy()
        self.log: list[dict[str, Any]] = log if log is not None else []

    # -- logging ------------------------------------------------------------
    def _record(self, op: str, message: str, **details: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"op": op, "message": message}
        entry.update(details)
        self.log.append(entry)
        return entry

    def messages(self) -> list[str]:
        return [e["message"] for e in self.log]

    # -- column names -------------------------------------------------------
    def standardize_column_names(self) -> "Cleaner":
        renames: dict[str, str] = {}
        used: dict[str, int] = {}
        new_cols: list[str] = []
        for col in self.df.columns:
            base = to_snake_case(col)
            name = base
            if name in used:
                used[base] += 1
                name = f"{base}_{used[base]}"
            else:
                used[base] = 0
            new_cols.append(name)
            if name != str(col):
                renames[str(col)] = name
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
        cols = self._object_columns(columns)
        changed = 0
        affected: list[str] = []
        for col in cols:
            before = self.df[col]
            after = before.map(lambda v: normalize_text(v, form))
            n = int((before.astype("object") != after.astype("object")).sum())
            if n:
                changed += n
                affected.append(col)
                self.df[col] = after
        self._record(
            "normalize_whitespace",
            f"Normalised whitespace/unicode in {changed} cell(s) "
            f"across {len(affected)} column(s).",
            columns=affected,
            cells_changed=changed,
        )
        return self

    # -- type coercion ------------------------------------------------------
    def coerce_types(self, mapping: dict[str, str] | None = None) -> "Cleaner":
        """Coerce columns to target types.

        ``mapping`` maps a column name to one of ``integer``, ``float``,
        ``boolean``, ``datetime``, ``string``. If ``mapping`` is ``None`` the
        type is inferred per column via the profiler.
        """
        if mapping is None:
            mapping = self._infer_type_mapping()
        for col, target in mapping.items():
            if col not in self.df.columns:
                self._record(
                    "coerce_types",
                    f"Skipped '{col}': column not found.",
                    columns=[col],
                    skipped=True,
                )
                continue
            self._coerce_one(col, target)
        return self

    def _coerce_one(self, col: str, target: str) -> None:
        series = self.df[col]
        n_before_null = int(series.map(_is_missing).sum())

        if target in ("integer", "int"):
            new = []
            for v in series:
                num = parse_number(v)
                new.append(pd.NA if num is None else int(round(num)))
            self.df[col] = pd.array(new, dtype="Int64")
        elif target in ("float", "number"):
            self.df[col] = pd.array(
                [parse_number(v) for v in series], dtype="Float64"
            )
        elif target in ("boolean", "bool"):
            self.df[col] = pd.array(
                [parse_boolean(v) for v in series], dtype="boolean"
            )
        elif target in ("datetime", "date"):
            dayfirst = self._detect_dayfirst(series)
            self.df[col] = self._parse_datetime(series, dayfirst)
        elif target in ("string", "str", "text"):
            self.df[col] = pd.array(
                [pd.NA if _is_missing(v) else str(v) for v in series],
                dtype="string",
            )
        else:
            self._record(
                "coerce_types",
                f"Unknown target type '{target}' for '{col}'; left unchanged.",
                columns=[col],
                skipped=True,
            )
            return

        n_after_null = int(pd.isna(self.df[col]).sum())
        newly_null = max(0, n_after_null - n_before_null)
        msg = f"Coerced '{col}' to {target}."
        if newly_null:
            msg += f" {newly_null} value(s) could not be parsed and became null."
        self._record(
            "coerce_types",
            msg,
            columns=[col],
            target=target,
            unparsed=newly_null,
        )

    @staticmethod
    def _parse_datetime(series: pd.Series, dayfirst: bool) -> pd.Series:
        """Parse a column of mixed date formats element by element.

        ISO ``YYYY-MM-DD`` values are parsed unambiguously (never day-first),
        so that ``2023-01-05`` stays January 5th even when the column also
        contains day-first ``13/02/2023`` values. Everything else is parsed
        with ``format="mixed"`` (pandas >= 2.0) so each value keeps its own
        format instead of the whole column locking onto row one's format.
        """
        import warnings

        s = series.astype("object")
        iso_mask = s.map(
            lambda v: isinstance(v, str)
            and bool(re.match(r"^\s*\d{4}-\d{1,2}-\d{1,2}", v))
        )
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if iso_mask.any():
                result.loc[iso_mask] = pd.to_datetime(
                    s[iso_mask], errors="coerce"
                )
            rest = ~iso_mask
            if rest.any():
                try:
                    parsed = pd.to_datetime(
                        s[rest], errors="coerce", dayfirst=dayfirst, format="mixed"
                    )
                except (ValueError, TypeError):
                    parsed = pd.to_datetime(
                        s[rest], errors="coerce", dayfirst=dayfirst
                    )
                result.loc[rest] = parsed
        return result

    @staticmethod
    def _detect_dayfirst(series: pd.Series) -> bool:
        """Inspect a date column: if any first component exceeds 12, day-first."""
        pat = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}")
        day_first_votes = 0
        month_first_votes = 0
        for v in series.dropna().astype(str).head(500):
            m = pat.match(v)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12 >= b:
                day_first_votes += 1
            elif b > 12 >= a:
                month_first_votes += 1
        return day_first_votes >= month_first_votes and day_first_votes > 0

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
            self._record(
                "standardize_categoricals",
                f"Skipped '{column}': column not found.",
                columns=[column],
                skipped=True,
            )
            return self

        canon_keys = {c: _canonical_key(c) for c in canonical}
        alias_keys = {_canonical_key(k): v for k, v in (extra_aliases or {}).items()}
        mapping: dict[str, str] = {}
        unmatched: list[str] = []
        cache: dict[str, str | None] = {}

        def resolve(raw: Any) -> Any:
            if _is_missing(raw):
                return raw
            key = _canonical_key(raw)
            if key in cache:
                mapped = cache[key]
                return mapped if mapped is not None else raw
            if key in alias_keys:
                cache[key] = alias_keys[key]
                mapping[str(raw)] = alias_keys[key]
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
                return best_label
            cache[key] = None
            if str(raw) not in unmatched:
                unmatched.append(str(raw))
            return raw

        self.df[column] = self.df[column].map(resolve)
        self._record(
            "standardize_categoricals",
            f"Standardised '{column}': mapped {len(mapping)} variant(s) to "
            f"{len(canonical)} canonical value(s); {len(unmatched)} left unmatched.",
            columns=[column],
            mapping=mapping,
            unmatched=unmatched,
            threshold=threshold,
        )
        return self

    # -- missing values -----------------------------------------------------
    def handle_missing(
        self,
        strategy: str = "flag",
        columns: Sequence[str] | None = None,
        value: Any = None,
    ) -> "Cleaner":
        """Handle missing values.

        strategy:
            ``drop_rows``     drop rows with any missing value in ``columns``
            ``drop_columns``  drop columns whose values are all missing
            ``fill``          fill with ``value`` or a keyword: ``mean``,
                              ``median``, ``mode``, ``ffill``, ``bfill``, or a
                              literal constant
            ``flag``          add a boolean ``has_missing`` column (default)
        """
        cols = list(columns) if columns else list(self.df.columns)
        cols = [c for c in cols if c in self.df.columns]

        if strategy == "drop_rows":
            before = len(self.df)
            mask = self.df[cols].map(_is_missing).any(axis=1)
            self.df = self.df[~mask].reset_index(drop=True)
            dropped = before - len(self.df)
            self._record(
                "handle_missing",
                f"Dropped {dropped} row(s) with missing values in "
                f"{len(cols)} column(s).",
                columns=cols,
                rows_before=before,
                rows_after=len(self.df),
            )
            return self

        if strategy == "drop_columns":
            to_drop = [c for c in cols if self.df[c].map(_is_missing).all()]
            self.df = self.df.drop(columns=to_drop)
            self._record(
                "handle_missing",
                f"Dropped {len(to_drop)} fully-empty column(s).",
                columns=to_drop,
            )
            return self

        if strategy == "fill":
            filled_total = 0
            touched: list[str] = []
            for col in cols:
                fill_value = self._resolve_fill(col, value)
                if fill_value is None:
                    continue
                mask = self.df[col].map(_is_missing)
                n = int(mask.sum())
                if not n:
                    continue
                if fill_value == "__ffill__":
                    self.df[col] = self.df[col].ffill()
                elif fill_value == "__bfill__":
                    self.df[col] = self.df[col].bfill()
                else:
                    # A string column cannot hold a non-string fill value
                    # (pandas 3.0 'str' dtype is strict); widen to object.
                    if not isinstance(fill_value, str) and _is_text_dtype(
                        self.df[col]
                    ):
                        self.df[col] = self.df[col].astype(object)
                    self.df.loc[mask, col] = fill_value
                filled_total += n
                touched.append(col)
            self._record(
                "handle_missing",
                f"Filled {filled_total} missing value(s) across "
                f"{len(touched)} column(s) using '{value}'.",
                columns=touched,
                cells_changed=filled_total,
                fill=str(value),
            )
            return self

        # Default: flag.
        flag_col = "has_missing"
        self.df[flag_col] = self.df[cols].map(_is_missing).any(axis=1)
        n = int(self.df[flag_col].sum())
        self._record(
            "handle_missing",
            f"Flagged {n} row(s) with missing values in new column '{flag_col}'.",
            columns=cols,
            flag_column=flag_col,
        )
        return self

    def _resolve_fill(self, col: str, value: Any) -> Any:
        if value == "ffill":
            return "__ffill__"
        if value == "bfill":
            return "__bfill__"
        present = self.df[col][~self.df[col].map(_is_missing)]
        if value == "mean":
            nums = present.map(parse_number).dropna()
            return float(nums.mean()) if len(nums) else None
        if value == "median":
            nums = present.map(parse_number).dropna()
            return float(nums.median()) if len(nums) else None
        if value == "mode":
            if present.empty:
                return None
            return present.mode().iloc[0]
        return value

    # -- duplicates ---------------------------------------------------------
    def deduplicate(
        self,
        subset: Sequence[str] | None = None,
        keep: str = "first",
        fuzzy: bool = False,
        fuzzy_keys: Sequence[str] | None = None,
        threshold: float = 0.9,
        flag: bool = False,
    ) -> "Cleaner":
        """Remove or flag duplicate rows.

        Exact duplicates are matched on ``subset`` (or all columns). When
        ``fuzzy`` is true, rows whose ``fuzzy_keys`` are similar above
        ``threshold`` are treated as duplicates of the kept occurrence.
        """
        subset = list(subset) if subset else None
        exact_mask = self.df.duplicated(subset=subset, keep=keep)

        fuzzy_mask = pd.Series(False, index=self.df.index)
        if fuzzy:
            keys = list(fuzzy_keys or subset or self.df.columns)
            fuzzy_mask = self._fuzzy_duplicate_mask(keys, threshold, keep)

        dup_mask = exact_mask | fuzzy_mask
        n_dupes = int(dup_mask.sum())

        if flag:
            self.df["is_duplicate"] = dup_mask.values
            self._record(
                "deduplicate",
                f"Flagged {n_dupes} duplicate row(s) in new column "
                f"'is_duplicate' (fuzzy={fuzzy}).",
                columns=subset or [],
                duplicates=n_dupes,
                flagged=True,
            )
        else:
            before = len(self.df)
            self.df = self.df[~dup_mask].reset_index(drop=True)
            self._record(
                "deduplicate",
                f"Removed {n_dupes} duplicate row(s) (exact + "
                f"{'fuzzy' if fuzzy else 'no fuzzy'}).",
                columns=subset or [],
                rows_before=before,
                rows_after=len(self.df),
                duplicates=n_dupes,
            )
        return self

    def _fuzzy_duplicate_mask(
        self, keys: Sequence[str], threshold: float, keep: str
    ) -> pd.Series:
        keys = [k for k in keys if k in self.df.columns]
        if not keys:
            return pd.Series(False, index=self.df.index)

        sig = self.df[keys].astype(object).apply(
            lambda row: " ".join(_canonical_key(v) for v in row), axis=1
        )
        # Block by first character to limit pairwise comparisons.
        blocks: dict[str, list[Any]] = {}
        order = list(sig.index)
        if keep == "last":
            order = list(reversed(order))
        for idx in order:
            blocks.setdefault(sig[idx][:1], []).append(idx)

        mask = pd.Series(False, index=self.df.index)
        for members in blocks.values():
            kept: list[str] = []
            for idx in members:
                s = sig[idx]
                if not s:
                    continue
                if any(_similarity(s, ks) >= threshold for ks in kept):
                    mask[idx] = True
                else:
                    kept.append(s)
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
        column ``<col>_is_outlier``).
        """
        touched: list[str] = []
        total = 0
        for col in columns:
            if col not in self.df.columns:
                continue
            nums = self.df[col].map(parse_number)
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

            outlier_mask = ((nums < low) | (nums > high)).fillna(False)
            n = int(outlier_mask.sum())
            if not n:
                continue
            total += n
            touched.append(col)
            if action == "cap":
                capped = nums.clip(lower=low, upper=high)
                self.df[col] = capped.where(~nums.isna(), self.df[col])
            else:
                self.df[f"{col}_is_outlier"] = outlier_mask.values
        self._record(
            "handle_outliers",
            f"{'Capped' if action == 'cap' else 'Flagged'} {total} outlier(s) "
            f"across {len(touched)} column(s) via {method}.",
            columns=touched,
            outliers=total,
            method=method,
            action=action,
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


def clean(df: pd.DataFrame) -> Cleaner:
    """Convenience constructor: ``clean(df).standardize_column_names()...``."""
    return Cleaner(df)
