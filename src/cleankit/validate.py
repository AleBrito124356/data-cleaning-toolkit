"""Rule-based validation.

Rules live in a YAML file (``rules.yaml``). Each column can declare
``not_null``, ``unique``, ``range`` (min/max), ``regex``, ``allowed`` (a set of
permitted values), and ``dtype`` (a light type check). Top-level ``checks``
declare cross-field comparisons between two columns or a column and a literal.

Validation produces a :class:`ValidationReport` with row-level failures. The CLI
turns a failing report into a non-zero exit code so it fits into CI.

A value made only of whitespace counts as null: ``"   "`` fails ``not_null``.
Cross-field checks compare like with like: columns that hold dates are parsed
with the same ISO-safe rules as the cleaner (``2023-01-05`` is always
5 January), numeric text is compared as numbers (``"10" > "9"``), and a
literal is converted to the type of the column it is compared with.

Example ``rules.yaml``::

    columns:
      customer_id:
        not_null: true
        unique: true
      email:
        not_null: true
        regex: "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"
      age:
        range: {min: 0, max: 120}
      country:
        allowed: [Panama, Colombia, Mexico, Costa Rica]
    checks:
      - name: signup_before_last_login
        left: signup_date
        op: "<="
        right: last_login
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .dates import looks_like_date_text, parse_dates

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

KNOWN_RULES = ("not_null", "unique", "range", "regex", "allowed", "dtype")


@dataclass
class ValidationFailure:
    rule: str
    column: str
    row: int | None
    value: Any
    message: str
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "column": self.column,
            "row": self.row,
            "line": self.line,
            "value": None if _is_na(self.value) else _to_jsonable(self.value),
            "message": self.message,
        }

    def location(self) -> str:
        """``row 3 (line 5)`` style location for human-readable output."""
        if self.row is None:
            return ""
        loc = f"row {self.row}"
        if self.line is not None:
            loc += f" (line {self.line})"
        return loc


@dataclass
class ValidationReport:
    n_rows: int
    n_rules: int
    failures: list[ValidationFailure] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    def failures_by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.failures:
            out[f.rule] = out.get(f.rule, 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_rules": self.n_rules,
            "n_failures": self.n_failures,
            "failures_by_rule": self.failures_by_rule(),
            "failures": [f.as_dict() for f in self.failures],
        }

    def to_markdown(self, max_rows: int = 50) -> str:
        lines = ["# Validation report", ""]
        status = "PASS" if self.ok else "FAIL"
        lines.append(f"- Status: **{status}**")
        lines.append(f"- Rows checked: {self.n_rows}")
        lines.append(f"- Rules applied: {self.n_rules}")
        lines.append(f"- Failures: {self.n_failures}")
        lines.append("")
        if self.ok:
            lines.append("All rules passed.")
            return "\n".join(lines) + "\n"
        lines.append("## Failures by rule")
        lines.append("")
        lines.append("| Rule | Failures |")
        lines.append("| --- | --- |")
        for rule, n in sorted(
            self.failures_by_rule().items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"| {md_cell(rule)} | {n} |")
        lines.append("")
        lines.append(f"## Row-level failures (first {max_rows})")
        lines.append("")
        lines.append("| Row | Line | Column | Rule | Value | Message |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for f in self.failures[:max_rows]:
            val = "" if _is_na(f.value) else str(f.value)
            if len(val) > 30:
                val = val[:29] + "…"
            row = "" if f.row is None else str(f.row)
            line = "" if f.line is None else str(f.line)
            lines.append(
                f"| {row} | {line} | {md_code(f.column)} | {md_cell(f.rule)} | "
                f"{md_cell(val)} | {md_cell(f.message)} |"
            )
        if self.n_failures > max_rows:
            lines.append("")
            lines.append(f"…and {self.n_failures - max_rows} more.")
        return "\n".join(lines) + "\n"


def md_cell(text: Any) -> str:
    """Make text safe inside a Markdown table cell (escape pipes, fold newlines)."""
    s = str(text).replace("\\", "\\\\").replace("|", "\\|")
    return re.sub(r"[\r\n]+", " ", s)


def md_code(text: Any) -> str:
    """A code span that survives table cells, even for names containing backticks."""
    s = re.sub(r"[\r\n]+", " ", str(text)).replace("|", "\\|")
    if not s:
        return ""
    fence = "``" if "`" in s else "`"
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def load_rules(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a rules file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("rules file must be a mapping at the top level")
    data.setdefault("columns", {})
    data.setdefault("checks", [])
    if data["columns"] is None:
        data["columns"] = {}
    if data["checks"] is None:
        data["checks"] = []
    if not isinstance(data["columns"], dict):
        raise ValueError("'columns' must be a mapping of column -> rules")
    if not isinstance(data["checks"], list):
        raise ValueError("'checks' must be a list of cross-field checks")
    for col, spec in data["columns"].items():
        if spec is not None and not isinstance(spec, dict):
            raise ValueError(f"rules for column '{col}' must be a mapping of rule -> value")
    for i, check in enumerate(data["checks"], 1):
        if not isinstance(check, dict):
            raise ValueError(f"check {i} must be a mapping with left, op and right")
    return data


def validate(
    df: pd.DataFrame,
    rules: dict[str, Any],
    line_numbers: Sequence[int] | None = None,
) -> ValidationReport:
    """Apply ``rules`` to ``df`` and return a :class:`ValidationReport`.

    ``line_numbers`` optionally maps each row position to the line where that
    record starts in the source file; the CLI passes it so failures point at
    the exact CSV line even when quoted fields span several lines.
    """
    failures: list[ValidationFailure] = []
    n_rules = 0

    columns: dict[str, Any] = rules.get("columns", {}) or {}
    for col, col_rules in columns.items():
        if col not in df.columns:
            n_rules += 1
            failures.append(
                ValidationFailure(
                    rule=f"{col}.exists",
                    column=col,
                    row=None,
                    value=None,
                    message=f"Column '{col}' is declared in rules but missing.",
                )
            )
            continue
        series = df[col]
        for rule_name, spec in (col_rules or {}).items():
            n_rules += 1
            failures.extend(_check_column_rule(col, series, rule_name, spec))

    for check in rules.get("checks", []) or []:
        n_rules += 1
        failures.extend(_check_cross_field(df, check))

    if line_numbers is not None:
        positions = {label: pos for pos, label in enumerate(df.index)}
        for f in failures:
            if f.row is not None and f.row in positions:
                pos = positions[f.row]
                if pos < len(line_numbers):
                    f.line = int(line_numbers[pos])

    return ValidationReport(n_rows=len(df), n_rules=n_rules, failures=failures)


# ---------------------------------------------------------------------------
# Column-level rules
# ---------------------------------------------------------------------------


def _check_column_rule(
    col: str, series: pd.Series, rule_name: str, spec: Any
) -> list[ValidationFailure]:
    if rule_name == "not_null":
        return _rule_not_null(col, series) if spec else []
    if rule_name == "unique":
        return _rule_unique(col, series) if spec else []
    if rule_name == "range":
        return _rule_range(col, series, spec)
    if rule_name == "regex":
        return _rule_regex(col, series, spec)
    if rule_name == "allowed":
        return _rule_allowed(col, series, spec)
    if rule_name == "dtype":
        return _rule_dtype(col, series, spec)
    import difflib

    close = difflib.get_close_matches(str(rule_name), KNOWN_RULES, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return [
        ValidationFailure(
            rule=f"{col}.{rule_name}",
            column=col,
            row=None,
            value=None,
            message=f"Unknown rule '{rule_name}' for column '{col}'.{hint}",
        )
    ]


def _label(row: Any) -> Any:
    return int(row) if isinstance(row, (int, np.integer)) else row


def _rule_not_null(col: str, series: pd.Series) -> list[ValidationFailure]:
    out = []
    for row, value in series.items():
        if _is_na(value):
            blank = isinstance(value, str)
            out.append(
                ValidationFailure(
                    rule=f"{col}.not_null",
                    column=col,
                    row=_label(row),
                    value=value,
                    message="value is blank" if blank else "value is null",
                )
            )
    return out


def _rule_unique(col: str, series: pd.Series) -> list[ValidationFailure]:
    out = []
    non_null = series[~series.map(_is_na)]
    dup_mask = non_null.duplicated(keep=False)
    for row, is_dup in dup_mask.items():
        if is_dup:
            out.append(
                ValidationFailure(
                    rule=f"{col}.unique",
                    column=col,
                    row=_label(row),
                    value=series[row],
                    message="duplicate value violates uniqueness",
                )
            )
    return out


def _rule_range(col: str, series: pd.Series, spec: Any) -> list[ValidationFailure]:
    from .clean import parse_number

    spec = spec or {}
    lo = spec.get("min")
    hi = spec.get("max")
    out = []
    for row, value in series.items():
        if _is_na(value):
            continue
        num = parse_number(value)
        if num is None:
            out.append(
                ValidationFailure(
                    rule=f"{col}.range",
                    column=col,
                    row=_label(row),
                    value=value,
                    message="value is not numeric",
                )
            )
            continue
        if lo is not None and num < lo:
            out.append(
                ValidationFailure(
                    rule=f"{col}.range",
                    column=col,
                    row=_label(row),
                    value=value,
                    message=f"{_num(num)} < min {lo}",
                )
            )
        elif hi is not None and num > hi:
            out.append(
                ValidationFailure(
                    rule=f"{col}.range",
                    column=col,
                    row=_label(row),
                    value=value,
                    message=f"{_num(num)} > max {hi}",
                )
            )
    return out


def _rule_regex(col: str, series: pd.Series, spec: Any) -> list[ValidationFailure]:
    pattern = re.compile(str(spec))
    out = []
    for row, value in series.items():
        if _is_na(value):
            continue
        if not pattern.search(str(value)):
            out.append(
                ValidationFailure(
                    rule=f"{col}.regex",
                    column=col,
                    row=_label(row),
                    value=value,
                    message="does not match pattern",
                )
            )
    return out


def _rule_allowed(col: str, series: pd.Series, spec: Any) -> list[ValidationFailure]:
    allowed = {str(x) for x in (spec or [])}
    out = []
    for row, value in series.items():
        if _is_na(value):
            continue
        if str(value) not in allowed:
            out.append(
                ValidationFailure(
                    rule=f"{col}.allowed",
                    column=col,
                    row=_label(row),
                    value=value,
                    message="value not in allowed set",
                )
            )
    return out


def _rule_dtype(col: str, series: pd.Series, spec: Any) -> list[ValidationFailure]:
    from .clean import parse_boolean, parse_number

    target = str(spec).lower()
    out = []
    dates = None
    if target in ("datetime", "date") and not pd.api.types.is_datetime64_any_dtype(series.dtype):
        dates = parse_dates(series).notna().tolist()
    for pos, (row, value) in enumerate(series.items()):
        if _is_na(value):
            continue
        ok = True
        if target in ("int", "integer"):
            num = parse_number(value)
            ok = num is not None and float(num).is_integer()
        elif target in ("float", "number"):
            ok = parse_number(value) is not None
        elif target in ("bool", "boolean"):
            ok = parse_boolean(value) is not None
        elif target in ("datetime", "date"):
            ok = True if dates is None else dates[pos]
        if not ok:
            out.append(
                ValidationFailure(
                    rule=f"{col}.dtype",
                    column=col,
                    row=_label(row),
                    value=value,
                    message=f"not a valid {target}",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Cross-field checks
# ---------------------------------------------------------------------------


def _check_cross_field(df: pd.DataFrame, check: dict[str, Any]) -> list[ValidationFailure]:
    name = str(check.get("name", "cross_field"))
    op = str(check.get("op", "=="))
    if op not in _OPS:
        return [
            ValidationFailure(
                rule=name,
                column="",
                row=None,
                value=None,
                message=f"unknown operator '{op}'",
            )
        ]
    left = check.get("left")
    right = check.get("right")
    if left is None or right is None:
        return [
            ValidationFailure(
                rule=name,
                column=str(left or ""),
                row=None,
                value=None,
                message="cross-field check needs 'left' and 'right'",
            )
        ]

    left_series, left_kind = _resolve_operand(df, left)
    right_series, right_kind = _resolve_operand(df, right)
    if left_series is None or right_series is None:
        return [
            ValidationFailure(
                rule=name,
                column=str(left),
                row=None,
                value=None,
                message="a referenced column does not exist",
            )
        ]
    left_series, right_series = _align_kinds(left_series, left_kind, right_series, right_kind)

    fn = _OPS[op]
    out = []
    left_values = left_series.tolist()
    right_values = right_series.tolist()
    raw_left = df[left].tolist() if isinstance(left, str) and left in df.columns else left_values
    for pos, row in enumerate(df.index):
        lv = left_values[pos]
        rv = right_values[pos]
        if _is_na(lv) or _is_na(rv):
            continue
        try:
            passed = bool(fn(lv, rv))
        except TypeError:
            passed = bool(fn(str(lv), str(rv)))
        if not passed:
            out.append(
                ValidationFailure(
                    rule=name,
                    column=str(left),
                    row=_label(row),
                    value=raw_left[pos],
                    message=f"expected {left} {op} {right}, got {_show(lv)} vs {_show(rv)}",
                )
            )
    return out


def _num(x: float) -> str:
    """9999999.0 -> '9999999', 2.5 -> '2.5' (never scientific notation)."""
    return str(int(x)) if float(x).is_integer() else f"{x:.10g}"


def _show(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value) if isinstance(value, str) else str(value)


def _column_kind(series: pd.Series) -> str:
    """'datetime', 'number' or 'text' for a column (text columns are sniffed)."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "text"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "number"
    present = [v for v in series.tolist() if not _is_na(v)]
    if not present:
        return "text"
    sample = present[:200]
    if sum(looks_like_date_text(str(v)) for v in sample) / len(sample) >= 0.8:
        return "datetime"
    from .clean import parse_number

    if all(parse_number(v) is not None for v in sample):
        return "number"
    return "text"


def _to_numbers(series: pd.Series) -> pd.Series:
    from .clean import parse_number

    return series.map(lambda v: np.nan if _is_na(v) else parse_number(v)).astype("float64")


def _resolve_operand(df: pd.DataFrame, operand: Any) -> tuple[pd.Series | None, str]:
    """A column name resolves to that column (typed); anything else is a literal."""
    if isinstance(operand, str) and operand in df.columns:
        series = df[operand]
        kind = _column_kind(series)
        if kind == "datetime":
            return parse_dates(series), kind
        if kind == "number" and not pd.api.types.is_numeric_dtype(series.dtype):
            return _to_numbers(series), kind
        return series, kind
    # Literal scalar broadcast to every row.
    return pd.Series([operand] * len(df), index=df.index, dtype=object), "literal"


def _align_kinds(
    left: pd.Series, left_kind: str, right: pd.Series, right_kind: str
) -> tuple[pd.Series, pd.Series]:
    """Convert a literal operand to the type of the column it is compared with."""
    if right_kind == "literal" and left_kind in ("datetime", "number"):
        right = _convert_literal(right, left_kind)
    elif left_kind == "literal" and right_kind in ("datetime", "number"):
        left = _convert_literal(left, right_kind)
    return left, right


def _convert_literal(series: pd.Series, kind: str) -> pd.Series:
    if kind == "datetime":
        values = series.map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
        return parse_dates(values, dayfirst=False)
    return _to_numbers(series)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_na(value: Any) -> bool:
    """Null, NaN/NaT/NA, or a string made only of whitespace."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
    except (ValueError, TypeError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
