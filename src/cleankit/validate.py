"""Rule-based validation.

Rules live in a YAML file (``rules.yaml``). Each column can declare
``not_null``, ``unique``, ``range`` (min/max), ``regex``, ``allowed`` (a set of
permitted values), and ``dtype`` (a light type check). Top-level ``checks``
declare cross-field comparisons between two columns or a column and a literal.

Validation produces a :class:`ValidationReport` with row-level failures. The CLI
turns a failing report into a non-zero exit code so it fits into CI.

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
from typing import Any

import pandas as pd
import yaml

_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class ValidationFailure:
    rule: str
    column: str
    row: int | None
    value: Any
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "column": self.column,
            "row": self.row,
            "value": None if _is_na(self.value) else _to_jsonable(self.value),
            "message": self.message,
        }


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
            lines.append(f"| {rule} | {n} |")
        lines.append("")
        lines.append(f"## Row-level failures (first {max_rows})")
        lines.append("")
        lines.append("| Row | Column | Rule | Value | Message |")
        lines.append("| --- | --- | --- | --- | --- |")
        for f in self.failures[:max_rows]:
            val = "" if _is_na(f.value) else str(f.value)
            if len(val) > 30:
                val = val[:29] + "…"
            row = "" if f.row is None else str(f.row)
            lines.append(
                f"| {row} | `{f.column}` | {f.rule} | {val} | {f.message} |"
            )
        if self.n_failures > max_rows:
            lines.append("")
            lines.append(f"…and {self.n_failures - max_rows} more.")
        return "\n".join(lines) + "\n"


def load_rules(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate a rules file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("rules file must be a mapping at the top level")
    data.setdefault("columns", {})
    data.setdefault("checks", [])
    if not isinstance(data["columns"], dict):
        raise ValueError("'columns' must be a mapping of column -> rules")
    if not isinstance(data["checks"], list):
        raise ValueError("'checks' must be a list of cross-field checks")
    return data


def validate(df: pd.DataFrame, rules: dict[str, Any]) -> ValidationReport:
    """Apply ``rules`` to ``df`` and return a :class:`ValidationReport`."""
    failures: list[ValidationFailure] = []
    n_rules = 0

    columns: dict[str, Any] = rules.get("columns", {})
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

    for check in rules.get("checks", []):
        n_rules += 1
        failures.extend(_check_cross_field(df, check))

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
    return [
        ValidationFailure(
            rule=f"{col}.{rule_name}",
            column=col,
            row=None,
            value=None,
            message=f"Unknown rule '{rule_name}' for column '{col}'.",
        )
    ]


def _rule_not_null(col: str, series: pd.Series) -> list[ValidationFailure]:
    out = []
    for row, value in series.items():
        if _is_na(value):
            out.append(
                ValidationFailure(
                    rule=f"{col}.not_null",
                    column=col,
                    row=int(row),
                    value=value,
                    message="value is null",
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
                    row=int(row),
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
        num = parse_number(value) if not isinstance(value, (int, float)) else float(value)
        if num is None:
            out.append(
                ValidationFailure(
                    rule=f"{col}.range",
                    column=col,
                    row=int(row),
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
                    row=int(row),
                    value=value,
                    message=f"{num:g} < min {lo}",
                )
            )
        elif hi is not None and num > hi:
            out.append(
                ValidationFailure(
                    rule=f"{col}.range",
                    column=col,
                    row=int(row),
                    value=value,
                    message=f"{num:g} > max {hi}",
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
                    row=int(row),
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
                    row=int(row),
                    value=value,
                    message="value not in allowed set",
                )
            )
    return out


def _rule_dtype(col: str, series: pd.Series, spec: Any) -> list[ValidationFailure]:
    from .clean import parse_boolean, parse_number

    target = str(spec).lower()
    out = []
    for row, value in series.items():
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
            ok = pd.notna(pd.to_datetime(value, errors="coerce"))
        if not ok:
            out.append(
                ValidationFailure(
                    rule=f"{col}.dtype",
                    column=col,
                    row=int(row),
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

    left_series = _resolve_operand(df, left)
    right_series = _resolve_operand(df, right)
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

    fn = _OPS[op]
    out = []
    for row in df.index:
        lv = left_series[row]
        rv = right_series[row]
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
                    row=int(row),
                    value=lv,
                    message=f"expected {left} {op} {right}, got {lv!r} vs {rv!r}",
                )
            )
    return out


def _resolve_operand(df: pd.DataFrame, operand: Any):
    """A column name resolves to that column; otherwise treat as a literal.

    Date-like literals and columns are coerced to datetime so comparisons work.
    """
    if isinstance(operand, str) and operand in df.columns:
        series = df[operand]
        if _looks_datetime(series):
            return pd.to_datetime(series, errors="coerce", dayfirst=True)
        return series
    # Literal scalar broadcast to every row.
    return pd.Series([operand] * len(df), index=df.index)


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Numeric columns are never date strings; skip to avoid noisy parsing.
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    # Only treat as dates if the values actually contain date separators;
    # this keeps free text from triggering dateutil's slow per-element path.
    if not sample.str.contains(r"[/\-.:]|\d{4}").any():
        return False
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= 0.8


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (ValueError, TypeError):
        return False
    if isinstance(result, bool):
        return result
    return False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
