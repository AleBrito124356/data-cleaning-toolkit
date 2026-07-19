"""Data profiling.

Produce a per-column report for a messy DataFrame: inferred type, missingness,
cardinality, a distribution summary, outlier counts, example values, and a list
of suspected issues. Render the report as Markdown or as a self-contained HTML
file (inline CSS, no external assets).

The profiler is deliberately conservative: it inspects the *raw* values of a
column (before any coercion) so that "numbers stored as text", "mixed date
formats" and stray whitespace are visible rather than hidden by pandas dtypes.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Value-level detectors
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "t", "yes", "y", "1", "si", "sí", "s", "on", "x", "✓"}
_BOOL_FALSE = {"false", "f", "no", "n", "0", "off", "", "-"}
_BOOL_VOCAB = _BOOL_TRUE | _BOOL_FALSE

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INT_RE = re.compile(r"^[+-]?\d{1,3}(?:[.,\s]?\d{3})*$|^[+-]?\d+$")
_NUM_CHARS_RE = re.compile(r"^[+-]?[\d.,\s'$€£%]+$")
_DATE_SEP_RE = re.compile(r"[/\-.]")

# Ordered list of (name, format) tried for date-format fingerprinting.
_DATE_FORMATS = [
    ("iso", "%Y-%m-%d"),
    ("iso_dt", "%Y-%m-%d %H:%M:%S"),
    ("dmy_slash", "%d/%m/%Y"),
    ("mdy_slash", "%m/%d/%Y"),
    ("dmy_dash", "%d-%m-%Y"),
    ("dmy_dot", "%d.%m.%Y"),
    ("dmy_short", "%d/%m/%y"),
    ("day_month_name", "%d %b %Y"),
    ("month_name_day", "%b %d, %Y"),
    ("day_month_name_full", "%d %B %Y"),
]

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


def _norm_str(value: Any) -> str:
    """Normalise a value to a stripped string for detection purposes."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def is_nullish(value: Any) -> bool:
    """True for pandas-null and common textual null placeholders."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return _norm_str(value).lower() in _NULL_TOKENS


def looks_boolean(text: str) -> bool:
    return text.lower() in _BOOL_VOCAB


def looks_integer(text: str) -> bool:
    return bool(_INT_RE.match(text))


def looks_numeric(text: str) -> bool:
    if not text or not _NUM_CHARS_RE.match(text):
        return False
    # Must contain at least one digit.
    return any(ch.isdigit() for ch in text)


def looks_email(text: str) -> bool:
    return bool(_EMAIL_RE.match(text))


def _try_parse_date(text: str) -> bool:
    if len(text) < 6 or not any(ch.isdigit() for ch in text):
        return False
    for _, fmt in _DATE_FORMATS:
        try:
            datetime.strptime(text, fmt)
            return True
        except ValueError:
            continue
    return False


def detected_date_formats(values: list[str]) -> list[str]:
    """Return the distinct named date formats that match the sampled values."""
    found: list[str] = []
    for text in values:
        for name, fmt in _DATE_FORMATS:
            try:
                datetime.strptime(text, fmt)
                if name not in found:
                    found.append(name)
                break
            except ValueError:
                continue
    return found


# ---------------------------------------------------------------------------
# Column profile
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    n_total: int
    n_missing: int
    n_present: int
    n_unique: int
    missing_pct: float
    unique_pct: float
    examples: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    top_values: list[tuple[str, int]] = field(default_factory=list)
    n_outliers: int = 0
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "n_present": self.n_present,
            "n_unique": self.n_unique,
            "missing_pct": round(self.missing_pct, 2),
            "unique_pct": round(self.unique_pct, 2),
            "examples": self.examples,
            "stats": self.stats,
            "top_values": self.top_values,
            "n_outliers": self.n_outliers,
            "issues": self.issues,
        }


def _infer_type(present_strings: list[str], n_unique: int, n_present: int) -> str:
    if not present_strings:
        return "empty"

    sample = present_strings[:2000]
    n = len(sample)

    frac_bool = sum(looks_boolean(v) for v in sample) / n
    if frac_bool == 1.0 and n_unique <= 3:
        return "boolean"

    frac_int = sum(looks_integer(v) for v in sample) / n
    if frac_int == 1.0:
        return "integer"

    frac_num = sum(looks_numeric(v) for v in sample) / n
    if frac_num >= 0.98:
        return "float"

    frac_date = sum(_try_parse_date(v) for v in sample) / n
    if frac_date >= 0.9:
        return "datetime"

    frac_email = sum(looks_email(v) for v in sample) / n
    if frac_email >= 0.8:
        return "email"

    unique_ratio = n_unique / n_present if n_present else 1.0
    if n_unique <= 50 and unique_ratio <= 0.5:
        return "categorical"

    return "text"


def _numeric_series(present_strings: list[str]) -> np.ndarray:
    from .clean import parse_number  # local import to avoid a cycle

    parsed: list[float] = []
    for v in present_strings:
        num = parse_number(v)
        if num is not None:
            parsed.append(num)
    return np.asarray(parsed, dtype="float64")


def _iqr_outlier_count(arr: np.ndarray) -> int:
    if arr.size < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.count_nonzero((arr < low) | (arr > high)))


def profile_column(series: pd.Series) -> ColumnProfile:
    name = str(series.name)
    n_total = len(series)

    present_mask = ~series.map(is_nullish)
    present = series[present_mask]
    present_strings = [_norm_str(v) for v in present.tolist()]
    present_strings = [s for s in present_strings if s != ""]

    n_present = len(present_strings)
    n_missing = n_total - n_present
    n_unique = len(set(present_strings))

    inferred = _infer_type(present_strings, n_unique, n_present)

    prof = ColumnProfile(
        name=name,
        inferred_type=inferred,
        n_total=n_total,
        n_missing=n_missing,
        n_present=n_present,
        n_unique=n_unique,
        missing_pct=100.0 * n_missing / n_total if n_total else 0.0,
        unique_pct=100.0 * n_unique / n_present if n_present else 0.0,
    )

    # Examples: first few distinct present values.
    seen: list[str] = []
    for s in present_strings:
        if s not in seen:
            seen.append(s)
        if len(seen) >= 5:
            break
    prof.examples = seen

    # Distribution summary.
    if inferred in {"integer", "float"}:
        arr = _numeric_series(present_strings)
        if arr.size:
            prof.stats = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": round(float(np.mean(arr)), 4),
                "median": float(np.median(arr)),
                "std": round(float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 4),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
            prof.n_outliers = _iqr_outlier_count(arr)
    else:
        counts: dict[str, int] = {}
        for s in present_strings:
            counts[s] = counts.get(s, 0) + 1
        prof.top_values = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    prof.issues = _detect_issues(prof, series, present, present_strings)
    return prof


def _detect_issues(
    prof: ColumnProfile,
    raw_series: pd.Series,
    present: pd.Series,
    present_strings: list[str],
) -> list[str]:
    issues: list[str] = []

    if prof.n_present == 0:
        issues.append("Column is entirely empty.")
        return issues

    if prof.missing_pct >= 50:
        issues.append(f"High missingness: {prof.missing_pct:.0f}% of values are missing.")
    elif prof.missing_pct >= 20:
        issues.append(f"Notable missingness: {prof.missing_pct:.0f}% missing.")

    if prof.n_unique == 1:
        issues.append("Constant column: only one distinct value.")

    if prof.n_present == prof.n_unique and prof.n_present > 1:
        issues.append("All values are unique — likely an identifier or free text.")

    # Whitespace problems in the raw values.
    ws = 0
    for v in present.tolist():
        s = "" if v is None else str(v)
        if s != s.strip() or "  " in s or " " in s or "\t" in s:
            ws += 1
    if ws:
        issues.append(f"Whitespace issues in {ws} value(s): leading/trailing or repeated spaces.")

    # Non-ASCII / non-normalised unicode.
    non_norm = sum(
        1 for s in present_strings if unicodedata.normalize("NFKC", s) != s
    )
    if non_norm:
        issues.append(f"{non_norm} value(s) are not Unicode-normalised (NFKC).")

    # Mixed date formats.
    if prof.inferred_type == "datetime":
        fmts = detected_date_formats(present_strings[:2000])
        if len(fmts) > 1:
            issues.append("Mixed date formats detected: " + ", ".join(fmts) + ".")
        if any(f in {"dmy_slash", "dmy_dash", "dmy_dot", "dmy_short"} for f in fmts):
            issues.append("Day-first dates present — parse with dayfirst=True.")

    # Numbers stored as text with separators.
    if prof.inferred_type in {"integer", "float"} and raw_series.dtype == object:
        sep = sum(1 for s in present_strings if re.search(r"[,\s'$€£]", s))
        if sep:
            issues.append(f"Numbers stored as text with separators in {sep} value(s).")

    # Mixed types: some numeric, some not, but not classified numeric.
    if prof.inferred_type in {"text", "categorical"}:
        num_like = sum(1 for s in present_strings if looks_numeric(s))
        if 0 < num_like < prof.n_present and num_like / prof.n_present > 0.1:
            issues.append(
                f"Mixed content: {num_like} value(s) look numeric among text."
            )

    # Malformed emails.
    if prof.inferred_type == "email":
        bad = sum(1 for s in present_strings if not looks_email(s))
        if bad:
            issues.append(f"{bad} value(s) do not look like valid email addresses.")

    if prof.n_outliers:
        issues.append(f"{prof.n_outliers} potential outlier(s) by the 1.5x IQR rule.")

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ProfileReport:
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    memory_bytes: int
    columns: list[ColumnProfile]
    generated_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_duplicate_rows": self.n_duplicate_rows,
            "memory_bytes": self.memory_bytes,
            "generated_at": self.generated_at,
            "columns": [c.as_dict() for c in self.columns],
        }

    # -- Markdown -----------------------------------------------------------
    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Data profile")
        lines.append("")
        lines.append(f"- Rows: **{self.n_rows}**")
        lines.append(f"- Columns: **{self.n_cols}**")
        lines.append(f"- Exact duplicate rows: **{self.n_duplicate_rows}**")
        lines.append(f"- In-memory size: **{_human_bytes(self.memory_bytes)}**")
        lines.append(f"- Generated: {self.generated_at}")
        lines.append("")
        lines.append("## Column summary")
        lines.append("")
        lines.append("| Column | Type | Missing | Unique | Outliers | Examples |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for c in self.columns:
            ex = ", ".join(_trunc(e, 24) for e in c.examples[:3])
            lines.append(
                f"| `{c.name}` | {c.inferred_type} | "
                f"{c.n_missing} ({c.missing_pct:.0f}%) | "
                f"{c.n_unique} | {c.n_outliers} | {ex} |"
            )
        lines.append("")
        lines.append("## Details")
        for c in self.columns:
            lines.append("")
            lines.append(f"### `{c.name}` — {c.inferred_type}")
            lines.append(
                f"- Present: {c.n_present} / {c.n_total} "
                f"({c.missing_pct:.1f}% missing)"
            )
            lines.append(f"- Distinct: {c.n_unique} ({c.unique_pct:.1f}% of present)")
            if c.stats:
                stat_str = ", ".join(f"{k}={_fmt_num(v)}" for k, v in c.stats.items())
                lines.append(f"- Stats: {stat_str}")
            if c.top_values:
                tv = ", ".join(f"{_trunc(k, 20)} ({n})" for k, n in c.top_values[:5])
                lines.append(f"- Top values: {tv}")
            if c.issues:
                lines.append("- Suspected issues:")
                for issue in c.issues:
                    lines.append(f"  - {issue}")
            else:
                lines.append("- Suspected issues: none detected.")
        lines.append("")
        return "\n".join(lines)

    # -- HTML ---------------------------------------------------------------
    def to_html(self) -> str:
        rows = []
        for c in self.columns:
            issue_html = (
                "<ul class='issues'>"
                + "".join(f"<li>{html.escape(i)}</li>" for i in c.issues)
                + "</ul>"
                if c.issues
                else "<span class='ok'>none</span>"
            )
            examples = ", ".join(html.escape(_trunc(e, 30)) for e in c.examples[:4])
            if c.stats:
                extra = ", ".join(
                    f"{html.escape(k)}={html.escape(_fmt_num(v))}"
                    for k, v in c.stats.items()
                )
            elif c.top_values:
                extra = ", ".join(
                    f"{html.escape(_trunc(k, 18))} ({n})"
                    for k, n in c.top_values[:4]
                )
            else:
                extra = "&mdash;"
            bar = _missing_bar(c.missing_pct)
            rows.append(
                "<tr>"
                f"<td class='col'>{html.escape(c.name)}</td>"
                f"<td><span class='type type-{html.escape(c.inferred_type)}'>"
                f"{html.escape(c.inferred_type)}</span></td>"
                f"<td>{bar}<span class='muted'>{c.n_missing} "
                f"({c.missing_pct:.0f}%)</span></td>"
                f"<td>{c.n_unique}</td>"
                f"<td>{c.n_outliers}</td>"
                f"<td class='muted'>{examples}</td>"
                f"<td class='extra'>{extra}</td>"
                f"<td>{issue_html}</td>"
                "</tr>"
            )
        table_rows = "\n".join(rows)
        return _HTML_TEMPLATE.format(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            n_dupes=self.n_duplicate_rows,
            size=_human_bytes(self.memory_bytes),
            generated=html.escape(self.generated_at),
            rows=table_rows,
        )


def profile_dataframe(df: pd.DataFrame) -> ProfileReport:
    """Profile every column of ``df`` and return a :class:`ProfileReport`."""
    columns = [profile_column(df[col]) for col in df.columns]
    return ProfileReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        n_duplicate_rows=int(df.duplicated().sum()),
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        columns=columns,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _fmt_num(v: Any) -> str:
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    return str(v)


def _trunc(s: str, width: int) -> str:
    s = str(s)
    return s if len(s) <= width else s[: width - 1] + "…"


def _missing_bar(pct: float) -> str:
    filled = min(100.0, max(0.0, pct))
    color = "#dc2626" if pct >= 50 else "#f59e0b" if pct >= 20 else "#16a34a"
    return (
        "<span class='bar'>"
        f"<span class='bar-fill' style='width:{filled:.0f}%;background:{color}'></span>"
        "</span>"
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cleankit data profile</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
      Arial, sans-serif;
    margin: 0; padding: 2rem; background: #f8fafc; color: #0f172a;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .sub {{ color: #64748b; margin: 0 0 1.5rem; font-size: .9rem; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: .75rem; margin-bottom: 1.5rem; }}
  .card {{
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: .85rem 1.1rem; min-width: 130px;
  }}
  .card .k {{ color: #64748b; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .04em; }}
  .card .v {{ font-size: 1.35rem; font-weight: 650; margin-top: .15rem; }}
  .wrap {{ overflow-x: auto; background: #fff; border: 1px solid #e2e8f0;
    border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
  th, td {{ text-align: left; padding: .55rem .75rem; vertical-align: top;
    border-bottom: 1px solid #f1f5f9; }}
  th {{ background: #f8fafc; font-weight: 600; color: #334155;
    position: sticky; top: 0; }}
  td.col {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600; white-space: nowrap; }}
  .muted {{ color: #64748b; }}
  .extra {{ color: #475569; font-size: .8rem; }}
  .type {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; background: #eef2ff; color: #4338ca; }}
  .type-integer, .type-float {{ background: #ecfeff; color: #0e7490; }}
  .type-datetime {{ background: #fef3c7; color: #b45309; }}
  .type-boolean {{ background: #f0fdf4; color: #15803d; }}
  .type-email {{ background: #fce7f3; color: #be185d; }}
  .type-categorical {{ background: #f1f5f9; color: #475569; }}
  .type-text {{ background: #f5f3ff; color: #6d28d9; }}
  .bar {{ display: inline-block; width: 60px; height: 7px; border-radius: 4px;
    background: #e2e8f0; margin-right: .4rem; vertical-align: middle;
    overflow: hidden; }}
  .bar-fill {{ display: block; height: 100%; }}
  ul.issues {{ margin: 0; padding-left: 1.1rem; color: #b91c1c; font-size: .8rem; }}
  .ok {{ color: #16a34a; font-size: .8rem; }}
  footer {{ margin-top: 1.5rem; color: #94a3b8; font-size: .78rem; }}
</style>
</head>
<body>
  <h1>Data profile</h1>
  <p class="sub">Generated by cleankit &middot; {generated}</p>
  <div class="cards">
    <div class="card"><div class="k">Rows</div><div class="v">{n_rows}</div></div>
    <div class="card"><div class="k">Columns</div><div class="v">{n_cols}</div></div>
    <div class="card"><div class="k">Duplicate rows</div><div class="v">{n_dupes}</div></div>
    <div class="card"><div class="k">Size</div><div class="v">{size}</div></div>
  </div>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th>Column</th><th>Type</th><th>Missing</th><th>Unique</th>
          <th>Outliers</th><th>Examples</th><th>Summary</th><th>Suspected issues</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </div>
  <footer>cleankit &mdash; profile then rules then pipeline then audit.</footer>
</body>
</html>
"""
