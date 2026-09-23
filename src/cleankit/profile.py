"""Data profiling.

Produce a per-column report for a messy DataFrame: inferred type, semantic
role (identifier, person name, address, country), missingness, cardinality, a
distribution summary, outlier counts, example values, spelling-variant
clusters, and a list of suspected issues. Render the report as Markdown, JSON,
or a self-contained HTML file (inline CSS, no external assets, light and dark).

The profiler is deliberately conservative: it inspects the *raw* values of a
column (before any coercion) so that "numbers stored as text", "mixed date
formats", "nine spellings of Panama" and stray whitespace are visible rather
than hidden by pandas dtypes.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .dates import dayfirst_evidence, detect_dayfirst, looks_like_date_text, parse_dates
from .text import canonical_key, same_variant, strip_noise

# ---------------------------------------------------------------------------
# Value-level detectors
# ---------------------------------------------------------------------------

_BOOL_TRUE = {"true", "t", "yes", "y", "1", "si", "sí", "s", "on", "x", "✓"}
_BOOL_FALSE = {"false", "f", "no", "n", "0", "off", "", "-"}
_BOOL_VOCAB = _BOOL_TRUE | _BOOL_FALSE

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INT_RE = re.compile(r"^[+-]?\d{1,3}(?:[.,\s]?\d{3})*$|^[+-]?\d+$")
_NUM_CHARS_RE = re.compile(r"^[+-]?[\d.,\s'$€£%]+$")
_PHONE_RE = re.compile(r"^\+?[\d\s().\-/]{6,24}$")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:[ ,.]\d{3})+$")
_CURRENCY_CHARS = re.compile(r"[$€£¥]|\b(?:usd|eur|mxn|cop|pab)\b", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"[,\s']|\d\.\d{3}(?:\D|$)")

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

# Header hints (matched against the snake_cased column name).
_PHONE_HINT = re.compile(r"(^|_)(phone|telefono|tel|celular|cel|movil|mobile|whatsapp|fax)(_|$)")
_ID_HINT = re.compile(r"(^|_)(id|uuid|guid|key|code|codigo|sku|nro|num|number)$|^id(_|$)")
_ADDRESS_HINT = re.compile(r"(^|_)(address|addr|direccion|street|calle|domicilio)(_|$)")
_EMAIL_HINT = re.compile(r"(^|_)(email|e_mail|mail|correo)(_|$)")
_NAME_HEADERS = {
    "name", "nombre", "nombres", "full_name", "fullname", "nombre_completo",
    "first_name", "last_name", "given_name", "family_name", "middle_name",
    "surname", "apellido", "apellidos",
}
_NAME_OWNER = re.compile(
    r"^(customer|client|contact|person|cliente|contacto|employee|empleado|user_full)_(name|nombre)$"
)
_COUNTRY_HEADERS = {"country", "pais", "country_name", "nation", "country_code"}


def _norm_str(value: Any) -> str:
    """Normalise a value to a stripped string for detection purposes."""
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def is_nullish(value: Any) -> bool:
    """True for pandas-null and common textual null placeholders."""
    if value is None or value is pd.NA or value is pd.NaT:
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


def looks_phone(text: str) -> bool:
    """Digits with phone punctuation (+, spaces, dashes, parentheses), 7-15 digits."""
    if not _PHONE_RE.match(text):
        return False
    digits = sum(ch.isdigit() for ch in text)
    return 7 <= digits <= 15


def _phone_punctuated(text: str) -> bool:
    if _THOUSANDS_RE.match(text):
        return False
    return text.startswith("+") or "(" in text or bool(re.search(r"\d[\s\-]\d", text))


# -- date fingerprints -----------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "set": 9, "septiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}
_TIME = r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?"
# (label, regex, component order). "dm" labels are resolved per column.
_DATE_FINGERPRINTS = [
    ("YYYY-MM-DD", re.compile(rf"^(\d{{4}})-(\d{{1,2}})-(\d{{1,2}}){_TIME}$"), "ymd"),
    ("YYYY/MM/DD", re.compile(rf"^(\d{{4}})/(\d{{1,2}})/(\d{{1,2}}){_TIME}$"), "ymd"),
    ("YYYY.MM.DD", re.compile(rf"^(\d{{4}})\.(\d{{1,2}})\.(\d{{1,2}}){_TIME}$"), "ymd"),
    ("{a}/{b}/YYYY", re.compile(rf"^(\d{{1,2}})/(\d{{1,2}})/(\d{{4}}){_TIME}$"), "dm"),
    ("{a}-{b}-YYYY", re.compile(rf"^(\d{{1,2}})-(\d{{1,2}})-(\d{{4}}){_TIME}$"), "dm"),
    ("{a}.{b}.YYYY", re.compile(rf"^(\d{{1,2}})\.(\d{{1,2}})\.(\d{{4}}){_TIME}$"), "dm"),
    ("{a}/{b}/YY", re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$"), "dm"),
    ("Mon DD YYYY", re.compile(r"^([A-Za-z]{3,10})\.? (\d{1,2}),? (\d{4})$"), "mdy_name"),
    ("DD Mon YYYY", re.compile(r"^(\d{1,2}) (?:de )?([A-Za-z]{3,10})\.?,? (?:de )?(\d{4})$", re.IGNORECASE), "dmy_name"),
    ("DD-Mon-YYYY", re.compile(r"^(\d{1,2})-([A-Za-z]{3,10})-(\d{4})$"), "dmy_name"),
]


def _valid_ymd(y: int, m: int, d: int) -> bool:
    try:
        datetime(y, m, d)
        return True
    except ValueError:
        return False


def date_fingerprint(text: str, dayfirst: bool = True) -> str | None:
    """Name the date layout of ``text`` (``"DD/MM/YYYY"``), or None if not a date."""
    for label, rx, order in _DATE_FINGERPRINTS:
        m = rx.match(text)
        if not m:
            continue
        g = m.groups()
        if order == "ymd":
            ok = _valid_ymd(int(g[0]), int(g[1]), int(g[2]))
        elif order == "dm":
            a, b, y = int(g[0]), int(g[1]), int(g[2])
            y = y + 2000 if y < 100 else y
            first_is_day = dayfirst
            if a > 12 >= b:
                first_is_day = True
            elif b > 12 >= a:
                first_is_day = False
            d, mth = (a, b) if first_is_day else (b, a)
            ok = _valid_ymd(y, mth, d)
            label = label.format(a="DD" if first_is_day else "MM", b="MM" if first_is_day else "DD")
        else:
            name_idx, day_idx = (0, 1) if order == "mdy_name" else (1, 0)
            month = _MONTHS.get(g[name_idx].lower().rstrip("."))
            ok = month is not None and _valid_ymd(int(g[2]), month, int(g[day_idx]))
        if ok:
            return label
    return None


def _try_parse_date(text: str) -> bool:
    return date_fingerprint(text) is not None


def detected_date_formats(values: list[str]) -> list[str]:
    """Return the distinct date layouts found in ``values``, most common first."""
    dayfirst = detect_dayfirst(values)
    counts = Counter(f for f in (date_fingerprint(v, dayfirst) for v in values) if f)
    return [label for label, _ in counts.most_common()]


# ---------------------------------------------------------------------------
# Variant clustering
# ---------------------------------------------------------------------------


def _is_nicely_cased(value: str) -> bool:
    letters = [c for c in value if c.isalpha()]
    if not letters:
        return False
    if len(letters) <= 3:
        return True  # USA, UK
    return not all(c.isupper() for c in letters) and not all(c.islower() for c in letters)


def cluster_values(counts: dict[str, int], max_distinct: int = 300) -> list[dict[str, Any]]:
    """Group spellings of the same value.

    ``Panamá``, ``panama``, ``PANAMA`` and ``Rep. de Panamá`` share an
    accent-free key once decorative words are dropped; ``Colombia`` and
    ``Columbia`` are one typo apart. Each cluster reports its canonical
    spelling (the most frequent, preferring nicely cased text) and its
    variants with counts. Returns ``[]`` when there are too many distinct
    values for clustering to be meaningful.
    """
    if len(counts) > max_distinct:
        return []
    by_key: dict[str, list[tuple[str, int]]] = {}
    first_seen: dict[str, int] = {}
    for i, (value, n) in enumerate(counts.items()):
        key = strip_noise(canonical_key(value))
        by_key.setdefault(key, []).append((value, n))
        first_seen.setdefault(value, i)

    keys = list(by_key)
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if same_variant(keys[i], keys[j]):
                parent[find(j)] = find(i)

    groups: dict[int, list[tuple[str, int]]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(find(i), []).extend(by_key[key])

    clusters = []
    for members in groups.values():
        members.sort(key=lambda vn: (-vn[1], first_seen[vn[0]]))
        canonical = max(
            members,
            key=lambda vn: (vn[1], _is_nicely_cased(vn[0]), -first_seen[vn[0]]),
        )[0]
        clusters.append(
            {
                "canonical": canonical,
                "count": int(sum(n for _, n in members)),
                "variants": [[v, int(n)] for v, n in members],
            }
        )
    clusters.sort(key=lambda c: (-c["count"], c["canonical"]))
    return clusters


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
    semantic: str | None = None
    clusters: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "semantic": self.semantic,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "n_present": self.n_present,
            "n_unique": self.n_unique,
            "missing_pct": round(self.missing_pct, 2),
            "unique_pct": round(self.unique_pct, 2),
            "examples": self.examples,
            "stats": self.stats,
            "top_values": [list(tv) for tv in self.top_values],
            "n_outliers": self.n_outliers,
            "issues": self.issues,
            "clusters": self.clusters,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnProfile":
        return cls(
            name=data["name"],
            inferred_type=data["inferred_type"],
            n_total=data["n_total"],
            n_missing=data["n_missing"],
            n_present=data["n_present"],
            n_unique=data["n_unique"],
            missing_pct=float(data["missing_pct"]),
            unique_pct=float(data["unique_pct"]),
            examples=list(data.get("examples", [])),
            stats=dict(data.get("stats", {})),
            top_values=[tuple(tv) for tv in data.get("top_values", [])],
            n_outliers=int(data.get("n_outliers", 0)),
            issues=list(data.get("issues", [])),
            semantic=data.get("semantic"),
            clusters=list(data.get("clusters", [])),
            details=dict(data.get("details", {})),
        )


def _infer_type(
    present_strings: list[str], n_unique: int, n_present: int, name: str = ""
) -> str:
    from .clean import to_snake_case

    if not present_strings:
        return "empty"

    sample = present_strings[:2000]
    n = len(sample)
    snake = to_snake_case(name) if name else ""

    if all(looks_boolean(v) for v in sample):
        lowered = {v.lower() for v in sample}
        both = bool(lowered & _BOOL_TRUE) and bool(lowered & _BOOL_FALSE)
        if n_unique <= 3 or both:
            return "boolean"

    phone_share = sum(looks_phone(v) for v in sample) / n
    if snake and _PHONE_HINT.search(snake) and phone_share >= 0.7:
        return "phone"

    frac_int = sum(looks_integer(v) for v in sample) / n
    if frac_int == 1.0:
        return "integer"

    frac_num = sum(looks_numeric(v) for v in sample) / n
    if frac_num >= 0.98:
        return "float"

    frac_date = sum(_try_parse_date(v) or looks_like_date_text(v) for v in sample) / n
    if frac_date >= 0.8:
        return "datetime"

    frac_email = sum(looks_email(v) for v in sample) / n
    if frac_email >= 0.8 or (snake and _EMAIL_HINT.search(snake) and frac_email >= 0.5):
        return "email"

    if phone_share >= 0.8 and sum(_phone_punctuated(v) for v in sample) / n >= 0.3:
        return "phone"

    if n_unique <= 300:
        counts = Counter(sample)
        n_groups = len(cluster_values(dict(counts))) if n_unique > 1 else 1
        if n_groups <= 50 and n_groups / n_present <= 0.5:
            return "categorical"

    return "text"


def _numeric_series(present_strings: list[str]) -> np.ndarray:
    from .clean import parse_number_column  # local import to avoid a cycle

    nums, _, _ = parse_number_column(present_strings)
    return np.asarray([n for n in nums if n is not None], dtype="float64")


def _iqr_outlier_count(arr: np.ndarray) -> int:
    if arr.size < 4:
        return 0
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(np.count_nonzero((arr < low) | (arr > high)))


def _semantic_role(name: str, inferred: str, sample: list[str], unique_pct: float) -> str | None:
    from .clean import to_snake_case
    from .contacts import region_for_country

    snake = to_snake_case(name)
    if _ID_HINT.search(snake) and inferred in {"integer", "text", "categorical"} and unique_pct >= 50:
        return "identifier"
    if inferred not in {"text", "categorical"} or not sample:
        return None
    if snake in _NAME_HEADERS or _NAME_OWNER.match(snake):
        wordy = sum(bool(re.fullmatch(r"[^\d@]+", v)) for v in sample) / len(sample)
        if wordy >= 0.8:
            return "person_name"
    if _ADDRESS_HINT.search(snake):
        return "address"
    if snake in _COUNTRY_HEADERS:
        return "country"
    distinct = list(dict.fromkeys(sample))[:60]
    if inferred == "categorical" and distinct:
        known = sum(region_for_country(v) is not None for v in distinct) / len(distinct)
        if known >= 0.8:
            return "country"
    return None


def _present_strings(raw_values: list[Any]) -> list[str]:
    """Stripped text of every value that is not null or a null placeholder."""
    out: list[str] = []
    tokens = _NULL_TOKENS
    na, nat = pd.NA, pd.NaT
    for v in raw_values:
        if v is None or v is na or v is nat or (isinstance(v, float) and v != v):
            continue
        s = (v if isinstance(v, str) else str(v)).strip()
        if s.lower() in tokens:
            continue
        out.append(s)
    return out


def profile_column(series: pd.Series) -> ColumnProfile:
    name = str(series.name)
    n_total = len(series)

    raw_values = series.tolist()
    present_strings = _present_strings(raw_values)

    n_present = len(present_strings)
    n_missing = n_total - n_present
    n_unique = len(set(present_strings))

    inferred = _infer_type(present_strings, n_unique, n_present, name)

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
    prof.semantic = _semantic_role(name, inferred, present_strings[:2000], prof.unique_pct)

    # Examples: first few distinct present values.
    prof.examples = list(dict.fromkeys(present_strings))[:5]

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
        counts = Counter(present_strings)
        prof.top_values = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        if inferred == "categorical" or prof.semantic == "country":
            prof.clusters = cluster_values(dict(counts))

    prof.issues = _detect_issues(prof, raw_values, present_strings)
    return prof


def _examples(values: list[str], k: int = 2) -> str:
    uniq = list(dict.fromkeys(values))[:k]
    return ", ".join(f"'{v}'" for v in uniq)


def _detect_issues(
    prof: ColumnProfile,
    raw_values: list[Any],
    present_strings: list[str],
) -> list[str]:
    from .clean import infer_number_format, parse_number
    from .contacts import clean_address, standardize_email, standardize_name

    issues: list[str] = []

    if prof.n_present == 0:
        issues.append("Column is entirely empty.")
        return issues

    if prof.missing_pct >= 50:
        issues.append(f"High missingness: {prof.missing_pct:.0f}% of values are missing.")
    elif prof.missing_pct >= 20:
        issues.append(f"Notable missingness: {prof.missing_pct:.0f}% missing.")

    raw_counts = Counter(v for v in raw_values if isinstance(v, str))
    present_counts = Counter(present_strings)
    blank = sum(n for v, n in raw_counts.items() if v != "" and v.strip() == "")
    if blank:
        issues.append(
            f"{blank} value(s) are whitespace-only (counted as missing; "
            "normalize_whitespace turns them into nulls)."
        )

    if prof.n_unique == 1:
        issues.append("Constant column: only one distinct value.")

    if prof.n_present == prof.n_unique and prof.n_present > 1 and prof.semantic != "identifier":
        issues.append("All values are unique — likely an identifier or free text.")

    # Whitespace problems in the raw values.
    ws = 0
    for v, n in raw_counts.items():
        if v.strip() == "":
            continue
        if v != v.strip() or "  " in v or "\u00a0" in v or "\t" in v:
            ws += n
    if ws:
        issues.append(f"Whitespace issues in {ws} value(s): leading/trailing or repeated spaces.")

    # Non-ASCII / non-normalised unicode.
    non_norm = sum(n for s, n in present_counts.items() if unicodedata.normalize("NFKC", s) != s)
    if non_norm:
        issues.append(f"{non_norm} value(s) are not Unicode-normalised (NFKC).")

    sample = present_strings[:2000]
    t = prof.inferred_type

    if t == "datetime":
        dayfirst = detect_dayfirst(sample)
        prints = [date_fingerprint(v, dayfirst) for v in sample]
        counts = Counter(p for p in prints if p)
        prof.details["date_formats"] = dict(counts.most_common())
        prof.details["dayfirst"] = dayfirst
        if len(counts) > 1:
            issues.append(
                "Mixed date formats detected: "
                + ", ".join(f"{label} ({n})" for label, n in counts.most_common())
                + "."
            )
        day_first, month_first, ambiguous = dayfirst_evidence(sample)
        if day_first and month_first:
            issues.append(
                f"Conflicting day/month order: {day_first} value(s) can only be "
                f"day-first and {month_first} only month-first."
            )
        elif day_first:
            ex = next(v for v in sample if date_fingerprint(v, True) and dayfirst_evidence([v])[0])
            msg = f"Day-first dates present (e.g. {ex}) — parse with dayfirst=True."
            if ambiguous:
                amb = next(v for v in sample if dayfirst_evidence([v])[2])
                msg += f" {ambiguous} ambiguous value(s) such as {amb} will be read day-first."
            issues.append(msg)
        unrecognised = [v for v, p in zip(sample, prints) if p is None]
        if unrecognised:
            parsed = parse_dates(pd.Series(unrecognised, dtype=object), dayfirst=dayfirst)
            bad = [v for v, ts in zip(unrecognised, parsed) if pd.isna(ts)]
            if bad:
                issues.append(f"{len(bad)} value(s) are not recognisable dates (e.g. {_examples(bad)}).")

    if t in {"integer", "float"}:
        sep = [s for s in present_counts if _SEPARATOR_RE.search(s)]
        if sep:
            n_sep = sum(present_counts[s] for s in sep)
            issues.append(
                f"Numbers stored as text with separators in {n_sep} value(s) (e.g. {_examples(sep)})."
            )
        cur = [s for s in present_counts if _CURRENCY_CHARS.search(s)]
        if cur:
            n_cur = sum(present_counts[s] for s in cur)
            issues.append(f"Currency symbols in {n_cur} value(s) (e.g. {_examples(cur)}).")
        fmt = infer_number_format(present_strings)
        n_comma = fmt.decimal[","] + fmt.decimal3[","]
        n_dot = fmt.decimal["."] + fmt.decimal3["."]
        if n_comma and n_dot:
            issues.append(
                f"Mixed decimal marks: {n_comma} value(s) use ',' "
                f"(e.g. '{fmt.examples.get('decimal,') or fmt.examples.get('decimal3,')}') and "
                f"{n_dot} use '.' (e.g. '{fmt.examples.get('decimal.') or fmt.examples.get('decimal3.')}')."
            )
        for sep_char in (".", ","):
            amb = [v for v in fmt.ambiguous if sep_char in v]
            if not amb:
                continue
            role, reason = fmt.resolve(sep_char)
            hint = sep_char if role == "decimal" else ("," if sep_char == "." else ".")
            read = [parse_number(v, decimal=hint) for v in amb]
            shown = ", ".join(
                f"'{v}' -> {_fmt_num(r) if r is not None else '?'}"
                for v, r in list(dict(zip(amb, read)).items())[:3]
            )
            issues.append(
                f"Ambiguous '{sep_char}' in {len(amb)} value(s) ({shown}): read as "
                f"{role} because {reason}."
            )
        prof.details["decimal_marks"] = {"comma": n_comma, "dot": n_dot}

    if t == "boolean":
        spellings = Counter(present_strings)
        if len(spellings) > 2:
            trues = [s for s, _ in spellings.most_common() if s.lower() in _BOOL_TRUE]
            falses = [s for s, _ in spellings.most_common() if s.lower() in _BOOL_FALSE]
            issues.append(
                "Inconsistent boolean spellings: "
                + ", ".join(trues)
                + " (true) / "
                + ", ".join(falses)
                + " (false)."
            )
            prof.details["boolean_spellings"] = dict(spellings.most_common())

    # Mixed types: some numeric, some not, but not classified numeric.
    if t in {"text", "categorical"}:
        num_like = sum(n for s, n in present_counts.items() if looks_numeric(s))
        if 0 < num_like < prof.n_present and num_like / prof.n_present > 0.1:
            issues.append(f"Mixed content: {num_like} value(s) look numeric among text.")

    # Spelling variants of the same category.
    multi = [c for c in prof.clusters if len(c["variants"]) > 1]
    for c in multi[:3]:
        variants = ", ".join(f"{v} ({n})" for v, n in c["variants"][:6])
        issues.append(f"{len(c['variants'])} spellings of '{c['canonical']}': {variants}.")
    if len(multi) > 3:
        issues.append(f"…and {len(multi) - 3} more value(s) with several spellings.")

    if prof.semantic == "person_name":
        off = [s for s in sample if standardize_name(s) != re.sub(r"\s+", " ", s)]
        if off:
            issues.append(
                f"Inconsistent casing in {len(off)} name(s) (e.g. {_examples(off)}) — "
                "standardize_names fixes this."
            )

    if prof.semantic == "address":
        off = [s for s in sample if clean_address(s) != re.sub(r"\s+", " ", s)]
        if off:
            issues.append(
                f"{len(off)} address(es) use abbreviations or untidy casing "
                f"(e.g. {_examples(off)}) — clean_addresses expands Ave, Edif, Apto, Cra."
            )

    if t == "email":
        results = [(s, standardize_email(s)) for s in sample]
        bad = [s for s, r in results if not r.valid]
        typos = [s for s, r in results if r.note == "fixed domain typo"]
        upper = [s for s in sample if s != s.lower()]
        if bad:
            issues.append(f"{len(bad)} value(s) do not look like valid email addresses (e.g. {_examples(bad)}).")
        if typos:
            issues.append(f"{len(typos)} value(s) have a mistyped provider domain (e.g. {_examples(typos)}).")
        if upper:
            issues.append(f"{len(upper)} value(s) contain upper-case letters.")

    if t == "phone":
        shapes = Counter(re.sub(r"\d", "9", s) for s in sample)
        if len(shapes) > 1:
            issues.append(
                f"Phone numbers written in {len(shapes)} different layouts "
                f"(e.g. {_examples(sample, 3)})."
            )
        no_cc = [s for s in sample if not s.startswith(("+", "00"))]
        if no_cc:
            issues.append(
                f"{len(no_cc)} number(s) have no '+' country code; standardize_phones "
                "reads them in the row's or the default region."
            )

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

    @property
    def n_issues(self) -> int:
        return sum(len(c.issues) for c in self.columns)

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_duplicate_rows": self.n_duplicate_rows,
            "memory_bytes": self.memory_bytes,
            "generated_at": self.generated_at,
            "columns": [c.as_dict() for c in self.columns],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileReport":
        return cls(
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            n_duplicate_rows=data["n_duplicate_rows"],
            memory_bytes=data["memory_bytes"],
            columns=[ColumnProfile.from_dict(c) for c in data["columns"]],
            generated_at=data.get("generated_at", ""),
        )

    # -- Markdown -----------------------------------------------------------
    def to_markdown(self) -> str:
        from .validate import md_cell, md_code

        lines: list[str] = []
        lines.append("# Data profile")
        lines.append("")
        lines.append(f"- Rows: **{self.n_rows}**")
        lines.append(f"- Columns: **{self.n_cols}**")
        lines.append(f"- Exact duplicate rows: **{self.n_duplicate_rows}**")
        lines.append(f"- In-memory size: **{_human_bytes(self.memory_bytes)}**")
        lines.append(f"- Suspected issues: **{self.n_issues}**")
        lines.append(f"- Generated: {self.generated_at}")
        lines.append("")
        lines.append("## Column summary")
        lines.append("")
        lines.append("| Column | Type | Role | Missing | Unique | Outliers | Issues | Examples |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in self.columns:
            ex = ", ".join(_trunc(e, 24) for e in c.examples[:3])
            lines.append(
                f"| {md_code(c.name)} | {c.inferred_type} | {c.semantic or ''} | "
                f"{c.n_missing} ({c.missing_pct:.0f}%) | "
                f"{c.n_unique} | {c.n_outliers} | {len(c.issues)} | {md_cell(ex)} |"
            )
        lines.append("")
        lines.append("## Details")
        for c in self.columns:
            lines.append("")
            role = f" ({c.semantic})" if c.semantic else ""
            lines.append(f"### {md_code(c.name)} — {c.inferred_type}{role}")
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
        findings = []
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
            role = (
                f" <span class='role'>{html.escape(c.semantic)}</span>" if c.semantic else ""
            )
            bar = _missing_bar(c.missing_pct)
            cells = [
                ("col", "Column", html.escape(c.name)),
                ("", "Type", f"<span class='type type-{html.escape(c.inferred_type)}'>"
                 f"{html.escape(c.inferred_type)}</span>{role}"),
                ("", "Missing", f"{bar}<span class='muted'>{c.n_missing} ({c.missing_pct:.0f}%)</span>"),
                ("", "Unique", str(c.n_unique)),
                ("", "Outliers", str(c.n_outliers)),
                ("muted", "Examples", examples or "&mdash;"),
                ("extra", "Summary", extra),
                ("", "Issues", issue_html),
            ]
            rows.append(
                "<tr>"
                + "".join(
                    f"<td{f' class={chr(39)}{cls}{chr(39)}' if cls else ''} data-label='{label}'>"
                    f"<span class='cell'>{content}</span></td>"
                    for cls, label, content in cells
                )
                + "</tr>"
            )
            for issue in c.issues:
                findings.append(
                    f"<li><code>{html.escape(c.name)}</code> {html.escape(issue)}</li>"
                )
        findings_html = (
            "<ol class='findings'>" + "".join(findings) + "</ol>"
            if findings
            else "<p class='ok'>No suspected issues.</p>"
        )
        return _HTML_TEMPLATE.format(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            n_dupes=self.n_duplicate_rows,
            n_issues=self.n_issues,
            size=_human_bytes(self.memory_bytes),
            generated=html.escape(self.generated_at),
            findings=findings_html,
            rows="\n".join(rows),
        )


def _repeated_key_issue(df: pd.DataFrame, col: str) -> tuple[str | None, dict[str, Any]]:
    """Describe identifier values that appear on more than one row."""
    keys = df[col].map(_norm_str)
    present = keys[keys != ""]
    dup = present[present.duplicated(keep=False)]
    if dup.empty:
        return None, {}
    conflicting, exact = [], []
    norm = df.loc[dup.index].astype(object).map(lambda v: "" if is_nullish(v) else _norm_str(v))
    for key, idx in dup.groupby(dup, sort=False).groups.items():
        rows = norm.loc[idx]
        (exact if len(rows.drop_duplicates()) == 1 else conflicting).append(str(key))
    parts = []
    if conflicting:
        parts.append(
            f"{', '.join(conflicting[:5])}{'…' if len(conflicting) > 5 else ''} with differing "
            "rows (conflicting records — review, or dedupe with keep: most_complete)"
        )
    if exact:
        parts.append(
            f"{', '.join(exact[:5])}{'…' if len(exact) > 5 else ''} as exact duplicate rows"
        )
    n_keys = len(conflicting) + len(exact)
    issue = f"Key repeated for {n_keys} value(s): " + "; ".join(parts) + "."
    return issue, {"conflicting_keys": conflicting, "duplicate_row_keys": exact}


def profile_dataframe(df: pd.DataFrame) -> ProfileReport:
    """Profile every column of ``df`` and return a :class:`ProfileReport`."""
    columns = [profile_column(df[col]) for col in df.columns]
    for prof, col in zip(columns, df.columns):
        if prof.semantic == "identifier":
            issue, details = _repeated_key_issue(df, col)
            if issue:
                prof.issues.insert(0, issue)
                prof.details.update(details)
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
    level = "high" if pct >= 50 else "mid" if pct >= 20 else "low"
    return (
        "<span class='bar' aria-hidden='true'>"
        f"<span class='bar-fill bar-{level}' style='width:{filled:.0f}%'></span>"
        "</span>"
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Data profile</title>
<style>
  :root {{
    --bg: #f6f7f9; --surface: #ffffff; --border: #e3e6ea; --row: #f0f2f5;
    --text: #14181f; --muted: #5d6673; --faint: #8a93a0;
    --accent: #3b5bdb; --danger: #b42318; --ok: #1a7f37;
    --bar: #e3e6ea; --low: #2f9e44; --mid: #e8a317; --high: #d9480f;
    --chip-bg: #eef1ff; --chip: #3b4cca;
    --num-bg: #e6f6f8; --num: #0b7285; --date-bg: #fff4db; --date: #9c5b00;
    --bool-bg: #e7f6ea; --bool: #1f7a35; --mail-bg: #fdeaf3; --mail: #a61e5d;
    --cat-bg: #eef0f3; --cat: #454d59; --text-bg: #f2eeff; --text-c: #5f3dc4;
    --phone-bg: #e8f1fd; --phone: #1c5fb8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0f1216; --surface: #171b21; --border: #2a3039; --row: #1d222a;
      --text: #e6e9ee; --muted: #a2abb8; --faint: #768090;
      --accent: #8ea2ff; --danger: #ff8a7a; --ok: #6fd08c;
      --bar: #2a3039; --low: #4cc26a; --mid: #f0b84a; --high: #ff7a45;
      --chip-bg: #252c4a; --chip: #b4c0ff;
      --num-bg: #13343a; --num: #7fd3e0; --date-bg: #3a2e14; --date: #f5c56b;
      --bool-bg: #173322; --bool: #86dca0; --mail-bg: #3a1a2a; --mail: #f59ac4;
      --cat-bg: #262b33; --cat: #c3cad4; --text-bg: #2b2342; --text-c: #c3b1ff;
      --phone-bg: #172a45; --phone: #9cc3ff;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0f1216; --surface: #171b21; --border: #2a3039; --row: #1d222a;
    --text: #e6e9ee; --muted: #a2abb8; --faint: #768090;
    --accent: #8ea2ff; --danger: #ff8a7a; --ok: #6fd08c;
    --bar: #2a3039; --low: #4cc26a; --mid: #f0b84a; --high: #ff7a45;
    --chip-bg: #252c4a; --chip: #b4c0ff;
    --num-bg: #13343a; --num: #7fd3e0; --date-bg: #3a2e14; --date: #f5c56b;
    --bool-bg: #173322; --bool: #86dca0; --mail-bg: #3a1a2a; --mail: #f59ac4;
    --cat-bg: #262b33; --cat: #c3cad4; --text-bg: #2b2342; --text-c: #c3b1ff;
    --phone-bg: #172a45; --phone: #9cc3ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 2rem clamp(1rem, 4vw, 2.5rem); background: var(--bg);
    color: var(--text); line-height: 1.5;
  }}
  main {{ max-width: 1280px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.01em; }}
  h2 {{ font-size: 1.05rem; margin: 2rem 0 .75rem; }}
  .sub {{ color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: .75rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: .85rem 1.1rem; }}
  .card .k {{ color: var(--muted); font-size: .72rem; text-transform: uppercase;
    letter-spacing: .05em; }}
  .card .v {{ font-size: 1.35rem; font-weight: 650; margin-top: .15rem;
    font-variant-numeric: tabular-nums; }}
  .card.warn .v {{ color: var(--danger); }}
  .findings {{ background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; margin: 0; padding: .9rem 1.1rem .9rem 2.6rem; font-size: .88rem; }}
  .findings li {{ margin: .2rem 0; overflow-wrap: anywhere; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .85em; background: var(--row); padding: .05rem .35rem; border-radius: 4px; }}
  .wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
  th, td {{ text-align: left; padding: .55rem .75rem; vertical-align: top;
    border-bottom: 1px solid var(--row); }}
  th {{ background: var(--surface); font-weight: 600; color: var(--muted); position: sticky;
    top: 0; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
  td {{ font-variant-numeric: tabular-nums; }}
  td.col {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 600; white-space: nowrap; }}
  .muted {{ color: var(--muted); }}
  .extra {{ color: var(--muted); font-size: .8rem; }}
  .type {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px;
    font-size: .74rem; font-weight: 600; background: var(--chip-bg); color: var(--chip); }}
  .type-integer, .type-float {{ background: var(--num-bg); color: var(--num); }}
  .type-datetime {{ background: var(--date-bg); color: var(--date); }}
  .type-boolean {{ background: var(--bool-bg); color: var(--bool); }}
  .type-email {{ background: var(--mail-bg); color: var(--mail); }}
  .type-phone {{ background: var(--phone-bg); color: var(--phone); }}
  .type-categorical {{ background: var(--cat-bg); color: var(--cat); }}
  .type-text {{ background: var(--text-bg); color: var(--text-c); }}
  .role {{ display: inline-block; margin-left: .3rem; font-size: .72rem; color: var(--faint); }}
  .bar {{ display: inline-block; width: 60px; height: 7px; border-radius: 4px;
    background: var(--bar); margin-right: .4rem; vertical-align: middle; overflow: hidden; }}
  .bar-fill {{ display: block; height: 100%; }}
  .bar-low {{ background: var(--low); }} .bar-mid {{ background: var(--mid); }}
  .bar-high {{ background: var(--high); }}
  ul.issues {{ margin: 0; padding-left: 1.1rem; color: var(--danger); font-size: .8rem; }}
  ul.issues li {{ overflow-wrap: anywhere; }}
  .ok {{ color: var(--ok); font-size: .8rem; }}
  footer {{ margin-top: 1.5rem; color: var(--faint); font-size: .78rem; }}
  @media (max-width: 720px) {{
    body {{ padding: 1.25rem 1rem; }}
    .wrap {{ background: transparent; border: 0; overflow: visible; }}
    table, tbody, tr, td {{ display: block; width: 100%; }}
    thead {{ display: none; }}
    tr {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      margin-bottom: .75rem; padding: .35rem 0; }}
    td {{ border: 0; padding: .3rem .9rem; display: grid;
      grid-template-columns: 6.5rem minmax(0, 1fr); gap: .5rem; overflow-wrap: anywhere; }}
    td::before {{ content: attr(data-label); color: var(--faint); font-size: .72rem;
      text-transform: uppercase; letter-spacing: .04em; padding-top: .15rem; }}
    td.col {{ white-space: normal; }}
  }}
  @media (prefers-reduced-motion: no-preference) {{
    .bar-fill {{ transition: width .4s ease-out; }}
  }}
</style>
</head>
<body>
<main>
  <h1>Data profile</h1>
  <p class="sub">Generated by cleankit &middot; {generated}</p>
  <div class="cards">
    <div class="card"><div class="k">Rows</div><div class="v">{n_rows}</div></div>
    <div class="card"><div class="k">Columns</div><div class="v">{n_cols}</div></div>
    <div class="card"><div class="k">Duplicate rows</div><div class="v">{n_dupes}</div></div>
    <div class="card warn"><div class="k">Suspected issues</div><div class="v">{n_issues}</div></div>
    <div class="card"><div class="k">Size</div><div class="v">{size}</div></div>
  </div>
  <h2>Findings</h2>
  {findings}
  <h2>Columns</h2>
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
  <footer>cleankit &mdash; profile, then pipeline, then rules, then audit.</footer>
</main>
</body>
</html>
"""
