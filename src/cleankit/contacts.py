"""Practical standardizers for contact data.

These functions are deliberately dependency-free and predictable: they favour
consistent output over exhaustive coverage. For strict international phone
parsing use the ``phonenumbers`` library; this module targets the common LATAM
cases (Panama first) that show up in spreadsheets exported from CRMs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Common typo domains -> canonical domain.
_DOMAIN_FIXES = {
    "gmial.com": "gmail.com",
    "gmai.com": "gmail.com",
    "gmail.co": "gmail.com",
    "gmail.con": "gmail.com",
    "gmail.cm": "gmail.com",
    "gnail.com": "gmail.com",
    "hotmial.com": "hotmail.com",
    "hotmai.com": "hotmail.com",
    "hotmail.co": "hotmail.com",
    "hotmail.con": "hotmail.com",
    "yahooo.com": "yahoo.com",
    "yaho.com": "yahoo.com",
    "yahoo.con": "yahoo.com",
    "outlok.com": "outlook.com",
    "outlook.co": "outlook.com",
}


@dataclass
class EmailResult:
    value: str | None
    valid: bool
    changed: bool
    note: str = ""


def standardize_email(raw: Any, fix_typos: bool = True) -> EmailResult:
    """Lower-case, trim, strip ``mailto:``, and optionally repair typo domains.

    Returns an :class:`EmailResult`; ``value`` is ``None`` when the input is
    empty. ``valid`` reflects a basic syntactic check on the cleaned value.
    """
    if raw is None:
        return EmailResult(value=None, valid=False, changed=False, note="empty")
    original = str(raw)
    s = original.strip().lower()
    s = re.sub(r"^mailto:", "", s)
    s = s.replace(" ", "")
    if s == "":
        return EmailResult(value=None, valid=False, changed=original != "", note="empty")

    note = ""
    if fix_typos and "@" in s:
        local, _, domain = s.rpartition("@")
        if domain in _DOMAIN_FIXES:
            domain = _DOMAIN_FIXES[domain]
            note = "fixed domain typo"
        s = f"{local}@{domain}"

    valid = bool(_EMAIL_RE.match(s))
    return EmailResult(
        value=s,
        valid=valid,
        changed=s != original,
        note=note or ("" if valid else "invalid format"),
    )


# ---------------------------------------------------------------------------
# Phone numbers -> E.164
# ---------------------------------------------------------------------------

# region -> (country calling code, national number length(s), trunk prefix)
_REGIONS = {
    "PA": {"cc": "507", "lengths": (8,), "trunk": None},   # Panama
    "US": {"cc": "1", "lengths": (10,), "trunk": "1"},     # USA / Canada
    "CA": {"cc": "1", "lengths": (10,), "trunk": "1"},
    "MX": {"cc": "52", "lengths": (10,), "trunk": "01"},   # Mexico
    "CO": {"cc": "57", "lengths": (10,), "trunk": None},   # Colombia
    "CR": {"cc": "506", "lengths": (8,), "trunk": None},   # Costa Rica
    "GT": {"cc": "502", "lengths": (8,), "trunk": None},   # Guatemala
    "SV": {"cc": "503", "lengths": (8,), "trunk": None},   # El Salvador
    "HN": {"cc": "504", "lengths": (8,), "trunk": None},   # Honduras
    "NI": {"cc": "505", "lengths": (8,), "trunk": None},   # Nicaragua
    "DO": {"cc": "1", "lengths": (10,), "trunk": "1"},     # Dominican Rep.
    "PE": {"cc": "51", "lengths": (9,), "trunk": None},    # Peru
    "CL": {"cc": "56", "lengths": (9,), "trunk": None},    # Chile
    "AR": {"cc": "54", "lengths": (10,), "trunk": "0"},    # Argentina
    "EC": {"cc": "593", "lengths": (9,), "trunk": "0"},    # Ecuador
    "VE": {"cc": "58", "lengths": (10,), "trunk": "0"},    # Venezuela
}

# Calling codes sorted longest-first for greedy prefix matching.
_CC_BY_LEN = sorted({r["cc"] for r in _REGIONS.values()}, key=len, reverse=True)


@dataclass
class PhoneResult:
    value: str | None
    valid: bool
    changed: bool
    note: str = ""


def to_e164(raw: Any, default_region: str = "PA") -> PhoneResult:
    """Normalise a phone number to E.164 (``+<cc><national>``).

    Rules of thumb:
      * a leading ``+`` or ``00`` marks an already-international number;
      * otherwise the number is interpreted in ``default_region``, with the
        region's trunk prefix stripped;
      * an 8/9/10-digit number without a country code is prefixed with the
        default region's calling code.
    """
    if raw is None:
        return PhoneResult(value=None, valid=False, changed=False, note="empty")
    original = str(raw)
    s = original.strip()
    if s == "" or s.lower() in {"na", "n/a", "none", "-"}:
        return PhoneResult(value=None, valid=False, changed=original != "", note="empty")

    has_plus = s.lstrip().startswith("+")
    # Keep digits only, but remember an extension marker.
    ext = ""
    ext_match = re.search(r"(?:ext|x|extension)\.?\s*(\d+)$", s, re.IGNORECASE)
    if ext_match:
        ext = ext_match.group(1)
        s = s[: ext_match.start()]

    digits = re.sub(r"\D", "", s)
    if digits == "":
        return PhoneResult(value=None, valid=False, changed=True, note="no digits")

    region = _REGIONS.get(default_region.upper(), _REGIONS["PA"])

    if has_plus or s.strip().startswith("00"):
        national_digits = digits
        if national_digits.startswith("00"):
            national_digits = national_digits[2:]
        cc, national = _split_country_code(national_digits)
        if cc is None:
            return PhoneResult(
                value="+" + national_digits,
                valid=False,
                changed=True,
                note="unknown country code",
            )
    else:
        # Domestic number: strip trunk prefix if present.
        national = digits
        trunk = region["trunk"]
        if trunk and national.startswith(trunk) and len(national) > max(
            region["lengths"]
        ):
            national = national[len(trunk):]
        # Some exports already include the country code without '+'.
        if national.startswith(region["cc"]) and len(national) > max(
            region["lengths"]
        ):
            cc = region["cc"]
            national = national[len(cc):]
        else:
            cc = region["cc"]

    valid = _valid_for_cc(cc, national)
    value = f"+{cc}{national}"
    if ext:
        value += f";ext={ext}"
    return PhoneResult(
        value=value,
        valid=valid,
        changed=value != original,
        note="" if valid else "unexpected length",
    )


def _split_country_code(digits: str) -> tuple[str | None, str]:
    for cc in _CC_BY_LEN:
        if digits.startswith(cc):
            return cc, digits[len(cc):]
    return None, digits


def _valid_for_cc(cc: str, national: str) -> bool:
    for r in _REGIONS.values():
        if r["cc"] == cc and len(national) in r["lengths"]:
            return True
    # Fallback: accept plausible international lengths.
    return 6 <= len(national) <= 12


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

# Lower-case connectors kept lower-cased when they sit inside a name.
_NAME_PARTICLES = {
    "de", "del", "la", "las", "los", "y", "e", "da", "das", "do", "dos",
    "van", "von", "der", "den", "di", "el", "al", "bin", "ibn",
}
# Prefixes that keep an internal capital (Mc, Mac, O').
_MC_RE = re.compile(r"^(mc)(\w)(.*)$", re.IGNORECASE)
_OAPOS_RE = re.compile(r"^(o)'(\w)(.*)$", re.IGNORECASE)


def standardize_name(raw: Any) -> str | None:
    """Title-case a personal name while respecting particles and Mc/O' forms.

    ``"  JUAN   de la CRUZ "`` -> ``"Juan de la Cruz"``;
    ``"maría josé"`` -> ``"María José"``; ``"o'brien"`` -> ``"O'Brien"``.
    """
    if raw is None:
        return None
    s = unicodedata.normalize("NFC", str(raw)).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "":
        return None

    words = s.split(" ")
    out: list[str] = []
    for i, word in enumerate(words):
        out.append(_case_name_word(word, is_first=(i == 0)))
    return " ".join(out)


def _case_name_word(word: str, is_first: bool) -> str:
    # Preserve hyphenated components (e.g. Jean-Paul).
    if "-" in word:
        return "-".join(_case_name_word(p, is_first) for p in word.split("-"))
    lower = word.lower()
    if not is_first and lower in _NAME_PARTICLES:
        return lower
    mc = _MC_RE.match(word)
    if mc:
        return "Mc" + mc.group(2).upper() + mc.group(3).lower()
    oa = _OAPOS_RE.match(word)
    if oa:
        return "O'" + oa.group(2).upper() + oa.group(3).lower()
    if not word[:1].isalpha():
        return word
    return word[:1].upper() + word[1:].lower()


# ---------------------------------------------------------------------------
# Addresses (light cleanup)
# ---------------------------------------------------------------------------

_ADDRESS_ABBREV = {
    r"\bave\b": "Avenida",
    r"\bav\b": "Avenida",
    r"\bavda\b": "Avenida",
    r"\bcl\b": "Calle",
    r"\bc/\b": "Calle",
    r"\bblvd\b": "Boulevard",
    r"\bapto\b": "Apartamento",
    r"\bapt\b": "Apartamento",
    r"\bedif\b": "Edificio",
    r"\bcorregimiento\b": "Corregimiento",
    r"\bph\b": "PH",
    r"\bno\.?\b": "No.",
}


def clean_address(raw: Any) -> str | None:
    """Trim, collapse whitespace, tidy punctuation, and expand common
    Spanish street abbreviations. Title-cases the result while keeping short
    connectors lower-cased.
    """
    if raw is None:
        return None
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "":
        return None
    # Normalise spacing around commas.
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r",\s*,", ",", s)
    s = s.strip(" ,")

    lowered = s.lower()
    for pat, repl in _ADDRESS_ABBREV.items():
        lowered = re.sub(pat, repl.lower(), lowered)

    # Title-case, keeping particles lower.
    words = lowered.split(" ")
    out: list[str] = []
    for i, word in enumerate(words):
        stripped = word.strip(",")
        trailing = "," if word.endswith(",") else ""
        if stripped.upper() in {"PH", "No."}:
            out.append(stripped.upper() if stripped.upper() == "PH" else "No." + trailing)
            continue
        if i != 0 and stripped in _NAME_PARTICLES:
            out.append(stripped + trailing)
        elif stripped.isdigit() or not stripped[:1].isalpha():
            out.append(stripped + trailing)
        else:
            out.append(stripped[:1].upper() + stripped[1:] + trailing)
    return " ".join(out)
