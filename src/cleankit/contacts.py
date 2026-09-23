"""Practical standardizers for contact data.

These functions are deliberately dependency-free and predictable: they favour
consistent output over exhaustive coverage. For strict international phone
parsing use the ``phonenumbers`` library; this module targets the common LATAM
cases (Panama first) that show up in spreadsheets exported from CRMs.

The same functions back the audited pipeline operations
``Cleaner.standardize_emails``, ``standardize_phones``, ``standardize_names``
and ``clean_addresses``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .text import canonical_key, is_blank, same_variant, strip_noise

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$"
)

# Mis-typed provider names -> the real one.
_PROVIDER_TYPOS = {
    "gmail": ["gmial", "gmai", "gamil", "gmaill", "gnail", "gmal", "gmali", "gmsil", "gmil"],
    "hotmail": ["hotmial", "hotmai", "hotmal", "homail", "hotmil", "hotamil", "hotmaill", "hormail"],
    "yahoo": ["yahooo", "yaho", "yhoo", "yahho"],
    "outlook": ["outlok", "outloo", "otlook", "outllok"],
    "icloud": ["iclod", "icoud", "iclound"],
}
# Mis-typed top-level domains for those providers -> ".com".
_TLD_TYPOS = {"con", "cm", "co", "om", "comm", "cmo", "ocm", "vom", "xom", "cpm", "coom"}
_PROVIDERS = set(_PROVIDER_TYPOS)
_NAME_FIXES = {typo: real for real, typos in _PROVIDER_TYPOS.items() for typo in typos}
KNOWN_PROVIDER_DOMAINS = tuple(sorted(f"{p}.com" for p in _PROVIDERS))


@dataclass
class EmailResult:
    value: str | None
    valid: bool
    changed: bool
    note: str = ""
    suggestion: str | None = None


def fix_email_domain(domain: str) -> str:
    """Repair a mis-typed provider domain (``gmial.con`` -> ``gmail.com``).

    Only well-known consumer providers are touched; anything else is returned
    unchanged, because a "close" domain may be a different, real one
    (``ymail.com`` is not a typo of ``gmail.com``).
    """
    name, dot, tld = domain.rpartition(".")
    if not dot:
        return domain
    name = _NAME_FIXES.get(name, name)
    if name in _PROVIDERS and tld in _TLD_TYPOS:
        tld = "com"
    return f"{name}.{tld}"


def _missing_at_suggestion(s: str) -> str | None:
    """``valentina.rojasgmail.com`` -> ``valentina.rojas@gmail.com`` (a hint only)."""
    if "@" in s:
        return None
    for domain in KNOWN_PROVIDER_DOMAINS:
        if s.endswith(domain) and len(s) > len(domain):
            local = s[: -len(domain)]
            if local and not local.endswith("."):
                return f"{local}@{domain}"
    return None


def standardize_email(raw: Any, fix_typos: bool = True) -> EmailResult:
    """Lower-case, trim, strip ``mailto:``, and optionally repair typo domains.

    Returns an :class:`EmailResult`; ``value`` is ``None`` when the input is
    empty. ``valid`` reflects a syntactic check on the cleaned value. For a
    malformed address that is probably missing its ``@`` a ``suggestion`` is
    offered, but the value itself is never guessed.
    """
    if is_blank(raw):
        original = "" if raw is None else str(raw)
        return EmailResult(value=None, valid=False, changed=original != "", note="empty")
    original = str(raw)
    s = original.strip().lower()
    s = re.sub(r"^mailto:", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.strip("<>\"'")

    note = ""
    if fix_typos and "@" in s:
        local, _, domain = s.rpartition("@")
        fixed = fix_email_domain(domain)
        if fixed != domain:
            note = "fixed domain typo"
        s = f"{local}@{fixed}"

    valid = bool(_EMAIL_RE.match(s))
    suggestion = None if valid else _missing_at_suggestion(s)
    return EmailResult(
        value=s,
        valid=valid,
        changed=s != original,
        note=note or ("" if valid else "invalid format"),
        suggestion=suggestion,
    )


# ---------------------------------------------------------------------------
# Phone numbers -> E.164
# ---------------------------------------------------------------------------

# region -> country calling code, national significant number lengths,
# domestic trunk prefix, and a pattern the national number must match.
_REGIONS: dict[str, dict[str, Any]] = {
    # Panama: 8-digit mobiles start with 6; landlines keep 7 digits.
    "PA": {"cc": "507", "lengths": (7, 8), "trunk": None, "pattern": r"6\d{7}|[2-57-9]\d{6}"},
    "US": {"cc": "1", "lengths": (10,), "trunk": "1", "pattern": r"[2-9]\d{2}[2-9]\d{6}"},
    "CA": {"cc": "1", "lengths": (10,), "trunk": "1", "pattern": r"[2-9]\d{2}[2-9]\d{6}"},
    "DO": {"cc": "1", "lengths": (10,), "trunk": "1", "pattern": r"(?:809|829|849)\d{7}"},
    "MX": {"cc": "52", "lengths": (10,), "trunk": "01", "pattern": r"[1-9]\d{9}"},
    # Colombia: mobiles 3xx xxx xxxx; landlines 60x xxx xxxx since 2021.
    "CO": {"cc": "57", "lengths": (10,), "trunk": None, "pattern": r"3\d{9}|60\d{8}"},
    "CR": {"cc": "506", "lengths": (8,), "trunk": None, "pattern": r"[245678]\d{7}"},
    "GT": {"cc": "502", "lengths": (8,), "trunk": None, "pattern": r"[2-7]\d{7}"},
    "SV": {"cc": "503", "lengths": (8,), "trunk": None, "pattern": r"[267]\d{7}"},
    "HN": {"cc": "504", "lengths": (8,), "trunk": None, "pattern": r"[2-9]\d{7}"},
    "NI": {"cc": "505", "lengths": (8,), "trunk": None, "pattern": r"[2-8]\d{7}"},
    "PE": {"cc": "51", "lengths": (8, 9), "trunk": "0", "pattern": r"9\d{8}|[1-8]\d{7}"},
    "CL": {"cc": "56", "lengths": (9,), "trunk": None, "pattern": r"[2-9]\d{8}"},
    "AR": {"cc": "54", "lengths": (10, 11), "trunk": "0", "pattern": r"9\d{10}|[1-9]\d{9}"},
    "EC": {"cc": "593", "lengths": (8, 9), "trunk": "0", "pattern": r"9\d{8}|[2-7]\d{7}"},
    "VE": {"cc": "58", "lengths": (10,), "trunk": "0", "pattern": r"[24]\d{9}"},
    "BR": {"cc": "55", "lengths": (10, 11), "trunk": "0", "pattern": r"[1-9]{2}9?\d{8}"},
    "ES": {"cc": "34", "lengths": (9,), "trunk": None, "pattern": r"[5-9]\d{8}"},
    "FR": {"cc": "33", "lengths": (9,), "trunk": "0", "pattern": r"[1-9]\d{8}"},
}
for _r in _REGIONS.values():
    _r["regex"] = re.compile(rf"^(?:{_r['pattern']})$")

KNOWN_REGIONS = tuple(sorted(_REGIONS))

# Calling codes sorted longest-first for greedy prefix matching.
_CC_BY_LEN = sorted({r["cc"] for r in _REGIONS.values()}, key=len, reverse=True)
_REGIONS_BY_CC: dict[str, list[str]] = {}
for _code, _r in _REGIONS.items():
    _REGIONS_BY_CC.setdefault(_r["cc"], []).append(_code)


@dataclass
class PhoneResult:
    value: str | None
    valid: bool
    changed: bool
    note: str = ""
    region: str | None = None


def _check_region(region: str | None) -> str:
    code = (region or "").strip().upper()
    if code not in _REGIONS:
        raise ValueError(
            f"unknown phone region {region!r}; known regions: {', '.join(KNOWN_REGIONS)}"
        )
    return code


def _valid_national(cc: str, national: str) -> str | None:
    """Return the region whose numbering plan accepts ``national``, else None."""
    for code in _REGIONS_BY_CC.get(cc, []):
        r = _REGIONS[code]
        if len(national) in r["lengths"] and r["regex"].match(national):
            return code
    return None


def _valid_for_cc(cc: str, national: str) -> bool:
    """True when ``national`` fits the numbering plan of calling code ``cc``."""
    return _valid_national(cc, national) is not None


def _international_candidates(digits: str) -> list[tuple[str, str]]:
    """(cc, national) readings of an international digit string."""
    out: list[tuple[str, str]] = []
    for cc in _CC_BY_LEN:
        if digits.startswith(cc):
            national = digits[len(cc):]
            out.append((cc, national))
            # Legacy Mexican mobile prefix: +52 1 55 1234 5678.
            if cc == "52" and len(national) == 11 and national.startswith("1"):
                out.append((cc, national[1:]))
            # A domestic trunk prefix left in after the country code: +593 0 99...
            for code in _REGIONS_BY_CC[cc]:
                trunk = _REGIONS[code]["trunk"]
                if trunk and national.startswith(trunk):
                    out.append((cc, national[len(trunk):]))
    return out


def _domestic_candidates(digits: str, region: str) -> list[tuple[str, str]]:
    r = _REGIONS[region]
    cc, trunk = r["cc"], r["trunk"]
    out = [(cc, digits)]
    if trunk and digits.startswith(trunk):
        out.append((cc, digits[len(trunk):]))
    # Exports often carry the country code without the '+'.
    if digits.startswith(cc):
        out.append((cc, digits[len(cc):]))
    return out


def to_e164(raw: Any, default_region: str = "PA") -> PhoneResult:
    """Normalise a phone number to E.164 (``+<cc><national>``).

    Rules of thumb:
      * a leading ``+`` or ``00`` marks an already-international number;
      * otherwise the number is read in ``default_region``: as written, with
        the domestic trunk prefix removed, or with a country code that was
        typed without its ``+`` (``507 6123 4567``);
      * the first reading that fits the country's numbering plan (length and
        leading digits) wins, and ``valid`` says whether any reading did.

    ``default_region`` must be one of :data:`KNOWN_REGIONS`; an unknown code
    raises ``ValueError`` instead of silently using Panama.
    """
    region = _check_region(default_region)
    if is_blank(raw) or str(raw).strip().lower() in {"na", "n/a", "none", "-"}:
        original = "" if raw is None else str(raw)
        return PhoneResult(value=None, valid=False, changed=original != "", note="empty")
    original = str(raw)
    s = original.strip()

    # Keep digits only, but remember an extension marker.
    ext = ""
    ext_match = re.search(r"(?:ext|x|extension)\.?\s*(\d+)$", s, re.IGNORECASE)
    if ext_match:
        ext = ext_match.group(1)
        s = s[: ext_match.start()]

    digits = re.sub(r"\D", "", s)
    if digits == "":
        return PhoneResult(value=None, valid=False, changed=True, note="no digits")

    international = s.startswith("+") or s.startswith("00")
    if international:
        if digits.startswith("00"):
            digits = digits[2:]
        candidates = _international_candidates(digits)
        if not candidates:
            return PhoneResult(
                value="+" + digits,
                valid=False,
                changed=True,
                note="country code not in the region table",
            )
    else:
        candidates = _domestic_candidates(digits, region)

    chosen, matched_region = candidates[0], None
    for cc, national in candidates:
        found = _valid_national(cc, national)
        if found:
            chosen, matched_region = (cc, national), found
            break

    cc, national = chosen
    value = f"+{cc}{national}"
    if ext:
        value += f";ext={ext}"
    valid = matched_region is not None
    return PhoneResult(
        value=value,
        valid=valid,
        changed=value != original,
        note="" if valid else "does not fit the numbering plan",
        region=matched_region if valid else (None if international else region),
    )


def to_e164_bare_international(raw: Any) -> PhoneResult | None:
    """Read a number as international although it lacks ``+`` (``507-6987-6543``).

    Returns a valid result or ``None``; used as a last resort when the number
    does not fit the row's own region.
    """
    if is_blank(raw):
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) < 8:
        return None
    for cc, national in _international_candidates(digits):
        found = _valid_national(cc, national)
        if found:
            value = f"+{cc}{national}"
            return PhoneResult(
                value=value,
                valid=True,
                changed=value != str(raw),
                note="read as an international number without '+'",
                region=found,
            )
    return None


# ---------------------------------------------------------------------------
# Countries -> phone regions
# ---------------------------------------------------------------------------

_COUNTRY_NAMES = {
    "PA": ["panama", "republica de panama"],
    "US": ["united states", "united states of america", "estados unidos",
           "estados unidos de america", "eeuu", "ee uu", "usa", "u s a", "us"],
    "CA": ["canada"],
    "MX": ["mexico", "estados unidos mexicanos"],
    "CO": ["colombia", "republica de colombia"],
    "CR": ["costa rica"],
    "GT": ["guatemala"],
    "SV": ["el salvador"],
    "HN": ["honduras"],
    "NI": ["nicaragua"],
    "DO": ["republica dominicana", "dominican republic", "rep dominicana"],
    "PE": ["peru"],
    "CL": ["chile"],
    "AR": ["argentina"],
    "EC": ["ecuador"],
    "VE": ["venezuela"],
    "BR": ["brasil", "brazil"],
    "ES": ["espana", "spain"],
    "FR": ["francia", "france"],
}
_ISO3 = {
    "PA": "pan", "US": "usa", "CA": "can", "MX": "mex", "CO": "col", "CR": "cri",
    "GT": "gtm", "SV": "slv", "HN": "hnd", "NI": "nic", "DO": "dom", "PE": "per",
    "CL": "chl", "AR": "arg", "EC": "ecu", "VE": "ven", "BR": "bra", "ES": "esp",
    "FR": "fra",
}
_COUNTRY_INDEX: dict[str, str] = {}
for _code, _names in _COUNTRY_NAMES.items():
    for _name in _names + [_code.lower(), _ISO3[_code]]:
        _key = canonical_key(_name)
        _COUNTRY_INDEX[_key] = _code
        _COUNTRY_INDEX[strip_noise(_key)] = _code


def region_for_country(value: Any) -> str | None:
    """Map a free-text country (``"Rep. de Panamá"``, ``"méxico"``, ``"CO"``,
    ``"Columbia"``) to a phone region code, or ``None`` if unknown."""
    key = canonical_key(value)
    if not key:
        return None
    for candidate in (key, strip_noise(key)):
        if candidate in _COUNTRY_INDEX:
            return _COUNTRY_INDEX[candidate]
    # Tolerate a single typo in longer names ("Columbia", "Mexcio").
    if len(key) >= 5:
        for known, code in _COUNTRY_INDEX.items():
            if len(known) >= 5 and same_variant(key, known):
                return code
    return None


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

# Lower-case connectors kept lower-cased when they sit inside a name.
_NAME_PARTICLES = {
    "de", "del", "la", "las", "los", "y", "e", "da", "das", "do", "dos",
    "van", "von", "der", "den", "di", "el", "al", "bin", "ibn",
}
# Prefixes that keep an internal capital (Mc, O', D').
_MC_RE = re.compile(r"^(mc)(\w)(.*)$", re.IGNORECASE)
_APOS_RE = re.compile(r"^(\w)['’](\w)(.*)$", re.IGNORECASE)


def standardize_name(raw: Any) -> str | None:
    """Title-case a personal name while respecting particles and Mc/O' forms.

    ``"  JUAN   de la CRUZ "`` -> ``"Juan de la Cruz"``;
    ``"maría josé"`` -> ``"María José"``; ``"o'brien"`` -> ``"O'Brien"``.
    """
    if is_blank(raw):
        return None
    s = unicodedata.normalize("NFC", str(raw)).strip()
    s = re.sub(r"\s+", " ", s)
    words = s.split(" ")
    return " ".join(_case_name_word(w, is_first=(i == 0)) for i, w in enumerate(words))


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
    ap = _APOS_RE.match(word)
    if ap:
        return ap.group(1).upper() + "'" + ap.group(2).upper() + ap.group(3).lower()
    if "." in word.strip("."):
        # Initials: "j.r." -> "J.R."
        return re.sub(r"(^|\.)(\w)", lambda m: m.group(1) + m.group(2).upper(), lower)
    if not word[:1].isalpha():
        return word
    return word[:1].upper() + word[1:].lower()


# ---------------------------------------------------------------------------
# Addresses (light cleanup)
# ---------------------------------------------------------------------------

# Abbreviations matched on a whole token (case-insensitive, optional trailing
# dot) and replaced by their expansion.
_ADDRESS_ABBREV = {
    "ave": "Avenida", "av": "Avenida", "avda": "Avenida", "avd": "Avenida",
    "cl": "Calle", "cll": "Calle", "c/": "Calle",
    "cra": "Carrera", "kra": "Carrera", "kr": "Carrera",
    "dg": "Diagonal", "diag": "Diagonal",
    "tv": "Transversal", "transv": "Transversal",
    "blvd": "Boulevard",
    "apto": "Apartamento", "apt": "Apartamento",
    "edif": "Edificio",
    "ofic": "Oficina",
    "urb": "Urbanización",
    "correg": "Corregimiento",
    "ph": "PH",
    "no": "No.", "nº": "No.", "n°": "No.", "nro": "No.",
}
_ADDRESS_PARTICLES = {"de", "del", "la", "las", "los", "y", "e", "el", "of"}
_ORDINAL_SUFFIXES = {"er", "ra", "ro", "do", "da", "to", "ta", "vo", "va", "mo", "ma", "st", "nd", "rd", "th"}


def clean_address(raw: Any) -> str | None:
    """Trim, collapse whitespace, tidy punctuation, and expand common
    Spanish street abbreviations (``Ave`` -> ``Avenida``, ``Cra`` ->
    ``Carrera``, ``Apto`` -> ``Apartamento``).

    Words are title-cased with short connectors kept lower-case. Acronyms such
    as ``CDMX`` or ``PH`` and unit codes such as ``12B`` are preserved, and
    ``No.`` is never doubled.
    """
    if is_blank(raw):
        return None
    s = unicodedata.normalize("NFKC", str(raw)).strip()
    s = re.sub(r"\s+", " ", s)
    # Normalise spacing around commas and drop empty segments.
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(,\s*)+,", ",", s)
    s = s.strip(" ,")
    if not s:
        return None

    letters = [ch for ch in s if ch.isalpha()]
    shouting = bool(letters) and all(ch.isupper() for ch in letters)

    out: list[str] = []
    for i, token in enumerate(s.split(" ")):
        m = re.match(r"^(.*?)([,;]*)$", token)
        core, trail = m.group(1), m.group(2)
        out.append(_case_address_token(core, i == 0, shouting) + trail)
    return " ".join(out)


def _case_address_token(core: str, is_first: bool, shouting: bool) -> str:
    if not core:
        return core
    lookup = core.lower().rstrip(".")
    if lookup in _ADDRESS_ABBREV or core.lower() in _ADDRESS_ABBREV:
        return _ADDRESS_ABBREV.get(lookup, _ADDRESS_ABBREV.get(core.lower(), core))
    if any(ch.isdigit() for ch in core):
        letters = "".join(ch for ch in core if ch.isalpha())
        if letters.lower() in _ORDINAL_SUFFIXES:
            return core.lower()
        return core.upper() if len(letters) <= 2 else core
    alpha = [ch for ch in core if ch.isalpha()]
    if not alpha:
        return core
    if not shouting and len(alpha) >= 2 and all(ch.isupper() for ch in alpha):
        return core  # acronym such as CDMX or USA
    if not is_first and core.lower() in _ADDRESS_PARTICLES:
        return core.lower()
    return core[:1].upper() + core[1:].lower()
