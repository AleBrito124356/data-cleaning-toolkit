"""Small text helpers shared by the profiler, the cleaner and the contacts module.

They answer one question in a predictable way: *are these two spellings the
same thing?* ``Panamá``, ``panama`` and ``Rep. de Panamá`` should be; so should
``Colombia`` and the typo ``Columbia``. ``Austria`` and ``Australia``,
``Gambia`` and ``Zambia``, or ``Category 10`` and ``Category 11`` should not.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np

# Words that decorate a name without changing what it refers to
# ("Rep. de Panamá", "Republic of Colombia", "El Salvador").
NOISE_TOKENS = frozenset(
    {"rep", "republica", "republic", "de", "del", "of", "the", "la", "el"}
)


def is_blank(value: Any) -> bool:
    """True for None/NaN/NA and for strings made only of whitespace."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return True
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        pass
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def canonical_key(value: Any) -> str:
    """A comparison key: accent-free, lower-cased, punctuation collapsed to spaces."""
    if is_blank(value):
        return ""
    s = strip_accents(str(value)).lower()
    return re.sub(r"[^0-9a-z]+", " ", s).strip()


def strip_noise(key: str) -> str:
    """Drop decorative tokens from a canonical key ("rep de panama" -> "panama")."""
    tokens = key.split()
    if len(tokens) < 2:
        return key
    kept = [t for t in tokens if t not in NOISE_TOKENS]
    return " ".join(kept) if kept else key


def _digits(key: str) -> str:
    return "".join(re.findall(r"\d+", key))


def within_one_edit(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by at most one insertion, deletion,
    substitution, or transposition of two adjacent characters."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return j == i + 1 and a[i] == b[j] and a[j] == b[i]
        return False
    if la > lb:
        a, b = b, a
    # b is one longer than a: find the single insertion.
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def same_variant(a: str, b: str) -> bool:
    """Decide whether two canonical keys are spellings of the same value.

    Equal keys (after dropping decorative tokens) always match. Otherwise a
    single typo is tolerated when both keys are at least 5 characters long,
    share the first character, and carry the same digits.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    na, nb = strip_noise(a), strip_noise(b)
    if na == nb:
        return True
    if _digits(na) != _digits(nb):
        return False
    if min(len(na), len(nb)) < 5 or na[0] != nb[0]:
        return False
    return within_one_edit(na, nb)
