"""CLI defaults from the environment or a ``.env`` file.

cleankit needs no secrets. Two optional settings only change CLI defaults:

``CLEANKIT_DEFAULT_REGION``
    Phone region for numbers without a country code (``PA``, ``CO``, ``MX``...).
``CLEANKIT_CSV_SEP``
    CSV delimiter when ``--sep`` is not given.

:func:`load_env_file` reads them from a dotenv-style file without any extra
dependency. Real environment variables always win over the file, and flags
always win over both.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_KEYS = ("CLEANKIT_DEFAULT_REGION", "CLEANKIT_CSV_SEP")


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines (``export`` prefix, quotes and ``#`` comments allowed)."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            # An unquoted value ends at an inline comment: KEY=PA  # Panama
            hash_at = value.find(" #")
            if hash_at != -1:
                value = value[:hash_at].rstrip()
        values[key] = value
    return values


def load_env_file(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Apply the documented ``CLEANKIT_*`` keys from ``path`` to ``os.environ``.

    Returns the keys that were applied. A missing file is not an error (the
    file is optional); unknown keys are ignored. Existing environment
    variables are kept unless ``override`` is true.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    parsed = parse_env_text(p.read_text(encoding="utf-8-sig"))
    applied: dict[str, str] = {}
    for key in ENV_KEYS:
        if key in parsed and (override or key not in os.environ):
            os.environ[key] = parsed[key]
            applied[key] = parsed[key]
    return applied


def default_region() -> str:
    return (os.environ.get("CLEANKIT_DEFAULT_REGION") or "PA").strip().upper()


def default_sep() -> str:
    return os.environ.get("CLEANKIT_CSV_SEP") or ","
