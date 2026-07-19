#!/usr/bin/env python3
"""cleankit CLI entry point.

Run directly from a clone without installing anything:

    python cli.py profile data/messy_sample.csv
    python cli.py run-pipeline data/messy_sample.csv --steps steps.yaml --out cleaned.csv
    python cli.py validate cleaned.csv --rules rules.yaml

The real implementation lives in ``src/cleankit/cli.py``. This shim puts
``src`` on the path so the package imports without an editable install; after
``pip install -e .`` the same CLI is available as ``cleankit`` and
``python -m cleankit``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cleankit.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
