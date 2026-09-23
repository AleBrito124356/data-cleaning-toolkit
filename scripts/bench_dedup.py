#!/usr/bin/env python3
"""Benchmark fuzzy deduplication: the pre-0.2 quadratic loop vs the pruned one.

Generates a seeded, realistic name list (first + last names, ~15% near
duplicates with one typo) and times ``Cleaner.deduplicate(fuzzy=True)``
against a verbatim copy of the old algorithm, checking that both mark exactly
the same rows.

    python scripts/bench_dedup.py                 # 2,000 and 5,000 rows, old vs new
    python scripts/bench_dedup.py --sizes 20000 --skip-old
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from cleankit import Cleaner  # noqa: E402
from cleankit.clean import _canonical_key  # noqa: E402

FIRST = [
    "Juan", "María", "José", "Ana", "Carlos", "Lucía", "Pedro", "Sofía", "Luis",
    "Elena", "Jorge", "Gabriela", "Daniel", "Valentina", "Miguel", "Camila",
    "Andrés", "Isabel", "Ricardo", "Paula", "Fernando", "Laura", "Diego",
    "Carmen", "Roberto", "Natalia", "Alejandro", "Mariana", "Javier", "Daniela",
]
LAST = [
    "Pérez", "González", "Rodríguez", "López", "Martínez", "Sánchez", "Ramírez",
    "Torres", "Flores", "Rivera", "Gómez", "Díaz", "Vargas", "Castro", "Rojas",
    "Morales", "Ortiz", "Gutiérrez", "Chávez", "Ruiz", "Mendoza", "Herrera",
    "Aguilar", "Medina", "Castillo", "Jiménez", "Moreno", "Romero", "Álvarez",
    "Navarro", "Domínguez", "Vega", "Ríos", "Cruz", "Reyes", "Molina",
]


def make_names(n: int, seed: int = 7, dup_rate: float = 0.15) -> list[str]:
    rng = random.Random(seed)
    names: list[str] = []
    for _ in range(n):
        if names and rng.random() < dup_rate:
            base = list(rng.choice(names))
            i = rng.randrange(len(base))
            base[i] = rng.choice("aeiourns")  # one-character typo
            names.append("".join(base))
        else:
            names.append(
                f"{rng.choice(FIRST)} {rng.choice(FIRST)} {rng.choice(LAST)} {rng.choice(LAST)}"
            )
    return names


def old_fuzzy_mask(df: pd.DataFrame, keys: list[str], threshold: float, keep: str) -> pd.Series:
    """The cleankit 0.1.0 implementation, kept verbatim as the reference."""
    sig = df[keys].astype(object).apply(
        lambda row: " ".join(_canonical_key(v) for v in row), axis=1
    )
    blocks: dict[str, list] = {}
    order = list(sig.index)
    if keep == "last":
        order = list(reversed(order))
    for idx in order:
        blocks.setdefault(sig[idx][:1], []).append(idx)
    mask = pd.Series(False, index=df.index)
    for members in blocks.values():
        kept: list[str] = []
        for idx in members:
            s = sig[idx]
            if not s:
                continue
            if any(SequenceMatcher(None, s, ks).ratio() >= threshold for ks in kept):
                mask[idx] = True
            else:
                kept.append(s)
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000])
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--skip-old", action="store_true", help="only time the new implementation")
    args = ap.parse_args()

    print(f"threshold={args.threshold}")
    print(f"{'rows':>7} {'old (s)':>9} {'new (s)':>9} {'speed-up':>9} {'dupes':>6}  identical")
    for n in args.sizes:
        df = pd.DataFrame({"name": make_names(n)})
        t0 = time.perf_counter()
        c = Cleaner(df).deduplicate(fuzzy=True, fuzzy_keys=["name"], threshold=args.threshold, flag=True)
        t_new = time.perf_counter() - t0
        new_mask = c.df["is_duplicate"]
        if args.skip_old:
            print(f"{n:>7} {'-':>9} {t_new:>9.2f} {'-':>9} {int(new_mask.sum()):>6}  -")
            continue
        t0 = time.perf_counter()
        old = old_fuzzy_mask(df, ["name"], args.threshold, "first")
        t_old = time.perf_counter() - t0
        same = bool((old.values == new_mask.values).all())
        print(f"{n:>7} {t_old:>9.2f} {t_new:>9.2f} {t_old / t_new:>8.1f}x {int(new_mask.sum()):>6}  {same}")
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
