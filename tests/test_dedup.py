"""Scalable, auditable deduplication.

The pruned fuzzy matcher must mark exactly the rows the 0.1.0 quadratic loop
marked; it may only skip comparisons whose exact upper bound is already
below the threshold.
"""

import random
from difflib import SequenceMatcher

import pandas as pd
import pytest

from cleankit import Cleaner
from cleankit.clean import _canonical_key, fuzzy_matches


def _reference_mask(df: pd.DataFrame, keys, threshold: float, keep: str) -> pd.Series:
    """Verbatim copy of cleankit 0.1.0's Cleaner._fuzzy_duplicate_mask."""
    keys = [k for k in keys if k in df.columns]
    sig = df[keys].astype(object).apply(
        lambda row: " ".join(_canonical_key(v) for v in row), axis=1
    )
    blocks: dict = {}
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


def _random_frame(seed: int, n: int = 250) -> pd.DataFrame:
    rng = random.Random(seed)
    words = ["ana", "maria", "jose", "juan", "perez", "gomez", "lopez", "diaz", "ruiz", "rojas", "cruz", "vega"]
    rows = []
    for _ in range(n):
        named = [r["name"] for r in rows if r["name"]]
        if named and rng.random() < 0.3:
            base = list(rng.choice(named))
            if base:
                base[rng.randrange(len(base))] = rng.choice("aeioux ")
            name = "".join(base)
        else:
            name = " ".join(rng.choice(words) for _ in range(rng.randint(1, 3)))
        if rng.random() < 0.05:
            name = None
        rows.append({"name": name, "city": rng.choice(["PTY", "BOG", "", None])})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("threshold", [0.6, 0.8, 0.9, 0.97])
@pytest.mark.parametrize("keep", ["first", "last"])
def test_fuzzy_mask_matches_the_reference(seed, threshold, keep):
    df = _random_frame(seed)
    for keys in (["name"], ["name", "city"]):
        new = Cleaner(df)._fuzzy_duplicate_mask(keys, threshold, keep)
        old = _reference_mask(df, keys, threshold, keep)
        assert new.tolist() == old.tolist()


def test_fuzzy_matches_reports_best_kept_row():
    sigs = ["jonathan smith", "jonathon smith", "jonathan smyth", "maria gomez"]
    result = fuzzy_matches(sigs, 0.85, order=[0, 1, 2, 3])
    assert set(result) == {1, 2}
    kept, score = result[1]
    assert kept == 0 and score == pytest.approx(SequenceMatcher(None, sigs[1], sigs[0]).ratio())


def test_pairs_are_logged_for_exact_and_fuzzy_removals():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3],
            "name": ["Jonathan Smith", "Jonathan Smith", "Jonathon Smith", "Maria Gomez"],
        }
    )
    c = Cleaner(df).deduplicate(subset=["id"], fuzzy=True, fuzzy_keys=["name"], threshold=0.85)
    entry = c.log[-1]
    assert entry["exact"] == 1 and entry["fuzzy"] == 1
    assert entry["pairs"][0] == {"row": 1, "kept_row": 0, "match": "exact", "score": 1.0}
    fuzzy = entry["pairs"][1]
    assert fuzzy["row"] == 2 and fuzzy["kept_row"] == 0 and fuzzy["match"] == "fuzzy"
    assert fuzzy["key"] == "jonathon smith" and fuzzy["kept_key"] == "jonathan smith"
    assert c.df["name"].tolist() == ["Jonathan Smith", "Maria Gomez"]
    assert c.review[1]["values"]["name"] == "Jonathon Smith"
    assert c.review[1]["kept_values"]["name"] == "Jonathan Smith"


def test_keep_most_complete_exact():
    df = pd.DataFrame(
        {
            "id": ["7", "7", "7"],
            "email": [None, "a@b.com", "a@b.com"],
            "phone": [None, None, "+50761234567"],
        }
    )
    c = Cleaner(df).deduplicate(subset=["id"], keep="most_complete")
    assert c.df.index.tolist() == [2]
    assert {p["row"] for p in c.log[-1]["pairs"]} == {0, 1}
    assert all(p["kept_row"] == 2 for p in c.log[-1]["pairs"])


def test_keep_most_complete_fuzzy():
    df = pd.DataFrame({"name": ["Jon Smith", "John Smith"], "email": [None, "j@x.com"]})
    c = Cleaner(df).deduplicate(fuzzy=True, fuzzy_keys=["name"], threshold=0.85, keep="most_complete")
    assert c.df["name"].tolist() == ["John Smith"]


def test_block_on_limits_comparisons():
    df = pd.DataFrame({"name": ["Ana Diaz", "Ana Dias"], "country": ["PA", "CO"]})
    same_block = Cleaner(df).deduplicate(fuzzy=True, fuzzy_keys=["name"], threshold=0.8)
    assert len(same_block.df) == 1
    blocked = Cleaner(df).deduplicate(fuzzy=True, fuzzy_keys=["name"], threshold=0.8, block_on=["country"])
    assert len(blocked.df) == 2


def test_flag_mode_records_duplicate_of():
    df = pd.DataFrame({"id": [1, 1, 2]})
    c = Cleaner(df).deduplicate(subset=["id"], flag=True)
    assert c.df["is_duplicate"].tolist() == [False, True, False]
    assert c.df["duplicate_of"].tolist()[1] == 0
    assert pd.isna(c.df["duplicate_of"].iloc[0])


def test_fuzzy_never_removes_a_row_whose_match_was_removed():
    # Exact pass drops row 1 (repeated id); fuzzy must not then drop row 2 as
    # a duplicate of the already-removed row 1.
    df = pd.DataFrame({"id": [1, 1, 2], "name": ["Ann", "Zed Zed", "Zed Zed"]})
    c = Cleaner(df).deduplicate(subset=["id"], fuzzy=True, fuzzy_keys=["name"], threshold=0.9)
    assert c.df["name"].tolist() == ["Ann", "Zed Zed"]


def test_deduplicate_validates_options():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="keep"):
        Cleaner(df).deduplicate(keep="best")
    with pytest.raises(ValueError, match="threshold"):
        Cleaner(df).deduplicate(fuzzy=True, threshold=1.5)
    with pytest.raises(ValueError, match="not found"):
        Cleaner(df).deduplicate(subset=["nope"])


def test_row_labels_survive_drops():
    df = pd.DataFrame({"id": [1, 1, 2, 2], "v": ["a", "a", "b", "c"]})
    c = Cleaner(df).deduplicate(subset=["id"]).handle_missing("drop_rows", ["v"])
    assert c.df.index.tolist() == [0, 2]
    assert [p["row"] for p in c.log[0]["pairs"]] == [1, 3]
