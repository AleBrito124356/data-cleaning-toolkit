"""Golden end-to-end run of the bundled demo: data/messy_sample.csv +
steps.yaml + rules.yaml, exactly as the README shows it."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cleankit import load_pipeline, load_rules, run_pipeline, validate

ROOT = Path(__file__).resolve().parents[1]


def _raw() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data" / "messy_sample.csv", dtype=str, keep_default_na=False, na_values=[""]
    )


@pytest.fixture(scope="module")
def cleaned(tmp_path_factory) -> pd.DataFrame:
    result = run_pipeline(_raw(), load_pipeline(ROOT / "steps.yaml"))
    path = tmp_path_factory.mktemp("demo") / "cleaned.csv"
    result.df.to_csv(path, index=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""]).set_index(
        "customer_id", drop=False
    )


def test_row_counts_and_hash():
    result = run_pipeline(_raw(), load_pipeline(ROOT / "steps.yaml"))
    assert result.before["rows"] == 20 and result.after["rows"] == 16
    assert result.input_hash == "0b90aee69dc0387d"
    assert result.after["duplicate_rows"] == 0
    ops = [e["op"] for e in result.audit_log]
    assert ops.count("coerce_types") == 6 and ops[-1] == "deduplicate"


def test_values_are_really_clean(cleaned):
    row = cleaned.loc
    assert row["1001", "email_address"] == "juan.perez@gmail.com"
    assert row["1002", "email_address"] == "maria.gonzalez@gmail.com"  # gmial.com fixed
    assert row["1005", "email_address"] == "carlos.ruiz@yahoo.com"  # yahoo.con fixed
    assert row["1005", "balance_usd"] == "1000.0"  # '1.000' next to '2.500,50'
    assert row["1005", "phone"] == "+573105551234"  # Colombia row
    assert row["1003", "full_name"] == "O'Brien Smith"
    assert row["1003", "address"] == "PH Ocean, Apartamento 12B"
    assert row["1010", "phone"] == "+5072001234"  # 7-digit Panama landline
    assert row["1002", "signup_date"] == "2023-02-05"  # 05/02/2023, day-first column
    assert row["1006", "signup_date"] == "2023-01-15"  # Jan 15 2023
    assert "1008" not in cleaned.index  # id-only record dropped
    assert all(re.fullmatch(r"\d+", a) for a in cleaned["age"])  # whole numbers
    assert set(cleaned["country"]) == {"Panama", "Colombia", "Mexico", "Costa Rica", "USA", "Ecuador"}
    assert cleaned["phone"].str.fullmatch(r"\+\d{8,14}").all()


def test_validation_fails_only_on_real_problems(cleaned):
    report = validate(cleaned.reset_index(drop=True), load_rules(ROOT / "rules.yaml"))
    assert report.failures_by_rule() == {"full_name.not_null": 1, "email_address.regex": 2}
    assert sorted(str(f.value) for f in report.failures if f.rule == "email_address.regex") == [
        "ana_lopezexample.com",
        "valentina.rojasgmail.com",
    ]


def test_output_hash_is_stable_across_processes(tmp_path):
    hashes = []
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLEANKIT_")}
    for i in range(2):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "cli.py"), "run-pipeline", str(ROOT / "data" / "messy_sample.csv"),
             "--steps", str(ROOT / "steps.yaml"), "--out", str(tmp_path / f"out{i}.csv")],
            capture_output=True, env=env, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        m = re.search(r"output hash ([0-9a-f]{16})", proc.stdout.decode("utf-8"))
        hashes.append(m.group(1))
    assert hashes[0] == hashes[1]
    assert (tmp_path / "out0.csv").read_bytes() == (tmp_path / "out1.csv").read_bytes()
