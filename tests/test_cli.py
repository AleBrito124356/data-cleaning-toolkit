"""The command-line interface: exit codes, outputs, errors, encodings, .env."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cleankit.cli import _record_lines, main
from cleankit.config import load_env_file, parse_env_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = str(ROOT / "data" / "messy_sample.csv")
STEPS = str(ROOT / "steps.yaml")
RULES = str(ROOT / "rules.yaml")


_KEYS = ("CLEANKIT_DEFAULT_REGION", "CLEANKIT_CSV_SEP", "CLEANKIT_DEBUG")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Run every test in an empty directory with no CLEANKIT_* variables.

    main() loads .env into os.environ, so the keys are restored by hand:
    monkeypatch.delenv does not record keys that were absent to begin with.
    """
    monkeypatch.chdir(tmp_path)
    saved = {k: os.environ.pop(k, None) for k in _KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the CLI in a real subprocess with stdout/stderr piped (not a console)."""
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("CLEANKIT_")}
    full_env.pop("PYTHONIOENCODING", None)
    full_env.pop("PYTHONUTF8", None)
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), *args],
        capture_output=True,
        env=full_env,
        timeout=120,
    )


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "cleankit 0.2.0" in capsys.readouterr().out


def test_profile_writes_md_html_json(tmp_path, capsys):
    assert main(["profile", SAMPLE, "--out", "p.md"]) == 0
    assert main(["profile", SAMPLE, "--out", "p.html"]) == 0
    assert main(["profile", SAMPLE, "--out", "p.json"]) == 0
    assert (tmp_path / "p.md").read_text(encoding="utf-8").startswith("# Data profile")
    assert "<!doctype html>" in (tmp_path / "p.html").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert data["n_rows"] == 20 and len(data["columns"]) == 11
    err = capsys.readouterr().err
    assert "Profiled 20 rows x 11 cols (1 duplicate rows, 24 suspected issues)." in err


def test_run_pipeline_then_validate_exit_codes(tmp_path):
    assert main(["run-pipeline", SAMPLE, "--steps", STEPS, "--out", "cleaned.csv",
                 "--audit", "audit.json", "--report", "run.md", "--review-out", "pairs.csv"]) == 0
    assert main(["validate", "cleaned.csv", "--rules", RULES, "--json", "v.json", "--report", "v.md"]) == 1
    report = json.loads((tmp_path / "v.json").read_text(encoding="utf-8"))
    assert report["failures_by_rule"] == {"email_address.regex": 2, "full_name.not_null": 1}
    lines = {f["value"]: f["line"] for f in report["failures"] if f["value"]}
    assert lines == {"ana_lopezexample.com": 5, "valentina.rojasgmail.com": 17}
    assert "| Row | Line |" in (tmp_path / "v.md").read_text(encoding="utf-8")
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    dedupe = audit["audit_log"][-1]
    assert dedupe["op"] == "deduplicate" and len(dedupe["pairs"]) == 3
    pairs = pd.read_csv(tmp_path / "pairs.csv", dtype=str)
    assert pairs["role"].tolist() == ["duplicate", "kept"] * 3
    assert pairs.loc[0, "line"] == "5" and pairs.loc[1, "line"] == "2"


def test_validate_passes_with_exit_zero(tmp_path, capsys):
    (tmp_path / "ok.csv").write_text("id,age\n1,30\n2,40\n", encoding="utf-8")
    (tmp_path / "r.yaml").write_text(
        "columns:\n  id: {unique: true}\n  age: {range: {min: 0, max: 120}}\n", encoding="utf-8"
    )
    assert main(["validate", "ok.csv", "--rules", "r.yaml"]) == 0
    assert capsys.readouterr().out.startswith("PASS")


def test_error_paths_exit_2(tmp_path, capsys):
    assert main(["validate", "missing.csv", "--rules", RULES]) == 2
    assert "file not found: missing.csv" in capsys.readouterr().err
    (tmp_path / "bad.yaml").write_text("steps:\n  - op: handle_missing\n    colums: [a]\n", encoding="utf-8")
    assert main(["run-pipeline", SAMPLE, "--steps", "bad.yaml"]) == 2
    assert "Did you mean 'columns'" in capsys.readouterr().err
    (tmp_path / "broken.yaml").write_text("steps: [\n", encoding="utf-8")
    assert main(["run-pipeline", SAMPLE, "--steps", "broken.yaml"]) == 2
    assert "invalid YAML" in capsys.readouterr().err
    assert main(["clean", SAMPLE, "--region", "ZZ"]) == 2
    assert "unknown phone region 'ZZ'" in capsys.readouterr().err
    assert main(["--env-file", "nope.env", "profile", SAMPLE]) == 2


def test_debug_flag_shows_the_traceback(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--debug", "validate", "missing.csv", "--rules", RULES])


def test_clean_region_changes_phone_output(tmp_path):
    (tmp_path / "p.csv").write_text("Name,Phone\nana,55 1234 5678\nbo,6123-4567\n", encoding="utf-8")
    assert main(["clean", "p.csv", "--region", "MX", "--out", "mx.csv"]) == 0
    assert main(["clean", "p.csv", "--region", "PA", "--out", "pa.csv"]) == 0
    mx = pd.read_csv(tmp_path / "mx.csv", dtype=str)["phone"].tolist()
    pa = pd.read_csv(tmp_path / "pa.csv", dtype=str)["phone"].tolist()
    assert mx == ["+525512345678", "6123-4567"]
    assert pa == ["55 1234 5678", "+50761234567"]


def test_clean_on_the_demo(tmp_path):
    assert main(["clean", SAMPLE, "--out", "c.csv", "--audit", "a.json",
                 "--dedupe-on", "customer_id", "--keep", "most_complete"]) == 0
    out = pd.read_csv(tmp_path / "c.csv", dtype=str, keep_default_na=False)
    assert len(out) == 17
    assert out.set_index("customer_id").loc["1005", "phone"] == "+573105551234"
    audit = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert audit["region"] == "PA"
    assert [s["op"] for s in audit["steps"]][-1] == "deduplicate"


def test_dotenv_defaults_and_precedence(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# defaults\nexport CLEANKIT_DEFAULT_REGION=MX  # Mexico\nCLEANKIT_CSV_SEP=\";\"\n",
        encoding="utf-8",
    )
    (tmp_path / "s.csv").write_text("Name;Phone\nana;55 1234 5678\n", encoding="utf-8")
    assert main(["clean", "s.csv", "--out", "e.csv"]) == 0
    assert pd.read_csv(tmp_path / "e.csv", dtype=str)["phone"].tolist() == ["+525512345678"]
    # A real environment variable wins over the file.
    monkeypatch.setenv("CLEANKIT_DEFAULT_REGION", "PA")
    assert main(["clean", "s.csv", "--out", "e2.csv"]) == 0
    assert pd.read_csv(tmp_path / "e2.csv", dtype=str)["phone"].tolist() == ["55 1234 5678"]
    # And a flag wins over both.
    assert main(["clean", "s.csv", "--region", "MX", "--out", "e3.csv"]) == 0
    assert pd.read_csv(tmp_path / "e3.csv", dtype=str)["phone"].tolist() == ["+525512345678"]


def test_env_parser_and_loader(tmp_path, monkeypatch):
    parsed = parse_env_text("A=1\n# c\nexport B = 'two words'\nC=x # note\nbroken\n")
    assert parsed == {"A": "1", "B": "two words", "C": "x"}
    path = tmp_path / "x.env"
    path.write_text("CLEANKIT_DEFAULT_REGION=CO\nOTHER=1\n", encoding="utf-8")
    assert load_env_file(path) == {"CLEANKIT_DEFAULT_REGION": "CO"}
    assert os.environ["CLEANKIT_DEFAULT_REGION"] == "CO"
    assert "OTHER" not in os.environ
    assert load_env_file(tmp_path / "absent.env") == {}


def test_suggest_writes_files_and_refuses_to_overwrite(tmp_path, capsys):
    assert main(["suggest", SAMPLE, "--steps-out", "s.yaml", "--rules-out", "r.yaml"]) == 0
    assert "Wrote 11 step(s) to s.yaml" in capsys.readouterr().out
    assert main(["suggest", SAMPLE, "--steps-out", "s.yaml"]) == 2
    assert "--force" in capsys.readouterr().err
    assert main(["suggest", SAMPLE, "--steps-out", "s.yaml", "--force"]) == 0
    assert main(["run-pipeline", SAMPLE, "--steps", "s.yaml", "--out", "o.csv"]) == 0
    assert main(["validate", "o.csv", "--rules", "r.yaml"]) == 1


def test_record_lines_follow_multiline_fields(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text('id,note\n1,"line one\nline two"\n\n2,plain\n', encoding="utf-8")
    df = pd.read_csv(path, dtype=str)
    assert _record_lines(str(path), ",", len(df)) == [2, 5]


def test_piped_output_is_utf8_even_on_windows(tmp_path):
    csv_path = tmp_path / "uni.csv"
    csv_path.write_text("id,name\n1,Zhāng 张伟 ✓\n2,Ana\n", encoding="utf-8")
    proc = _run("profile", str(csv_path))
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert "Zhāng 张伟 ✓" in proc.stdout.decode("utf-8")


def test_subprocess_validate_exit_code_and_encoding(tmp_path):
    out = tmp_path / "cleaned.csv"
    assert _run("run-pipeline", SAMPLE, "--steps", STEPS, "--out", str(out)).returncode == 0
    proc = _run("validate", str(out), "--rules", RULES)
    assert proc.returncode == 1
    err = proc.stderr.decode("utf-8")
    assert err.startswith("FAIL — 3 failure(s) over 16 rows.")
    assert "row 3 (line 5), email_address: does not match pattern ('ana_lopezexample.com')" in err


def test_clean_fuzzy_dedupe_writes_a_review_file(tmp_path, capsys):
    assert main(["clean", SAMPLE, "--out", "c.csv", "--fuzzy-keys", "full_name,phone",
                 "--threshold", "0.9", "--review-out", "review.csv"]) == 0
    review = pd.read_csv(tmp_path / "review.csv", dtype=str, keep_default_na=False)
    fuzzy = review[review["match"] == "fuzzy"]
    assert len(fuzzy) >= 2  # 1002 'María José González' vs 'Maria Jose Gonzalez', etc.
    assert set(review["role"]) == {"duplicate", "kept"}
    assert "duplicate pair(s) to review.csv" in capsys.readouterr().out
