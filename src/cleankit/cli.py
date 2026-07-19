"""cleankit command-line interface.

Commands:
    profile        Profile a CSV and write a Markdown or HTML report.
    clean          Run a sensible default cleaning pass and write output + audit.
    validate       Validate a CSV against a rules.yaml (non-zero exit on failure).
    run-pipeline   Run a declarative steps.yaml pipeline and write output + audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import (
    Cleaner,
    __version__,
    load_pipeline,
    load_rules,
    profile_dataframe,
    run_pipeline,
    validate,
)


def _read_csv(path: str, sep: str) -> pd.DataFrame:
    """Read a CSV keeping raw string values so messiness stays visible."""
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        skipinitialspace=False,
    )


def _write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep)
    report = profile_dataframe(df)

    fmt = args.format
    if fmt is None:
        fmt = "html" if (args.out and args.out.lower().endswith(".html")) else "md"

    content = report.to_html() if fmt == "html" else report.to_markdown()
    if args.out:
        Path(args.out).write_text(content, encoding="utf-8")
        print(f"Wrote {fmt} profile to {args.out}")
    else:
        print(report.to_markdown())

    n_issues = sum(len(c.issues) for c in report.columns)
    print(
        f"Profiled {report.n_rows} rows x {report.n_cols} cols "
        f"({report.n_duplicate_rows} duplicate rows, {n_issues} suspected issues).",
        file=sys.stderr,
    )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep)
    region = args.region or os.getenv("CLEANKIT_DEFAULT_REGION", "PA")

    cleaner = Cleaner(df)
    cleaner.standardize_column_names()
    cleaner.normalize_whitespace()
    if not args.no_coerce:
        cleaner.coerce_types()  # inferred per column
    if not args.keep_duplicates:
        cleaner.deduplicate(keep="first")

    out = args.out or _default_out(args.input, "cleaned")
    _write_csv(cleaner.df, out)
    print(f"Wrote cleaned data to {out}")
    for entry in cleaner.log:
        print(f"  - {entry['message']}")

    if args.audit:
        _write_json({"region": region, "audit_log": cleaner.log}, args.audit)
        print(f"Wrote audit log to {args.audit}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep)
    rules = load_rules(args.rules)
    report = validate(df, rules)

    if args.report:
        Path(args.report).write_text(report.to_markdown(), encoding="utf-8")
        print(f"Wrote validation report to {args.report}")
    if args.json:
        _write_json(report.as_dict(), args.json)
        print(f"Wrote validation JSON to {args.json}")

    if report.ok:
        print(f"PASS — {report.n_rules} rule(s), 0 failures over {report.n_rows} rows.")
        return 0

    print(
        f"FAIL — {report.n_failures} failure(s) over {report.n_rows} rows.",
        file=sys.stderr,
    )
    for rule, n in sorted(report.failures_by_rule().items(), key=lambda kv: -kv[1]):
        print(f"  {rule}: {n}", file=sys.stderr)
    for f in report.failures[: args.max_show]:
        loc = "" if f.row is None else f"row {f.row}, "
        print(f"    {loc}{f.column}: {f.message}", file=sys.stderr)
    if report.n_failures > args.max_show:
        print(f"    …and {report.n_failures - args.max_show} more.", file=sys.stderr)
    return 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep)
    config = load_pipeline(args.steps)
    result = run_pipeline(df, config)

    out = args.out or _default_out(args.input, "pipeline")
    _write_csv(result.df, out)
    print(f"Wrote pipeline output to {out}")
    print(
        f"Rows {result.before['rows']} -> {result.after['rows']}, "
        f"missing cells {result.before['missing_cells']} -> "
        f"{result.after['missing_cells']}, "
        f"duplicate rows {result.before['duplicate_rows']} -> "
        f"{result.after['duplicate_rows']}."
    )
    print(f"Input hash {result.input_hash} -> output hash {result.output_hash}")
    for i, entry in enumerate(result.audit_log, 1):
        print(f"  {i}. {entry['op']}: {entry['message']}")

    if args.report:
        Path(args.report).write_text(result.to_markdown(), encoding="utf-8")
        print(f"Wrote run report to {args.report}")
    if args.audit:
        _write_json(result.as_dict(), args.audit)
        print(f"Wrote audit JSON to {args.audit}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_out(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_{suffix}.csv"))


def _write_json(obj, path: str) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleankit",
        description="Profile, clean, validate and run reproducible pipelines "
        "on messy CSVs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"cleankit {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    default_sep = os.getenv("CLEANKIT_CSV_SEP", ",")

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input", help="path to the input CSV")
        sp.add_argument(
            "--sep",
            default=default_sep,
            help="CSV delimiter (default ',' or $CLEANKIT_CSV_SEP)",
        )

    p_profile = sub.add_parser("profile", help="profile a CSV")
    add_common(p_profile)
    p_profile.add_argument("--out", help="output file (.md or .html)")
    p_profile.add_argument(
        "--format", choices=["md", "html"], help="force output format"
    )
    p_profile.set_defaults(func=cmd_profile)

    p_clean = sub.add_parser("clean", help="run a default cleaning pass")
    add_common(p_clean)
    p_clean.add_argument("--out", help="output CSV path")
    p_clean.add_argument("--audit", help="write the audit log as JSON here")
    p_clean.add_argument("--region", help="default phone region (e.g. PA)")
    p_clean.add_argument(
        "--no-coerce", action="store_true", help="skip type coercion"
    )
    p_clean.add_argument(
        "--keep-duplicates", action="store_true", help="do not drop duplicate rows"
    )
    p_clean.set_defaults(func=cmd_clean)

    p_validate = sub.add_parser("validate", help="validate against rules.yaml")
    add_common(p_validate)
    p_validate.add_argument("--rules", required=True, help="path to rules.yaml")
    p_validate.add_argument("--report", help="write a Markdown report here")
    p_validate.add_argument("--json", help="write the report as JSON here")
    p_validate.add_argument(
        "--max-show", type=int, default=20, help="max failures to print"
    )
    p_validate.set_defaults(func=cmd_validate)

    p_pipe = sub.add_parser("run-pipeline", help="run a steps.yaml pipeline")
    add_common(p_pipe)
    p_pipe.add_argument("--steps", required=True, help="path to steps.yaml")
    p_pipe.add_argument("--out", help="output CSV path")
    p_pipe.add_argument("--report", help="write a Markdown run report here")
    p_pipe.add_argument("--audit", help="write the full audit as JSON here")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
