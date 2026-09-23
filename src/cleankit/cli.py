"""cleankit command-line interface.

Commands:
    profile        Profile a CSV and write a Markdown, HTML or JSON report.
    suggest        Generate a starter steps.yaml and rules.yaml from the profile.
    clean          Run a profile-driven default cleaning pass and write output + audit.
    validate       Validate a CSV against a rules.yaml (exit 1 when a rule fails).
    run-pipeline   Run a declarative steps.yaml pipeline and write output + audit.

Exit codes: 0 success, 1 validation failed, 2 usage or input error.
Defaults for ``--sep`` and ``--region`` come from ``CLEANKIT_CSV_SEP`` and
``CLEANKIT_DEFAULT_REGION``, read from the environment or from ``.env``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from . import (
    __version__,
    load_pipeline,
    load_rules,
    profile_dataframe,
    run_pipeline,
    validate,
)
from .config import default_region, default_sep, load_env_file

_CONTACT_OPS = {"standardize_emails", "standardize_phones", "standardize_names", "clean_addresses"}


def _utf8_streams() -> None:
    """Never crash on names like 'Zhāng 张伟' when stdout is a pipe or a file.

    On Windows a redirected stdout defaults to the ANSI code page (cp1252),
    which cannot encode most of the world's names.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, io.UnsupportedOperation):
            pass


def _read_csv(path: str, sep: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """Read a CSV keeping raw string values so messiness stays visible."""
    return pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        skipinitialspace=False,
        encoding=encoding,
    )


def _record_lines(path: str, sep: str, n_rows: int, encoding: str = "utf-8-sig") -> list[int]:
    """The 1-based file line where each data record starts (header-aware).

    Quoted fields may span several lines, so this walks the file with the csv
    module instead of assuming ``row + 2``; it falls back to that assumption
    when the delimiter is not a single character or the counts disagree.
    """
    fallback = [i + 2 for i in range(n_rows)]
    if len(sep) != 1:
        return fallback
    starts: list[int] = []
    try:
        with open(path, newline="", encoding=encoding) as fh:
            reader = csv.reader(fh, delimiter=sep)
            prev_end = 0
            for i, row in enumerate(reader):
                start = prev_end + 1
                prev_end = reader.line_num
                if i == 0 or not row:
                    continue
                starts.append(start)
    except (OSError, csv.Error, UnicodeDecodeError):
        return fallback
    return starts if len(starts) == n_rows else fallback


def _write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def _write_json(obj: Any, path: str) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_review(pairs: list[dict[str, Any]], path: str, line_of: Callable[[Any], Any]) -> None:
    """Duplicate pairs in long format: one 'duplicate' and one 'kept' line per pair."""
    cols: list[str] = []
    for p in pairs:
        for c in list(p["values"]) + list(p["kept_values"]):
            if c not in cols:
                cols.append(c)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "role", "row", "line", "match", "score", *cols])
        for i, p in enumerate(pairs, 1):
            for role, row, values in (
                ("duplicate", p["row"], p["values"]),
                ("kept", p["kept_row"], p["kept_values"]),
            ):
                w.writerow(
                    [i, role, row, line_of(row), p["match"], p["score"]]
                    + ["" if values.get(c) is None else values.get(c) for c in cols]
                )


def _line_lookup(lines: list[int]) -> Callable[[Any], Any]:
    def line_of(row: Any) -> Any:
        if isinstance(row, int) and 0 <= row < len(lines):
            return lines[row]
        return ""

    return line_of


def _refuse_overwrite(paths: list[str | None], force: bool) -> None:
    existing = [p for p in paths if p and Path(p).exists()]
    if existing and not force:
        raise ValueError(
            f"{', '.join(existing)} already exist(s); pass --force to overwrite"
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep, args.encoding)
    report = profile_dataframe(df)

    fmt = args.format
    if fmt is None:
        suffix = Path(args.out).suffix.lower() if args.out else ""
        fmt = {".html": "html", ".htm": "html", ".json": "json"}.get(suffix, "md")

    render = {"html": report.to_html, "json": report.to_json, "md": report.to_markdown}[fmt]
    if args.out:
        Path(args.out).write_text(render(), encoding="utf-8")
        print(f"Wrote {fmt} profile to {args.out}")
    else:
        print(render() if fmt in ("json", "md") else report.to_markdown())

    print(
        f"Profiled {report.n_rows} rows x {report.n_cols} cols "
        f"({report.n_duplicate_rows} duplicate rows, {report.n_issues} suspected issues).",
        file=sys.stderr,
    )
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    from .suggest import suggest

    df = _read_csv(args.input, args.sep, args.encoding)
    sugg = suggest(df, default_region=args.region, name=args.name, source=args.input)
    steps_text, rules_text = sugg.steps_yaml(), sugg.rules_yaml()
    # The files must load exactly as generated; fail loudly rather than write a broken file.
    if yaml.safe_load(steps_text) != sugg.pipeline or yaml.safe_load(rules_text) != sugg.rules:
        raise RuntimeError("internal error: generated YAML does not round-trip; please report it")

    if not args.steps_out and not args.rules_out:
        print(steps_text)
        print("---")
        print(rules_text)
    else:
        _refuse_overwrite([args.steps_out, args.rules_out], args.force)
        if args.steps_out:
            Path(args.steps_out).write_text(steps_text, encoding="utf-8")
            print(f"Wrote {len(sugg.pipeline['steps'])} step(s) to {args.steps_out}")
        if args.rules_out:
            Path(args.rules_out).write_text(rules_text, encoding="utf-8")
            n_rules = sum(len(v) for v in sugg.rules["columns"].values()) + len(sugg.rules["checks"])
            print(f"Wrote {n_rules} rule(s) to {args.rules_out}")
    print(
        f"Profiled {sugg.profile.n_rows} rows x {sugg.profile.n_cols} cols; "
        f"{sugg.profile.n_issues} suspected issues.",
        file=sys.stderr,
    )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    from .suggest import suggest

    df = _read_csv(args.input, args.sep, args.encoding)
    region = args.region
    sugg = suggest(df, default_region=region, name="clean", source=args.input)

    steps = []
    for step in sugg.core_pipeline()["steps"]:
        op = step["op"]
        if op == "deduplicate":
            continue  # rebuilt below from the command's own options
        if args.no_coerce and op == "coerce_types":
            continue
        if args.no_contacts and op in _CONTACT_OPS:
            continue
        if args.no_categoricals and op == "standardize_categoricals":
            continue
        steps.append(step)
    if not args.keep_duplicates:
        dedupe: dict[str, Any] = {"op": "deduplicate", "keep": args.keep}
        if args.dedupe_on:
            dedupe["subset"] = [c.strip() for c in args.dedupe_on.split(",") if c.strip()]
        if args.fuzzy_keys:
            dedupe["fuzzy"] = True
            dedupe["fuzzy_keys"] = [c.strip() for c in args.fuzzy_keys.split(",") if c.strip()]
            dedupe["threshold"] = args.threshold
            if args.block_on:
                dedupe["block_on"] = [c.strip() for c in args.block_on.split(",") if c.strip()]
        steps.append(dedupe)

    result = run_pipeline(df, {"name": "clean", "steps": steps})
    out = args.out or _default_out(args.input, "cleaned")
    _write_csv(result.df, out)
    print(f"Wrote cleaned data to {out}")
    print(
        f"Rows {result.before['rows']} -> {result.after['rows']}, "
        f"input hash {result.input_hash} -> output hash {result.output_hash}"
    )
    for entry in result.audit_log:
        print(f"  - {entry['message']}")

    lines = _record_lines(args.input, args.sep, len(df), args.encoding)
    if args.audit:
        _write_json({"region": region, "steps": steps, **result.as_dict()}, args.audit)
        print(f"Wrote audit log to {args.audit}")
    if args.review_out:
        _write_review(result.review, args.review_out, _line_lookup(lines))
        print(f"Wrote {len(result.review)} duplicate pair(s) to {args.review_out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep, args.encoding)
    rules = load_rules(args.rules)
    lines = _record_lines(args.input, args.sep, len(df), args.encoding)
    report = validate(df, rules, line_numbers=lines)

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
        loc = f.location()
        loc = f"{loc}, " if loc else ""
        value = "" if f.value is None or (not isinstance(f.value, str) and pd.isna(f.value)) else f" ({str(f.value)[:40]!r})"
        print(f"    {loc}{f.column}: {f.message}{value}", file=sys.stderr)
    if report.n_failures > args.max_show:
        print(f"    …and {report.n_failures - args.max_show} more.", file=sys.stderr)
    return 1


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    df = _read_csv(args.input, args.sep, args.encoding)
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
    if args.review_out:
        lines = _record_lines(args.input, args.sep, len(df), args.encoding)
        _write_review(result.review, args.review_out, _line_lookup(lines))
        print(f"Wrote {len(result.review)} duplicate pair(s) to {args.review_out}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _default_out(input_path: str, suffix: str) -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_{suffix}.csv"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleankit",
        description="Profile, clean, validate and run reproducible pipelines "
        "on messy CSVs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"cleankit {__version__}"
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="read CLEANKIT_* defaults from this file (default: ./.env if present)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="show a traceback instead of a one-line error"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input", help="path to the input CSV")
        sp.add_argument(
            "--sep",
            default=None,
            help="CSV delimiter (default ',' or $CLEANKIT_CSV_SEP)",
        )
        sp.add_argument(
            "--encoding", default="utf-8-sig", help="input file encoding (default utf-8, BOM-tolerant)"
        )

    def add_region(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--region",
            default=None,
            help="phone region for numbers without a country code, e.g. PA, CO, MX "
            "(default $CLEANKIT_DEFAULT_REGION or PA)",
        )

    p_profile = sub.add_parser("profile", help="profile a CSV")
    add_common(p_profile)
    p_profile.add_argument("--out", help="output file (.md, .html or .json)")
    p_profile.add_argument(
        "--format", choices=["md", "html", "json"], help="force output format"
    )
    p_profile.set_defaults(func=cmd_profile)

    p_suggest = sub.add_parser(
        "suggest", help="generate a starter steps.yaml and rules.yaml from the profile"
    )
    add_common(p_suggest)
    add_region(p_suggest)
    p_suggest.add_argument("--steps-out", help="write the suggested pipeline here")
    p_suggest.add_argument("--rules-out", help="write the suggested rules here")
    p_suggest.add_argument("--name", default="suggested", help="pipeline name")
    p_suggest.add_argument("--force", action="store_true", help="overwrite existing files")
    p_suggest.set_defaults(func=cmd_suggest)

    p_clean = sub.add_parser("clean", help="run a profile-driven default cleaning pass")
    add_common(p_clean)
    add_region(p_clean)
    p_clean.add_argument("--out", help="output CSV path")
    p_clean.add_argument("--audit", help="write the audit log as JSON here")
    p_clean.add_argument("--review-out", help="write duplicate pairs (CSV) for review here")
    p_clean.add_argument(
        "--no-coerce", action="store_true", help="skip type coercion"
    )
    p_clean.add_argument(
        "--no-contacts", action="store_true", help="skip email/phone/name/address standardization"
    )
    p_clean.add_argument(
        "--no-categoricals", action="store_true", help="do not merge category spelling variants"
    )
    p_clean.add_argument(
        "--keep-duplicates", action="store_true", help="do not drop duplicate rows"
    )
    p_clean.add_argument(
        "--keep", choices=["first", "last", "most_complete"], default="first",
        help="which row of a duplicate group survives (default first)",
    )
    p_clean.add_argument("--dedupe-on", help="comma-separated columns that identify a duplicate")
    p_clean.add_argument("--fuzzy-keys", help="comma-separated columns for fuzzy duplicate matching")
    p_clean.add_argument("--threshold", type=float, default=0.9, help="fuzzy similarity threshold")
    p_clean.add_argument("--block-on", help="only compare fuzzy candidates sharing these columns")
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
    p_pipe.add_argument("--review-out", help="write duplicate pairs (CSV) for review here")
    p_pipe.set_defaults(func=cmd_run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    debug = args.debug or os.environ.get("CLEANKIT_DEBUG") == "1"
    try:
        if args.env_file and not Path(args.env_file).is_file():
            raise FileNotFoundError(2, "env file not found", args.env_file)
        load_env_file(args.env_file or ".env")
        if getattr(args, "sep", "x") is None:
            args.sep = default_sep()
        if hasattr(args, "region"):
            from .contacts import _check_region

            args.region = _check_region(args.region or default_region())
        return args.func(args)
    except FileNotFoundError as exc:
        if debug:
            raise
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, yaml.YAMLError, pd.errors.ParserError,
            pd.errors.EmptyDataError, UnicodeDecodeError, PermissionError, IsADirectoryError) as exc:
        if debug:
            raise
        kind = type(exc).__name__
        detail = str(exc).strip() or kind
        if isinstance(exc, KeyError):
            detail = f"missing key {exc}"
        elif isinstance(exc, UnicodeDecodeError):
            detail = f"cannot decode input as {getattr(args, 'encoding', 'utf-8')}; pass --encoding (e.g. latin-1)"
        elif isinstance(exc, yaml.YAMLError):
            detail = f"invalid YAML: {detail}"
        print(f"error: {detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
