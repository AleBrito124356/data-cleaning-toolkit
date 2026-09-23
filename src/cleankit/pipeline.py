"""Declarative, reproducible cleaning pipelines.

A pipeline is a YAML file listing ordered steps; each step names a
:class:`~cleankit.clean.Cleaner` operation and its parameters. Running the
pipeline on a DataFrame produces a :class:`PipelineResult` with a before/after
summary, the full audit log, and content hashes so a run can be verified as
reproducible: the same input plus the same steps yields the same output hash.

Steps are checked before anything runs: unknown operations, misspelled
parameters (``colums``), missing required parameters, and values outside an
option's vocabulary (``strategy: fil``) are rejected with the step number and
a "did you mean" hint instead of a traceback or a silent fallback.

Example ``steps.yaml``::

    name: customers
    steps:
      - op: standardize_column_names
      - op: normalize_whitespace
      - op: coerce_types
        mapping: {signup_date: datetime, age: integer, balance: float}
      - op: standardize_categoricals
        column: country
        canonical: [Panama, Colombia, Mexico, Costa Rica]
      - op: standardize_phones
        columns: [phone]
        default_region: PA
        region_column: country
      - op: handle_missing
        strategy: fill
        columns: [age]
        value: median
      - op: deduplicate
        subset: [customer_id]
        keep: most_complete
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import clean as _clean
from .clean import Cleaner

# Operations that a pipeline step may invoke (each is a Cleaner method).
_ALLOWED_OPS = {
    "standardize_column_names",
    "normalize_whitespace",
    "coerce_types",
    "standardize_categoricals",
    "handle_missing",
    "deduplicate",
    "handle_outliers",
    "standardize_emails",
    "standardize_phones",
    "standardize_names",
    "clean_addresses",
}

# Parameters whose value must come from a fixed vocabulary.
_ENUMS: dict[str, dict[str, tuple[str, ...]]] = {
    "handle_missing": {"strategy": _clean.MISSING_STRATEGIES, "how": _clean.MISSING_HOW},
    "handle_outliers": {"method": _clean.OUTLIER_METHODS, "action": _clean.OUTLIER_ACTIONS},
    "deduplicate": {"keep": _clean.KEEP_OPTIONS},
    "coerce_types": {"on_fraction": _clean.ON_FRACTION},
    "standardize_emails": {"invalid": _clean.INVALID_ACTIONS},
    "standardize_phones": {"invalid": _clean.INVALID_ACTIONS},
}
# Parameters that name columns: a string or a list of strings.
_COLUMN_LISTS = {"columns", "subset", "fuzzy_keys", "block_on"}
_COLUMN_NAMES = {"column", "region_column"}


def _did_you_mean(word: str, options: list[str]) -> str:
    close = difflib.get_close_matches(str(word), options, n=1, cutoff=0.6)
    return f" Did you mean '{close[0]}'?" if close else ""


def _params_of(op: str) -> tuple[list[str], list[str]]:
    """(all parameter names, required parameter names) of a Cleaner method."""
    sig = inspect.signature(getattr(Cleaner, op))
    names, required = [], []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        names.append(name)
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return names, required


def validate_steps(config: dict[str, Any]) -> None:
    """Check every step of a pipeline config; raise ``ValueError`` on the first problem."""
    if not isinstance(config, dict):
        raise ValueError("pipeline file must be a mapping at the top level")
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("pipeline 'steps' must be a non-empty list")
    ops = sorted(_ALLOWED_OPS)
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict) or "op" not in step:
            raise ValueError(f"step {i} must be a mapping with an 'op' key")
        op = step["op"]
        if op not in _ALLOWED_OPS:
            raise ValueError(
                f"step {i}: unknown op '{op}'.{_did_you_mean(op, ops)} "
                f"Allowed: {', '.join(ops)}"
            )
        names, required = _params_of(op)
        params = {k: v for k, v in step.items() if k != "op"}
        for key in params:
            if key not in names:
                raise ValueError(
                    f"step {i} ({op}): unknown parameter '{key}'."
                    f"{_did_you_mean(key, names)} Accepted: {', '.join(names) or 'none'}"
                )
        for key in required:
            if key not in params:
                raise ValueError(f"step {i} ({op}): missing required parameter '{key}'")
        for key, choices in _ENUMS.get(op, {}).items():
            if key in params:
                try:
                    _clean.check_choice(key, params[key], choices)
                except ValueError as exc:
                    raise ValueError(f"step {i} ({op}): {exc}") from None
        for key, value in params.items():
            if key in _COLUMN_LISTS and value is not None:
                ok = isinstance(value, str) or (
                    isinstance(value, list) and all(isinstance(v, str) for v in value)
                )
                if not ok:
                    raise ValueError(
                        f"step {i} ({op}): '{key}' must be a column name or a list "
                        f"of names, got {value!r}"
                    )
            if key in _COLUMN_NAMES and value is not None and not isinstance(value, str):
                raise ValueError(f"step {i} ({op}): '{key}' must be a column name, got {value!r}")
        if op == "coerce_types" and params.get("mapping") is not None:
            mapping = params["mapping"]
            if not isinstance(mapping, dict):
                raise ValueError(f"step {i} (coerce_types): 'mapping' must be column: type pairs")
            targets = sorted(_clean.TYPE_TARGETS)
            for col, target in mapping.items():
                if str(target).lower() not in _clean.TYPE_TARGETS:
                    raise ValueError(
                        f"step {i} (coerce_types): '{col}: {target}' is not a known type."
                        f"{_did_you_mean(str(target).lower(), targets)} "
                        f"Known: integer, float, boolean, datetime, string"
                    )
        if op == "standardize_categoricals" and "extra_aliases" in params:
            if params["extra_aliases"] is not None and not isinstance(params["extra_aliases"], dict):
                raise ValueError(f"step {i} (standardize_categoricals): 'extra_aliases' must map variant: label")


@dataclass
class PipelineResult:
    name: str
    before: dict[str, Any]
    after: dict[str, Any]
    audit_log: list[dict[str, Any]]
    input_hash: str
    output_hash: str
    df: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]
    review: list[dict[str, Any]] = field(repr=False, default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "before": self.before,
            "after": self.after,
            "audit_log": self.audit_log,
        }

    def to_markdown(self) -> str:
        lines = [f"# Pipeline run: {self.name}", ""]
        lines.append(f"- Generated: {self.generated_at}")
        lines.append(f"- Input hash: `{self.input_hash}`")
        lines.append(f"- Output hash: `{self.output_hash}`")
        lines.append("")
        lines.append("## Before / after")
        lines.append("")
        lines.append("| Metric | Before | After |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| Rows | {self.before['rows']} | {self.after['rows']} |")
        lines.append(f"| Columns | {self.before['cols']} | {self.after['cols']} |")
        lines.append(
            f"| Missing cells | {self.before['missing_cells']} "
            f"| {self.after['missing_cells']} |"
        )
        lines.append(
            f"| Duplicate rows | {self.before['duplicate_rows']} "
            f"| {self.after['duplicate_rows']} |"
        )
        lines.append("")
        lines.append("## Audit log")
        lines.append("")
        for i, entry in enumerate(self.audit_log, 1):
            lines.append(f"{i}. **{entry['op']}** — {entry['message']}")
        pairs = [p for p in self.review]
        if pairs:
            lines.append("")
            lines.append("## Duplicate pairs")
            lines.append("")
            lines.append("| Removed row | Kept row | Match | Score |")
            lines.append("| --- | --- | --- | --- |")
            for p in pairs[:50]:
                lines.append(f"| {p['row']} | {p['kept_row']} | {p['match']} | {p['score']} |")
            if len(pairs) > 50:
                lines.append("")
                lines.append(f"…and {len(pairs) - 50} more (see the audit JSON).")
        lines.append("")
        return "\n".join(lines)


def load_pipeline(path: str | Path) -> dict[str, Any]:
    """Load and validate a pipeline definition file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("pipeline file must be a mapping at the top level")
    validate_steps(data)
    data.setdefault("name", "pipeline")
    return data


def run_pipeline(df: pd.DataFrame, config: dict[str, Any]) -> PipelineResult:
    """Execute ``config['steps']`` against ``df`` and return a result object.

    Row numbers in the audit log refer to positions in the input frame; the
    returned ``df`` is renumbered 0..n-1.
    """
    validate_steps(config)
    name = str(config.get("name", "pipeline"))
    before = _summary(df)
    input_hash = frame_hash(df)

    cleaner = Cleaner(df.reset_index(drop=True))
    for step in config["steps"]:
        op = step["op"]
        params = {k: v for k, v in step.items() if k != "op"}
        getattr(cleaner, op)(**params)

    out_df = cleaner.df.reset_index(drop=True)
    after = _summary(out_df)
    output_hash = frame_hash(out_df)

    return PipelineResult(
        name=name,
        before=before,
        after=after,
        audit_log=cleaner.log,
        input_hash=input_hash,
        output_hash=output_hash,
        df=out_df,
        review=cleaner.review,
    )


def _summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def frame_hash(df: pd.DataFrame) -> str:
    """A deterministic content hash of a DataFrame.

    Uses pandas' row hashing plus the column names, so two runs that produce
    identical data (values and column order) produce identical hashes,
    independent of the pandas index.
    """
    row_hashes = pd.util.hash_pandas_object(
        df.reset_index(drop=True), index=False
    ).values
    header = ",".join(str(c) for c in df.columns).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()[:16]
