"""Declarative, reproducible cleaning pipelines.

A pipeline is a YAML file listing ordered steps; each step names a
:class:`~cleankit.clean.Cleaner` operation and its parameters. Running the
pipeline on a DataFrame produces a :class:`PipelineResult` with a before/after
summary, the full audit log, and content hashes so a run can be verified as
reproducible: the same input plus the same steps yields the same output hash.

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
      - op: handle_missing
        strategy: fill
        columns: [age]
        value: median
      - op: deduplicate
        subset: [email]
      - op: handle_outliers
        columns: [balance]
        method: iqr
        action: cap
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .clean import Cleaner

# Operations that a pipeline step may invoke, mapped to the Cleaner method.
_ALLOWED_OPS = {
    "standardize_column_names",
    "normalize_whitespace",
    "coerce_types",
    "standardize_categoricals",
    "handle_missing",
    "deduplicate",
    "handle_outliers",
}


@dataclass
class PipelineResult:
    name: str
    before: dict[str, Any]
    after: dict[str, Any]
    audit_log: list[dict[str, Any]]
    input_hash: str
    output_hash: str
    df: pd.DataFrame = field(repr=False, default=None)  # type: ignore[assignment]
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
        lines.append(
            f"| Rows | {self.before['rows']} | {self.after['rows']} |"
        )
        lines.append(
            f"| Columns | {self.before['cols']} | {self.after['cols']} |"
        )
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
        lines.append("")
        return "\n".join(lines)


def load_pipeline(path: str | Path) -> dict[str, Any]:
    """Load and validate a pipeline definition file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("pipeline file must be a mapping at the top level")
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("pipeline 'steps' must be a non-empty list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "op" not in step:
            raise ValueError(f"step {i} must be a mapping with an 'op' key")
        if step["op"] not in _ALLOWED_OPS:
            raise ValueError(
                f"step {i}: unknown op '{step['op']}'. "
                f"Allowed: {sorted(_ALLOWED_OPS)}"
            )
    data.setdefault("name", "pipeline")
    return data


def run_pipeline(df: pd.DataFrame, config: dict[str, Any]) -> PipelineResult:
    """Execute ``config['steps']`` against ``df`` and return a result object."""
    name = str(config.get("name", "pipeline"))
    before = _summary(df)
    input_hash = frame_hash(df)

    cleaner = Cleaner(df)
    for step in config["steps"]:
        op = step["op"]
        params = {k: v for k, v in step.items() if k != "op"}
        method = getattr(cleaner, op)
        method(**params)

    out_df = cleaner.df
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
