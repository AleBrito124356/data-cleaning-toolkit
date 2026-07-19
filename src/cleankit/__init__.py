"""cleankit — the unglamorous 80% of data work, done well.

A small, dependency-light pandas toolkit for profiling messy CSVs,
standardizing types and categories, deduplicating, validating against a
rules file, and running reproducible, audited cleaning pipelines.

Public API
----------
Profiling:
    profile_dataframe, ProfileReport
Cleaning:
    Cleaner
Validation:
    validate, load_rules, ValidationReport
Pipeline:
    run_pipeline, load_pipeline, PipelineResult
Contacts:
    standardize_email, to_e164, standardize_name, clean_address
"""

from __future__ import annotations

from .profile import ProfileReport, ColumnProfile, profile_dataframe
from .clean import Cleaner, to_snake_case, parse_number, parse_boolean
from .validate import (
    ValidationReport,
    ValidationFailure,
    load_rules,
    validate,
)
from .pipeline import PipelineResult, load_pipeline, run_pipeline
from .contacts import (
    standardize_email,
    to_e164,
    standardize_name,
    clean_address,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # profile
    "ProfileReport",
    "ColumnProfile",
    "profile_dataframe",
    # clean
    "Cleaner",
    "to_snake_case",
    "parse_number",
    "parse_boolean",
    # validate
    "ValidationReport",
    "ValidationFailure",
    "load_rules",
    "validate",
    # pipeline
    "PipelineResult",
    "load_pipeline",
    "run_pipeline",
    # contacts
    "standardize_email",
    "to_e164",
    "standardize_name",
    "clean_address",
]
