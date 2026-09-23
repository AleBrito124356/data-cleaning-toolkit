"""cleankit — the unglamorous 80% of data work, done well.

A small, dependency-light pandas toolkit for profiling messy CSVs,
standardizing types, categories and contact data, deduplicating, validating
against a rules file, and running reproducible, audited cleaning pipelines.

Public API
----------
Profiling:
    profile_dataframe, ProfileReport, ColumnProfile
Suggestions:
    suggest, Suggestion
Cleaning:
    Cleaner, to_snake_case, snake_case_columns, parse_number,
    parse_number_column, parse_boolean, parse_dates
Validation:
    validate, load_rules, ValidationReport, ValidationFailure
Pipeline:
    run_pipeline, load_pipeline, validate_steps, PipelineResult
Contacts:
    standardize_email, to_e164, standardize_name, clean_address,
    region_for_country
Configuration:
    load_env_file
"""

from __future__ import annotations

from .profile import ProfileReport, ColumnProfile, profile_dataframe
from .clean import (
    Cleaner,
    parse_boolean,
    parse_number,
    parse_number_column,
    snake_case_columns,
    to_snake_case,
)
from .dates import parse_dates
from .validate import (
    ValidationReport,
    ValidationFailure,
    load_rules,
    validate,
)
from .pipeline import PipelineResult, load_pipeline, run_pipeline, validate_steps
from .contacts import (
    standardize_email,
    to_e164,
    standardize_name,
    clean_address,
    region_for_country,
)
from .suggest import Suggestion, suggest
from .config import load_env_file

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # profile
    "ProfileReport",
    "ColumnProfile",
    "profile_dataframe",
    # suggest
    "Suggestion",
    "suggest",
    # clean
    "Cleaner",
    "to_snake_case",
    "snake_case_columns",
    "parse_number",
    "parse_number_column",
    "parse_boolean",
    "parse_dates",
    # validate
    "ValidationReport",
    "ValidationFailure",
    "load_rules",
    "validate",
    # pipeline
    "PipelineResult",
    "load_pipeline",
    "run_pipeline",
    "validate_steps",
    # contacts
    "standardize_email",
    "to_e164",
    "standardize_name",
    "clean_address",
    "region_for_country",
    # config
    "load_env_file",
]
