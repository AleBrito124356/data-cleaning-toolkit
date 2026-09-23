# Changelog

## 0.2.0

### Fixed (these silently produced wrong data or wrong verdicts)

- **Cross-field date checks** parsed ISO dates day-first, so `2023-01-05 <=
  2023-02-01` failed and real violations passed. Dates now go through one
  ISO-safe parser shared with the cleaner (`cleankit.dates`); year-first values
  such as `2024/03/05` are never read day-first either. Numeric text is compared
  as numbers (`"10" > "9"`), and literals take the type of the column.
- **Whitespace-only cells** (`"   "`, non-breaking spaces) survived
  `normalize_whitespace` and passed `not_null`. They now become null (and are
  counted in the log), and the validator treats blank strings as null.
- **`handle_missing` fill on integer columns** crashed with `TypeError` when the
  median was fractional (the bundled `steps.yaml` broke after a one-cell
  change). Fill values are converted to the column's type; integer columns get
  a rounded value and the log says so. Boolean, float and date columns are
  handled explicitly.
- **`handle_outliers(action="cap")`** turned `Int64` columns into floats with
  values like `70.25`. Capping now keeps the dtype, using whole-number fences
  for integer columns, and logs the fences.
- **`1.000` next to `2.500,50`** was read as `1.0`. Numbers are parsed with
  column context: a separator's role is inferred from the column's unambiguous
  values, and every ambiguous value is logged with the reading chosen and why.
  `parse_number` gains a `decimal` hint and no longer turns `"12 years"` into 12.
- **`coerce_types` to integer** rounded `2.5` silently. Fractions are counted in
  the log (`rounded`, `fraction_examples`); `on_fraction: round | nullify |
  error` decides what happens. Rounding is now half away from zero (2.5 -> 3).
- **Phones were always "valid"** for any 6-12 digits. Validity follows each
  country's numbering plan (length and leading digits; Panama keeps 7-digit
  landlines and 8-digit mobiles starting with 6), and an unknown region raises
  instead of silently using Panama.
- **`clean_address`** doubled the dot in `No.`, dropped the comma after `PH`,
  lower-cased unit codes (`12B`) and acronyms (`CDMX`).
- **`standardize_column_names`** could still produce duplicates
  (`Name, name, name_1`); names are now guaranteed unique.
- **Pipeline files** were only checked for op names: a misspelled parameter
  crashed with a traceback and a misspelled option (`strategy: fil`) silently
  fell back to other behaviour. Steps are now validated against each method's
  signature and option vocabulary, with did-you-mean hints; the same checks run
  for configs passed to `run_pipeline` directly.
- **The CLI crashed on Windows** when output was redirected and the data held
  characters outside cp1252. Output is UTF-8; `TypeError`, `KeyError`, YAML and
  CSV parse errors become one-line `error:` messages with exit code 2.
- `normalize_whitespace` over-counted changed cells (NaN never equals NaN).
- `standardize_categoricals` logged only the first raw spelling per key.
- `pandas>=2.1` is now the declared minimum (the code uses `DataFrame.map`).

### Added

- Contact standardizers as audited operations: `standardize_emails`,
  `standardize_phones` (per-row region from a country column, default-region
  and "international without +" fallbacks, all logged), `standardize_names`,
  `clean_addresses`; `invalid: keep | null | flag` for emails and phones.
- `region_for_country` maps free-text countries (`Rep. de Panamá`, `méxico`,
  `EEUU`, one-typo `Columbia`) to phone regions.
- `cleankit suggest` / `cleankit.suggest()`: a commented starter `steps.yaml`
  and `rules.yaml` generated from the profile.
- Profiler: date-format fingerprints (`Mon DD YYYY`, `YYYY/MM/DD`, ...),
  day-first evidence, separators on pandas 3 string columns, mixed decimal
  marks, ambiguous separators, boolean spellings, spelling-variant clusters,
  semantic roles (identifier, person name, address, country), phone detection,
  repeated keys with differing rows, name casing, address abbreviations.
  JSON output (`--format json`, `ProfileReport.from_dict`), escaped Markdown
  tables, and an HTML report with light/dark themes and a phone layout.
- Deduplication: exact-bound pruning makes fuzzy dedup 150x faster on 5,000
  rows with identical results; `block_on`; `keep: most_complete`; every
  removal logged as a pair with its score; `duplicate_of` in flag mode;
  `--review-out` CSV of pairs.
- Validation failures carry the CSV line number (multi-line quoted fields
  included) in text, Markdown and JSON reports.
- `.env` is actually read (it was documented but ignored): a built-in parser
  for `CLEANKIT_DEFAULT_REGION` and `CLEANKIT_CSV_SEP`; environment variables
  win over the file and flags win over both. `--region` now changes the output
  of `clean` (it was only written to the audit JSON).
- `cleankit clean` is profile-driven: inferred types, merged category
  spellings, contact standardization, and new `--keep`, `--dedupe-on`,
  `--fuzzy-keys`, `--block-on`, `--threshold`, `--review-out`,
  `--no-contacts`, `--no-categoricals` options.
- `handle_missing(strategy="drop_rows", how="all")`.
- `scripts/bench_dedup.py`; tests grew from 63 to 188 (CLI, golden demo,
  suggest end to end, dedup equivalence, every audited bug).

### Changed

- The bundled `steps.yaml` uses the contact operations and `keep:
  most_complete`, and drops the record that holds nothing but an id;
  `rules.yaml` checks phones against E.164 and ages as integers. Validation of
  the demo now fails on the two malformed emails and on customer 1014's blank
  name, which the whitespace bug used to hide.
- The `Cleaner` keeps the input's row labels while it works (so audit row
  numbers always refer to input rows); `run_pipeline` still returns a frame
  numbered `0..n-1`. A frame with a non-unique index is renumbered on entry.
- `clean` drops exact duplicate rows by default as before, but its audit JSON
  now also lists the steps it ran.
- `normalize_text` returns `None` for whitespace-only strings.
