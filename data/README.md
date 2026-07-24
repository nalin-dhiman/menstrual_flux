# Adding data to the project

The files in this directory are schemas and synthetic examples. Do not place identifiable human-participant exports in a shared repository.

## 1. Preserve the raw source first

For every source export, create an immutable raw object and record:

- `source_file_id` and cryptographic checksum;
- source system, device/assay model, firmware/software version, and extraction version;
- extraction timestamp and responsible pipeline version;
- consent/data-use scope and access classification.

Do not manually edit raw files. Corrections belong in versioned transformation code.

## 2. Map into normalized tables

Use `data_dictionary.csv` as the machine-readable contract. The main logical tables are participants, devices, daily observations, raw sensor manifests, bleeding events, hormone tests, symptoms, exposures, medications, reference labels, and data-quality flags.

The included `example_common_observations.csv` is a synthetic long-form example
accepted by the main schema validator. `example_daily_observations.csv` is a
synthetic wide-format example for the retained circular-state prototype.
Neither is a substitute for the full normalized schema in a formal study.

## 3. Timestamp correctly

Retain occurrence time, local clock time, IANA time-zone name, UTC conversion, device-reported time, availability time, and reporting time where relevant. Do not collapse these into one timestamp. Define how overnight wearable windows are assigned to a calendar day and test alternative rules.

## 4. Preserve units and provenance

Store raw value and raw unit as well as canonical value and canonical unit. Record conversion-rule version. For assays, retain platform, lot, limits of detection/quantification, repeat/dilution status, and QC status. Censored values require a censored likelihood; do not replace them with zero or half the detection limit without a prespecified sensitivity analysis.

## 5. Represent missingness explicitly

Use distinct values for not scheduled, participant did not measure, device not worn, technical failure, failed QC, structurally unavailable, and withdrawal/loss to follow-up. Never encode missingness as zero. Preserve reporting delay and revision history.

## 6. Add reference labels as intervals

Use `reference_labels_template.csv`. An ovulation reference is normally an interval or probability distribution, not an exact timestamp. Record lower and upper bounds, reference method, confidence grade, and blinded adjudication metadata. Temperature-only or app-derived labels must not be treated as imaging-confirmed truth.

## 7. Validate before modelling

Run:

```bash
PYTHONPATH=src python -m digital_twin.cli validate-data path/to/common_observations.csv

# Legacy wide-format prototype input
PYTHONPATH=src python scripts/validate_data.py path/to/daily_observations.csv
```

The supplied validator checks only a minimum prototype contract. A production ingestion layer must also enforce table-level schemas, referential integrity, device/assay mappings, timezone rules, duplicate/revision handling, limits of detection, data availability times, and protocol-specific QC.

## 8. Freeze analysis datasets

Generate a versioned curated release containing feature definitions, reference-label intervals, participant/cycle identifiers, split assignments, data-sheet documentation, and a manifest of included raw source hashes. Training, calibration, test, and external cohorts must be frozen before final evaluation.
