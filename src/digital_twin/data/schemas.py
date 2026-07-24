from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


OBSERVATION_FIELDS = (
    "participant_id", "source_dataset", "source_record_id", "cycle_id", "signal_name",
    "value", "unit", "event_time", "measurement_time", "report_time", "availability_time",
    "device", "assay", "quality_flag", "missingness_reason", "raw_column",
    "transformation_version",
)
EVENT_FIELDS = (
    "participant_id", "cycle_id", "event_type", "event_time_lower", "event_time_upper",
    "event_source", "certainty", "availability_time",
)
PARTICIPANT_FIELDS = (
    "participant_id", "source_dataset", "age", "age_unit", "reproductive_stage",
    "contraceptive_status", "medication_status", "device_type", "timezone",
    "enrollment_date", "withdrawal_date", "available_covariates",
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    rule: str
    message: str
    rows: int = 0


def _required(df: pd.DataFrame, fields: tuple[str, ...]) -> list[ValidationIssue]:
    missing = sorted(set(fields) - set(df.columns))
    return [] if not missing else [ValidationIssue("error", "required_columns", f"Missing columns: {missing}")]


def validate_observations(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = _required(df, OBSERVATION_FIELDS)
    if issues:
        return issues
    for col in ("event_time", "measurement_time", "report_time", "availability_time"):
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            issues.append(ValidationIssue("error", "valid_timestamps", f"Invalid {col}", bad))
    event = pd.to_datetime(df["event_time"])
    measurement = pd.to_datetime(df["measurement_time"])
    availability = pd.to_datetime(df["availability_time"])
    bad_order = int(((measurement < event) | (availability < measurement)).sum())
    if bad_order:
        issues.append(ValidationIssue("error", "timestamp_order", "Expected event <= measurement <= availability", bad_order))
    missing_value_without_reason = int((df["value"].isna() & df["missingness_reason"].fillna("").eq("")).sum())
    if missing_value_without_reason:
        issues.append(ValidationIssue("error", "missingness_reason", "Missing values require an explicit reason", missing_value_without_reason))
    duplicates = int(df["source_record_id"].duplicated().sum())
    if duplicates:
        issues.append(ValidationIssue("error", "unique_source_record", "Duplicate source_record_id", duplicates))
    directly_identifying = {"name", "email", "phone", "address", "date_of_birth"} & set(df.columns)
    if directly_identifying:
        issues.append(ValidationIssue("error", "data_minimization", f"Direct identifiers are prohibited: {sorted(directly_identifying)}"))
    return issues


def validate_events(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = _required(df, EVENT_FIELDS)
    if issues:
        return issues
    lower = pd.to_datetime(df["event_time_lower"], errors="coerce")
    upper = pd.to_datetime(df["event_time_upper"], errors="coerce")
    invalid = int((lower.isna() | upper.isna() | (lower > upper)).sum())
    if invalid:
        issues.append(ValidationIssue("error", "valid_event_interval", "Invalid event interval", invalid))
    return issues


def validate_participants(df: pd.DataFrame) -> list[ValidationIssue]:
    issues = _required(df, PARTICIPANT_FIELDS)
    if issues:
        return issues
    duplicated = int(df["participant_id"].duplicated().sum())
    if duplicated:
        issues.append(ValidationIssue("error", "unique_participant", "Duplicate participant_id", duplicated))
    return issues


def assert_valid(*issue_groups: list[ValidationIssue]) -> None:
    errors = [issue for group in issue_groups for issue in group if issue.severity == "error"]
    if errors:
        raise ValueError("; ".join(f"{x.rule}: {x.message} ({x.rows})" for x in errors))
