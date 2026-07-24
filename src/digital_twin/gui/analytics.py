from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from digital_twin.config import (
    DataSection,
    ExperimentConfig,
    ExperimentSection,
    MissingnessSection,
    validate_config,
)
from digital_twin.data.schemas import OBSERVATION_FIELDS, validate_observations
from digital_twin.dynamics.first_passage import (
    dimensionless_groups,
    first_passage_density,
    first_passage_hazard,
    first_passage_moments,
    first_passage_survival,
)
from menstrual_twin.data_validation import validate_daily_observations


@dataclass(frozen=True)
class UploadValidation:
    layout: str
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    rows: int
    participants: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None


def build_research_config(
    *,
    participants: int,
    cycles: int,
    seed: int,
    scenario: str,
    follicular_days: float,
    luteal_days: float,
    progress_noise: float,
    between_person_sd: float,
    missingness_mechanism: str,
    missing_probability: float,
) -> ExperimentConfig:
    """Build a validated lightweight configuration from GUI controls."""

    base = ExperimentConfig()
    config = replace(
        base,
        experiment=ExperimentSection(
            name="gui_synthetic_lab",
            seed=int(seed),
            output_dir="outputs/gui/synthetic_lab",
        ),
        data=DataSection(
            participants=int(participants),
            cycles_per_participant=int(cycles),
            start_date=base.data.start_date,
            timezone=base.data.timezone,
            scenario=str(scenario),
        ),
        process=replace(
            base.process,
            follicular_days=float(follicular_days),
            luteal_days=float(luteal_days),
            sigma_progress_f=float(progress_noise),
            sigma_progress_l=float(progress_noise * 0.72),
            between_person_log_speed_sd=float(between_person_sd),
            max_cycle_days=max(
                base.process.max_cycle_days,
                int(4 * (follicular_days + luteal_days)),
            ),
        ),
        missingness=MissingnessSection(
            mechanism=str(missingness_mechanism),
            mcar_probability=float(missing_probability),
            mar_probability=float(missing_probability),
            informative_probability=float(missing_probability * 0.75),
            hormone_schedule_days=base.missingness.hormone_schedule_days,
            block_probability=min(float(missing_probability), 0.35),
            block_min_days=base.missingness.block_min_days,
            block_max_days=base.missingness.block_max_days,
            report_delay_probability=base.missingness.report_delay_probability,
            max_report_delay_days=base.missingness.max_report_delay_days,
        ),
    )
    validate_config(config)
    return config


def cycle_duration_table(events: pd.DataFrame) -> pd.DataFrame:
    """Derive complete menstruation-to-menstruation intervals."""

    required = {"participant_id", "event_type", "event_time"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"event table missing columns: {sorted(missing)}")
    menses = events.loc[
        events["event_type"].eq("menstruation_onset"),
        ["participant_id", "cycle_id", "event_time"],
    ].copy()
    menses["event_time"] = pd.to_datetime(menses["event_time"])
    menses = menses.sort_values(["participant_id", "event_time"])
    menses["next_event_time"] = menses.groupby("participant_id")[
        "event_time"
    ].shift(-1)
    menses["cycle_length_days"] = (
        menses["next_event_time"] - menses["event_time"]
    ).dt.total_seconds() / 86400.0
    return menses.loc[
        menses["cycle_length_days"].notna()
        & menses["cycle_length_days"].gt(0)
    ].reset_index(drop=True)


def cohort_summary(
    observed: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, float | int]:
    durations = cycle_duration_table(events)
    if "is_observed" in observed:
        observed_fraction = float(observed["is_observed"].fillna(False).mean())
    else:
        observed_fraction = float(observed["value"].notna().mean())
    return {
        "participants": int(observed["participant_id"].nunique()),
        "complete_cycles": int(len(durations)),
        "median_cycle_days": float(durations["cycle_length_days"].median()),
        "cycle_iqr_days": float(
            durations["cycle_length_days"].quantile(0.75)
            - durations["cycle_length_days"].quantile(0.25)
        ),
        "observed_fraction": observed_fraction,
        "signals": int(observed["signal_name"].nunique()),
    }


def analytic_stage_table(
    *,
    drift: float,
    diffusion: float,
    kappa: float,
    sigma_log_speed: float,
    horizon: float = 60.0,
    points: int = 600,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate analytic first-passage curves and dimensionless controls."""

    time = np.linspace(0.0, float(horizon), int(points))
    moments = first_passage_moments(drift, diffusion)
    groups = dimensionless_groups(
        drift,
        diffusion,
        kappa,
        sigma_log_speed,
    )
    table = pd.DataFrame(
        {
            "time_days": time,
            "density": first_passage_density(time, drift, diffusion),
            "survival": first_passage_survival(time, drift, diffusion),
            "hazard": first_passage_hazard(time, drift, diffusion),
        }
    )
    metrics = {
        "mean_days": moments.mean,
        "sd_days": moments.standard_deviation,
        "cv": moments.coefficient_of_variation,
        "skewness": moments.skewness,
        "peclet": groups.peclet,
        "relaxation_ratio": groups.relaxation_ratio,
        "stationary_log_speed_sd": groups.stationary_log_speed_sd,
    }
    return table, metrics


def missingness_table(observations: pd.DataFrame) -> pd.DataFrame:
    frame = observations.copy()
    observed = (
        frame["is_observed"].fillna(False)
        if "is_observed" in frame
        else frame["value"].notna()
    )
    frame["status"] = np.where(
        observed,
        "observed",
        frame.get(
            "missingness_reason",
            pd.Series("missing", index=frame.index),
        )
        .fillna("missing")
        .replace("", "missing"),
    )
    return (
        frame.groupby(["signal_name", "status"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )


def _date_range(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    for column in candidates:
        if column in frame:
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not dates.empty:
                return dates.min(), dates.max()
    return None, None


def validate_uploaded_frame(frame: pd.DataFrame) -> UploadValidation:
    """Validate either the main long schema or retained wide prototype input."""

    long_layout = set(OBSERVATION_FIELDS).issubset(frame.columns)
    participants = (
        int(frame["participant_id"].nunique())
        if "participant_id" in frame
        else 0
    )
    if long_layout:
        issues = validate_observations(frame)
        errors = tuple(
            f"{issue.rule}: {issue.message} ({issue.rows} rows)"
            for issue in issues
            if issue.severity == "error"
        )
        warnings = tuple(
            f"{issue.rule}: {issue.message} ({issue.rows} rows)"
            for issue in issues
            if issue.severity != "error"
        )
        date_min, date_max = _date_range(
            frame,
            ("event_time", "measurement_time"),
        )
        return UploadValidation(
            layout="common long format",
            valid=not errors,
            errors=errors,
            warnings=warnings,
            rows=len(frame),
            participants=participants,
            date_min=date_min,
            date_max=date_max,
        )

    report = validate_daily_observations(frame)
    date_min, date_max = _date_range(frame, ("timestamp_local",))
    return UploadValidation(
        layout="legacy wide format",
        valid=report.valid,
        errors=tuple(report.errors),
        warnings=tuple(report.warnings),
        rows=len(frame),
        participants=participants,
        date_min=date_min,
        date_max=date_max,
    )


def dataframe_to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def safe_numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame(
            columns=["variable", "count", "mean", "sd", "min", "median", "max"]
        )
    summary = numeric.describe(percentiles=[0.5]).T.reset_index()
    summary = summary.rename(
        columns={
            "index": "variable",
            "std": "sd",
            "50%": "median",
        }
    )
    return summary[
        ["variable", "count", "mean", "sd", "min", "median", "max"]
    ]


def format_config(config: ExperimentConfig) -> dict[str, Any]:
    return config.to_dict()
