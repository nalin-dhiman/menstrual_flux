from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "participant_id",
    "day_in_study",
    "timestamp_local",
    "bleeding",
}

ALLOWED_NUMERIC_RANGES = {
    "temperature_c": (30.0, 42.0),
    "rhr_bpm": (25.0, 220.0),
    "sleep_hours": (0.0, 24.0),
    "lh": (0.0, 500.0),
    "e3g": (0.0, 2000.0),
    "pdg": (0.0, 200.0),
}


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str]
    warnings: list[str]


def validate_daily_observations(df: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return ValidationReport(False, errors, warnings)

    if df[["participant_id", "day_in_study"]].duplicated().any():
        errors.append("Duplicate participant_id/day_in_study rows detected.")
    if (pd.to_numeric(df["day_in_study"], errors="coerce") <= 0).any():
        errors.append("day_in_study must be positive.")

    for col, (lo, hi) in ALLOWED_NUMERIC_RANGES.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = vals.notna() & ((vals < lo) | (vals > hi))
            if bad.any():
                warnings.append(f"{col}: {int(bad.sum())} values outside broad plausibility range [{lo}, {hi}].")

    # Missing data must remain missing; zero is not a generic sentinel.
    for col in ["temperature_c", "rhr_bpm", "sleep_hours"]:
        if col in df.columns and (pd.to_numeric(df[col], errors="coerce") == 0).any():
            warnings.append(f"{col} contains zero(s); verify these are not missing-value sentinels.")

    for pid, g in df.groupby("participant_id"):
        d = pd.to_numeric(g["day_in_study"], errors="coerce").sort_values().to_numpy()
        if d.size > 1 and np.any(np.diff(d) <= 0):
            errors.append(f"Non-increasing day_in_study for participant {pid}.")

    return ValidationReport(not errors, errors, warnings)


def validate_csv(path: str | Path) -> ValidationReport:
    return validate_daily_observations(pd.read_csv(path))
