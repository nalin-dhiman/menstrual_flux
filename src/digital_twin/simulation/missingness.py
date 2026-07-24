from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.config import ExperimentConfig


WEARABLE_SIGNALS = {"temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"}
HORMONE_SIGNALS = {"lh", "e3g", "pdg", "estradiol", "progesterone"}
REPORT_SIGNALS = {"bleeding", "symptom_severity", "reporting_availability"}


def apply_missingness(observations: pd.DataFrame, cfg: ExperimentConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Apply explicit missingness without replacing missing values by zero."""
    out = observations.copy()
    m = cfg.missingness
    mechanism = m.mechanism
    out["missingness_reason"] = out["missingness_reason"].astype(object)

    def mark(mask: np.ndarray | pd.Series, reason: str) -> None:
        eligible = np.asarray(mask) & out["is_observed"].to_numpy(bool)
        out.loc[eligible, "value"] = np.nan
        out.loc[eligible, "is_observed"] = False
        out.loc[eligible, "missingness_reason"] = reason

    # Sparse scheduled assays are structural and distinct from skipped tests.
    hormone = out["signal_name"].isin(HORMONE_SIGNALS)
    day_number = out["event_time"].dt.dayofyear
    mark(hormone & ((day_number % m.hormone_schedule_days) != 0), "not_scheduled")

    if mechanism in {"mcar", "mixed"}:
        mark(rng.random(len(out)) < m.mcar_probability, "technical_failure")
    if mechanism in {"mar", "mixed"}:
        late_week = out["event_time"].dt.dayofweek >= 5
        mar_prob = np.where(late_week, m.mar_probability * 1.8, m.mar_probability * 0.5)
        mark(rng.random(len(out)) < mar_prob, "participant_skipped_measurement")
    if mechanism in {"informative", "mixed"}:
        symptom = out.groupby(["participant_id", "event_time"])["value"].transform(
            lambda x: np.nanmax(x.to_numpy(dtype=float)) if np.any(np.isfinite(x.to_numpy(dtype=float))) else 0.0
        )
        info_prob = m.informative_probability * (1 + 0.20 * np.clip(symptom, 0, 4))
        mark(rng.random(len(out)) < info_prob, "participant_burden")

    if mechanism == "mixed":
        for participant_id, participant in out.groupby("participant_id"):
            dates = np.sort(participant["event_time"].unique())
            if len(dates) and rng.random() < m.block_probability:
                start_idx = int(rng.integers(0, max(1, len(dates) - m.block_min_days)))
                length = int(rng.integers(m.block_min_days, m.block_max_days + 1))
                block_dates = set(dates[start_idx : start_idx + length])
                block = (out["participant_id"] == participant_id) & out["event_time"].isin(block_dates) & out["signal_name"].isin(WEARABLE_SIGNALS)
                mark(block, "device_not_worn")

    delayed = out["signal_name"].isin(REPORT_SIGNALS) & (rng.random(len(out)) < m.report_delay_probability)
    delays = rng.integers(1, m.max_report_delay_days + 1, len(out))
    out.loc[delayed, "report_time"] = out.loc[delayed, "report_time"] + pd.to_timedelta(delays[delayed], unit="D")
    out.loc[delayed, "availability_time"] = out.loc[delayed, "report_time"]

    below = out["signal_name"].isin(HORMONE_SIGNALS) & out["is_observed"] & (out["value"] < cfg.observation.assay_detection_limit)
    out.loc[below, "quality_flag"] = "below_detection_limit"
    return out


def available_snapshot(observations: pd.DataFrame, issue_time: pd.Timestamp, observed_only: bool = True) -> pd.DataFrame:
    """Return only information genuinely available at a prospective issue time."""
    issue = pd.Timestamp(issue_time)
    eligible = observations["availability_time"] <= issue
    if observed_only:
        eligible &= observations["is_observed"]
    result = observations[eligible].copy()
    if not result.empty and (result["availability_time"] > issue).any():
        raise AssertionError("future information leaked into forecast snapshot")
    return result
