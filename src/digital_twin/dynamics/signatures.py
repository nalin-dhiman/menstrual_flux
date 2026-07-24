from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew


@dataclass
class DynamicalSignatures:
    summary: pd.DataFrame
    empirical_hazard: pd.DataFrame
    participant_statistics: pd.DataFrame
    stage_durations: pd.DataFrame
    missingness_statistics: pd.DataFrame


def _eligible_cycles(cycles: pd.DataFrame) -> pd.DataFrame:
    frame = cycles.copy()
    eligible = frame["eligible_for_primary_evaluation"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().eq("true")
    duration = pd.to_numeric(frame["cycle_length_days"], errors="coerce")
    frame = frame.loc[eligible & duration.gt(0)].copy()
    frame["cycle_length_days"] = duration.loc[frame.index]
    frame["cycle_start_day"] = pd.to_numeric(
        frame["cycle_start_day"], errors="coerce"
    )
    return frame.sort_values(["participant_id", "cycle_start_day"])


def _serial_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    raw_pairs: list[tuple[float, float]] = []
    centered_pairs: list[tuple[float, float]] = []
    for _, group in frame.groupby("participant_id"):
        values = group.sort_values("cycle_start_day")[
            "cycle_length_days"
        ].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        raw_pairs.extend(zip(values[:-1], values[1:]))
        centered = values - values.mean()
        centered_pairs.extend(zip(centered[:-1], centered[1:]))
    raw = np.asarray(raw_pairs, dtype=float)
    centered = np.asarray(centered_pairs, dtype=float)
    raw_correlation = (
        float(np.corrcoef(raw[:, 0], raw[:, 1])[0, 1])
        if len(raw) >= 3 and np.std(raw[:, 0]) > 0 and np.std(raw[:, 1]) > 0
        else np.nan
    )
    denominator = float(np.sum(centered[:, 0] ** 2)) if len(centered) else 0.0
    fixed_effect_ar1 = (
        float(np.sum(centered[:, 0] * centered[:, 1]) / denominator)
        if denominator > 0
        else np.nan
    )
    return {
        "consecutive_pairs": int(len(raw)),
        "raw_lag1_correlation": raw_correlation,
        "participant_centered_ar1": fixed_effect_ar1,
    }


def _duration_summary(
    frame: pd.DataFrame, source: str
) -> pd.DataFrame:
    values = frame["cycle_length_days"].to_numpy(dtype=float)
    participant_mean = frame.groupby("participant_id")[
        "cycle_length_days"
    ].transform("mean")
    grand_mean = float(np.mean(values))
    between = float(
        np.mean(
            (
                frame.groupby("participant_id")["cycle_length_days"].mean()
                - grand_mean
            )
            ** 2
        )
    )
    within = float(np.mean((frame["cycle_length_days"] - participant_mean) ** 2))
    serial = _serial_statistics(frame)
    return pd.DataFrame(
        [
            {
                "source": source,
                "cycles": int(len(frame)),
                "participants": int(frame["participant_id"].nunique()),
                "mean_days": grand_mean,
                "sd_days": float(np.std(values, ddof=1)),
                "coefficient_of_variation": float(
                    np.std(values, ddof=1) / grand_mean
                ),
                "skewness": float(skew(values, bias=False)),
                "q05_days": float(np.quantile(values, 0.05)),
                "median_days": float(np.median(values)),
                "q95_days": float(np.quantile(values, 0.95)),
                "between_person_variance": between,
                "within_person_variance": within,
                "intraclass_fraction": float(
                    between / (between + within)
                    if between + within > 0
                    else np.nan
                ),
                **serial,
            }
        ]
    )


def _hazard(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rounded = np.rint(frame["cycle_length_days"].to_numpy(dtype=float)).astype(int)
    rows = []
    for day in range(max(1, int(rounded.min())), int(rounded.max()) + 1):
        at_risk = int(np.sum(rounded >= day))
        events = int(np.sum(rounded == day))
        rows.append(
            {
                "source": source,
                "day": day,
                "at_risk": at_risk,
                "events": events,
                "hazard": float(events / at_risk) if at_risk else np.nan,
                "survival_before_day": float(at_risk / len(rounded)),
            }
        )
    return pd.DataFrame(rows)


def _participant_statistics(
    frame: pd.DataFrame, source: str
) -> pd.DataFrame:
    rows = []
    for participant_id, group in frame.groupby("participant_id"):
        values = group.sort_values("cycle_start_day")[
            "cycle_length_days"
        ].to_numpy(dtype=float)
        rows.append(
            {
                "source": source,
                "participant_id": participant_id,
                "cycles": int(len(values)),
                "mean_days": float(np.mean(values)),
                "sd_days": float(np.std(values, ddof=1))
                if len(values) > 1
                else np.nan,
                "range_days": float(np.ptp(values)),
                "first_last_change_days": float(values[-1] - values[0])
                if len(values) > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _stage_durations(
    frame: pd.DataFrame, references: pd.DataFrame, source: str
) -> pd.DataFrame:
    if references.empty:
        return pd.DataFrame()
    ref = references.loc[
        references["event_type"].astype(str).str.contains("lh", case=False)
    ].copy()
    if ref.empty:
        return pd.DataFrame()
    merged = ref.merge(
        frame[
            [
                "participant_id",
                "cycle_id",
                "cycle_start_day",
                "cycle_length_days",
            ]
        ],
        on=["participant_id", "cycle_id"],
        how="inner",
    )
    if "cycle_day" in merged:
        transition = pd.to_numeric(merged["cycle_day"], errors="coerce")
    else:
        transition = (
            pd.to_numeric(merged["day_in_study"], errors="coerce")
            - pd.to_numeric(merged["cycle_start_day"], errors="coerce")
            + 1
        )
    merged["follicular_proxy_days"] = transition
    merged["luteal_proxy_days"] = (
        merged["cycle_length_days"] - merged["follicular_proxy_days"]
    )
    merged = merged.loc[
        merged["follicular_proxy_days"].gt(0)
        & merged["luteal_proxy_days"].gt(0)
    ].copy()
    if merged.empty:
        return merged
    merged["source"] = source
    merged["reference_semantics"] = (
        "urinary_LH_positive_surrogate_not_confirmed_ovulation"
    )
    return merged[
        [
            "source",
            "participant_id",
            "cycle_id",
            "cycle_length_days",
            "follicular_proxy_days",
            "luteal_proxy_days",
            "reference_semantics",
        ]
    ]


def _missingness_statistics(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
    source: str,
    signals: tuple[str, ...],
) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame()
    observations = observations.loc[
        observations["signal_name"].isin(signals)
    ].copy()
    observations["observed"] = pd.to_numeric(
        observations["value"], errors="coerce"
    ).notna()
    coverage = (
        observations.groupby(["participant_id", "cycle_id", "signal_name"])[
            "observed"
        ]
        .mean()
        .unstack("signal_name")
        .reindex(columns=signals)
    )
    coverage["minimum_signal_coverage"] = coverage.min(axis=1)
    merged = frame.merge(
        coverage["minimum_signal_coverage"],
        on=["participant_id", "cycle_id"],
        how="left",
    )
    merged["minimum_signal_coverage"] = merged[
        "minimum_signal_coverage"
    ].fillna(0.0)
    correlation = (
        float(
            np.corrcoef(
                merged["minimum_signal_coverage"],
                merged["cycle_length_days"],
            )[0, 1]
        )
        if len(merged) >= 3
        and merged["minimum_signal_coverage"].std() > 0
        else np.nan
    )
    try:
        merged["coverage_quartile"] = pd.qcut(
            merged["minimum_signal_coverage"],
            4,
            duplicates="drop",
        ).astype(str)
    except ValueError:
        merged["coverage_quartile"] = "single_bin"
    rows = []
    for quartile, group in merged.groupby("coverage_quartile", observed=True):
        rows.append(
            {
                "source": source,
                "coverage_group": quartile,
                "cycles": int(len(group)),
                "participants": int(group["participant_id"].nunique()),
                "mean_coverage": float(group["minimum_signal_coverage"].mean()),
                "mean_cycle_days": float(group["cycle_length_days"].mean()),
                "sd_cycle_days": float(group["cycle_length_days"].std(ddof=1)),
                "coverage_duration_correlation_all_cycles": correlation,
            }
        )
    return pd.DataFrame(rows)


def calculate_dynamical_signatures(
    cycles: pd.DataFrame,
    references: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    source: str,
    signals: tuple[str, ...],
) -> DynamicalSignatures:
    frame = _eligible_cycles(cycles)
    if frame.empty:
        raise ValueError(f"no eligible cycle durations for {source}")
    return DynamicalSignatures(
        summary=_duration_summary(frame, source),
        empirical_hazard=_hazard(frame, source),
        participant_statistics=_participant_statistics(frame, source),
        stage_durations=_stage_durations(frame, references, source),
        missingness_statistics=_missingness_statistics(
            frame, observations, source, signals
        ),
    )
