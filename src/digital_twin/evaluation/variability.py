from __future__ import annotations

import numpy as np
import pandas as pd


def _components(cycles: pd.DataFrame) -> dict[str, float]:
    frame = cycles.dropna(subset=["cycle_length", "follicular_duration", "luteal_duration"]).copy()
    total = float(np.var(frame["cycle_length"], ddof=0))
    var_f = float(np.var(frame["follicular_duration"], ddof=0))
    var_l = float(np.var(frame["luteal_duration"], ddof=0))
    cov_term = float(2 * np.cov(frame["follicular_duration"], frame["luteal_duration"], ddof=0)[0, 1])
    participant_mean = frame.groupby("participant_id")["cycle_length"].transform("mean")
    grand = float(frame["cycle_length"].mean())
    between = float(np.mean((participant_mean - grand) ** 2))
    within = float(np.mean((frame["cycle_length"] - participant_mean) ** 2))
    drift_prediction = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby("participant_id"):
        x = group["cycle_id"].to_numpy(dtype=float)
        y = group["cycle_length"].to_numpy(dtype=float)
        if len(group) >= 3 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            prediction = intercept + slope * x
            prediction -= prediction.mean() - y.mean()
        else:
            prediction = np.full_like(y, y.mean())
        drift_prediction.loc[group.index] = prediction
    drift = float(np.mean((drift_prediction - participant_mean) ** 2))
    residual = max(within - drift, 0.0)
    return {
        "total_variance": total,
        "follicular_variance": var_f,
        "luteal_variance": var_l,
        "twice_stage_covariance": cov_term,
        "between_person": between,
        "within_person": within,
        "estimated_drift": drift,
        "measurement": 0.0,
        "residual_within": residual,
    }


def variability_decomposition(cycles: pd.DataFrame, bootstrap_replicates: int = 300, seed: int = 0) -> pd.DataFrame:
    """Truth-based synthetic decomposition with participant bootstrap intervals."""
    point = _components(cycles)
    participant_ids = cycles["participant_id"].unique()
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in point}
    for _ in range(bootstrap_replicates):
        sampled = rng.choice(participant_ids, len(participant_ids), replace=True)
        pieces = []
        for index, participant_id in enumerate(sampled):
            piece = cycles[cycles["participant_id"] == participant_id].copy()
            piece["participant_id"] = f"bootstrap-{index}"
            pieces.append(piece)
        values = _components(pd.concat(pieces, ignore_index=True))
        for key, value in values.items():
            draws[key].append(value)
    return pd.DataFrame([
        {
            "component": key,
            "estimate_days_squared": value,
            "lower_95": float(np.quantile(draws[key], 0.025)),
            "upper_95": float(np.quantile(draws[key], 0.975)),
            "truth_basis": "simulated_exact_event_times",
        }
        for key, value in point.items()
    ])
