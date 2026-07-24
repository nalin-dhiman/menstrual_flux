from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew

from .first_passage import simulate_ou_first_passage


def classify_regime(
    *,
    peclet: float,
    relaxation_ratio: float,
    coefficient_of_variation: float,
    noncrossing_fraction: float,
    q90_to_median: float,
) -> str:
    if noncrossing_fraction >= 0.05:
        return "stalled_or_extreme_tail"
    if coefficient_of_variation >= 0.35 or q90_to_median >= 1.6:
        return "broad_first_passage_tail"
    if peclet <= 10:
        return "progress_diffusion_dominated"
    if relaxation_ratio <= 0.3:
        return "persistent_speed_heterogeneity"
    if coefficient_of_variation <= 0.15 and peclet >= 40:
        return "drift_dominated_regular"
    return "mixed_drift_diffusion"


def build_regime_map(
    *,
    mean_duration: float,
    peclet_values: tuple[float, ...],
    relaxation_values: tuple[float, ...],
    stationary_log_speed_sd: float,
    trajectories: int,
    dt: float,
    max_time: float,
    seed: int,
) -> pd.DataFrame:
    """Simulate the nondimensional Pe--R regime map.

    The stationary spread of log speed is held fixed while its relaxation
    timescale changes, separating persistence from amplitude.
    """

    if mean_duration <= 0 or stationary_log_speed_sd < 0:
        raise ValueError("invalid regime-map parameters")
    mean_speed = 1.0 / mean_duration
    rows = []
    for pe_index, peclet in enumerate(peclet_values):
        if peclet <= 0:
            raise ValueError("Peclet numbers must be positive")
        diffusion = mean_speed / peclet
        for r_index, relaxation in enumerate(relaxation_values):
            if relaxation < 0:
                raise ValueError("relaxation ratios must be nonnegative")
            kappa = relaxation / mean_duration
            sigma_log_speed = (
                stationary_log_speed_sd * np.sqrt(2.0 * kappa)
                if kappa > 0
                else 0.0
            )
            samples = simulate_ou_first_passage(
                trajectories,
                mean_speed,
                kappa,
                sigma_log_speed,
                diffusion,
                dt=dt,
                max_time=max_time,
                seed=seed + 1009 * pe_index + 9176 * r_index,
            )
            finite = samples[np.isfinite(samples)]
            noncrossing = 1.0 - len(finite) / trajectories
            mean = float(np.mean(finite)) if len(finite) else np.nan
            sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
            median = float(np.median(finite)) if len(finite) else np.nan
            q90 = float(np.quantile(finite, 0.90)) if len(finite) else np.nan
            cv = sd / mean if mean > 0 else np.nan
            q90_ratio = q90 / median if median > 0 else np.nan
            rows.append(
                {
                    "peclet": float(peclet),
                    "relaxation_ratio": float(relaxation),
                    "mean_days": mean,
                    "sd_days": sd,
                    "coefficient_of_variation": cv,
                    "skewness": float(skew(finite, bias=False))
                    if len(finite) >= 3
                    else np.nan,
                    "median_days": median,
                    "q10_days": float(np.quantile(finite, 0.10))
                    if len(finite)
                    else np.nan,
                    "q90_days": q90,
                    "q90_to_median": q90_ratio,
                    "noncrossing_fraction": noncrossing,
                    "stationary_log_speed_sd": stationary_log_speed_sd,
                    "regime": classify_regime(
                        peclet=peclet,
                        relaxation_ratio=relaxation,
                        coefficient_of_variation=cv,
                        noncrossing_fraction=noncrossing,
                        q90_to_median=q90_ratio,
                    ),
                }
            )
    return pd.DataFrame(rows)
