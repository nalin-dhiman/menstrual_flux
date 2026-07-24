from __future__ import annotations

import numpy as np


def map_log_speed(stage_durations: np.ndarray, prior_mean: float, prior_sd: float = 0.25) -> tuple[float, float]:
    """Normal-normal MAP approximation for mean log speed from stage durations."""
    durations = np.asarray(stage_durations, dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if not durations.size:
        return prior_mean, prior_sd
    observations = -np.log(durations)
    obs_var = max(float(np.var(observations, ddof=1)) if durations.size > 1 else 0.04, 0.01)
    precision = 1 / prior_sd**2 + durations.size / obs_var
    mean = (prior_mean / prior_sd**2 + observations.sum() / obs_var) / precision
    return float(mean), float(np.sqrt(1 / precision))
