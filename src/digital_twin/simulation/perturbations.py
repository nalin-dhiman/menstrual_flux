from __future__ import annotations

import numpy as np


def ou_recovery_path(n_days: int, kappa: float, sigma: float, impulse: float, rng: np.random.Generator) -> np.ndarray:
    """Exploratory synthetic displacement path; not used for real-data claims."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    r = np.zeros(n_days, dtype=float)
    if n_days:
        r[0] = impulse
    for t in range(1, n_days):
        r[t] = r[t - 1] - kappa * r[t - 1] + sigma * rng.normal()
    return r
