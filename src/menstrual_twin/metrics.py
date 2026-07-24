from __future__ import annotations

import numpy as np

from .math_utils import TWO_PI, circular_difference


def circular_mae_days(true_phase: np.ndarray, estimated_phase: np.ndarray, cycle_days: float = 29.0) -> float:
    diff = np.abs(circular_difference(estimated_phase, true_phase))
    return float(np.mean(diff / TWO_PI * cycle_days))


def interval_coverage(samples: np.ndarray, truth: float, level: float = 0.90) -> bool:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return bool(lo <= truth <= hi)


def continuous_ranked_probability_score(samples: np.ndarray, truth: float) -> float:
    """Monte Carlo CRPS for a scalar predictive distribution."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    term1 = np.mean(np.abs(x - truth))
    # Efficient E|X-X'| using sorted values.
    xs = np.sort(x)
    n = xs.size
    coeff = 2 * np.arange(1, n + 1) - n - 1
    term2 = 2.0 * np.sum(coeff * xs) / (n * n)
    return float(term1 - 0.5 * term2)
