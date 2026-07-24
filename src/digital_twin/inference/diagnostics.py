from __future__ import annotations

import numpy as np


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    log_weights = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        return np.full(log_weights.size, 1 / log_weights.size), -np.inf
    unnormalized = np.exp(log_weights - maximum)
    total = float(unnormalized.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(log_weights.size, 1 / log_weights.size), -np.inf
    return unnormalized / total, maximum + np.log(total)
