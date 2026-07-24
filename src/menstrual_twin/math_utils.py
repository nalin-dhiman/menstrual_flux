from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_phase(theta: np.ndarray | float) -> np.ndarray | float:
    return np.mod(theta, TWO_PI)


def circular_difference(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """Signed shortest angular difference a-b in [-pi, pi)."""
    return (a - b + np.pi) % TWO_PI - np.pi


def circular_mean(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.full(theta.shape, 1.0 / theta.size)
    z = np.sum(weights * np.exp(1j * theta))
    return float(np.angle(z) % TWO_PI)


def circular_concentration(theta: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.full(theta.shape, 1.0 / theta.size)
    return float(np.abs(np.sum(weights * np.exp(1j * theta))))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def normal_logpdf(x: np.ndarray, mean: np.ndarray, sd: float) -> np.ndarray:
    var = sd * sd
    return -0.5 * ((x - mean) ** 2 / var + np.log(2.0 * np.pi * var))


def lognormal_logpdf(x: float, median: np.ndarray, log_sd: float) -> np.ndarray:
    if not np.isfinite(x) or x <= 0:
        return np.full_like(median, -np.inf, dtype=float)
    logx = np.log(x)
    logmed = np.log(np.clip(median, 1e-9, None))
    return -np.log(x * log_sd * np.sqrt(2.0 * np.pi)) - 0.5 * ((logx - logmed) / log_sd) ** 2


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)
