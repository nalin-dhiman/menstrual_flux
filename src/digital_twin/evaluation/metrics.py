from __future__ import annotations

import numpy as np


def crps_ensemble(samples: np.ndarray, truth: float) -> float:
    x = np.sort(np.asarray(samples, dtype=float))
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    first = float(np.mean(np.abs(x - truth)))
    # O(n log n) identity for half the mean pairwise absolute distance.
    coefficients = 2 * np.arange(1, x.size + 1) - x.size - 1
    half_pairwise = float(np.sum(coefficients * x) / (x.size**2))
    return first - half_pairwise


def coverage(samples: np.ndarray, truth: float, level: float) -> tuple[bool, float, float, float]:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return False, np.nan, np.nan, np.nan
    alpha = 1 - level
    lower, upper = np.quantile(x, [alpha / 2, 1 - alpha / 2])
    return bool(lower <= truth <= upper), float(lower), float(upper), float(upper - lower)


def interval_score(lower: float, upper: float, truth: float, alpha: float) -> float:
    score = upper - lower
    if truth < lower:
        score += 2 / alpha * (lower - truth)
    elif truth > upper:
        score += 2 / alpha * (truth - upper)
    return float(score)


def weighted_interval_score(samples: np.ndarray, truth: float, levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    median_error = 0.5 * abs(float(np.median(x)) - truth)
    terms = [median_error]
    weights = [0.5]
    for level in levels:
        alpha = 1 - level
        lower, upper = np.quantile(x, [alpha / 2, 1 - alpha / 2])
        terms.append((alpha / 2) * interval_score(float(lower), float(upper), truth, alpha))
        weights.append(alpha / 2)
    return float(np.sum(terms) / np.sum(weights))


def negative_log_predictive_density(samples: np.ndarray, truth: float, bandwidth: float = 1.0) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return np.nan
    density = np.mean(np.exp(-0.5 * ((truth - x) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi)))
    return float(-np.log(max(density, 1e-12)))


def event_metrics(samples: np.ndarray, truth: float, levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)) -> dict[str, float]:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"median": np.nan, "absolute_error": np.nan, "crps": np.nan, "nlpd": np.nan, "wis": np.nan, "abstained": 1.0}
    median = float(np.median(x))
    result = {
        "median": median,
        "absolute_error": abs(median - truth),
        "squared_error": (median - truth) ** 2,
        "crps": crps_ensemble(x, truth),
        "nlpd": negative_log_predictive_density(x, truth),
        "wis": weighted_interval_score(x, truth, levels),
        "abstained": 0.0,
    }
    for level in levels:
        covered, lower, upper, width = coverage(x, truth, level)
        label = int(level * 100)
        result[f"coverage_{label}"] = float(covered)
        result[f"lower_{label}"] = lower
        result[f"upper_{label}"] = upper
        result[f"width_{label}"] = width
    return result


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(outcomes)) ** 2))


def circular_error(estimated: np.ndarray, truth: np.ndarray) -> np.ndarray:
    difference = np.asarray(estimated) - np.asarray(truth)
    return np.arctan2(np.sin(difference), np.cos(difference))


def posterior_mass_in_interval(samples: np.ndarray, lower: float, upper: float) -> float:
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    return np.nan if not x.size else float(np.mean((x >= lower) & (x <= upper)))
