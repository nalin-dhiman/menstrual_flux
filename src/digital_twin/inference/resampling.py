from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(w**2))


def systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def stratified(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (np.arange(n) + rng.random(n)) / n
    return np.searchsorted(np.cumsum(weights), positions, side="right")


def multinomial(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(weights), size=len(weights), replace=True, p=weights)


def residual(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    copies = np.floor(n * weights).astype(int)
    fixed = np.repeat(np.arange(n), copies)
    remaining = n - len(fixed)
    if remaining:
        residual_weights = n * weights - copies
        residual_weights /= residual_weights.sum()
        extra = rng.choice(n, size=remaining, replace=True, p=residual_weights)
        fixed = np.concatenate([fixed, extra])
    rng.shuffle(fixed)
    return fixed


RESAMPLERS = {"systematic": systematic, "stratified": stratified, "residual": residual, "multinomial": multinomial}


def resample(weights: np.ndarray, method: str, rng: np.random.Generator) -> np.ndarray:
    try:
        return RESAMPLERS[method](weights, rng)
    except KeyError as exc:
        raise ValueError(f"Unknown resampling method: {method}") from exc
