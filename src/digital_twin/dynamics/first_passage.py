from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import invgauss


@dataclass(frozen=True)
class FirstPassageMoments:
    """Moments for drifted Brownian first passage to a positive boundary."""

    mean: float
    variance: float
    standard_deviation: float
    coefficient_of_variation: float
    skewness: float


@dataclass(frozen=True)
class DimensionlessGroups:
    """Control groups for a stage of length ``boundary``.

    ``peclet`` compares directional progression with progress diffusion.
    ``relaxation_ratio`` compares the log-speed relaxation time with the
    deterministic stage-passage time. ``stationary_log_speed_sd`` is the
    stationary OU spread when kappa is positive.
    """

    peclet: float
    relaxation_ratio: float
    stationary_log_speed_sd: float
    deterministic_passage_time: float


def _validate(drift: float, diffusion: float, boundary: float) -> None:
    if drift <= 0:
        raise ValueError("drift must be positive")
    if diffusion <= 0:
        raise ValueError("diffusion must be positive")
    if boundary <= 0:
        raise ValueError("boundary must be positive")


def first_passage_moments(
    drift: float, diffusion: float, boundary: float = 1.0
) -> FirstPassageMoments:
    """Closed-form moments for ``dZ=v dt + sqrt(2D) dW``.

    The process begins at zero on the real line and stops on first reaching
    ``boundary``. Its passage time is inverse Gaussian with
    ``mean=boundary/drift`` and ``shape=boundary**2/(2*diffusion)``.
    """

    _validate(drift, diffusion, boundary)
    mean = boundary / drift
    variance = 2.0 * diffusion * boundary / drift**3
    sd = np.sqrt(variance)
    return FirstPassageMoments(
        mean=float(mean),
        variance=float(variance),
        standard_deviation=float(sd),
        coefficient_of_variation=float(sd / mean),
        skewness=float(3.0 * np.sqrt(2.0 * diffusion / (boundary * drift))),
    )


def _scipy_inverse_gaussian(
    drift: float, diffusion: float, boundary: float
):
    _validate(drift, diffusion, boundary)
    mean = boundary / drift
    shape = boundary**2 / (2.0 * diffusion)
    return invgauss(mu=mean / shape, scale=shape)


def first_passage_density(
    time: np.ndarray | float,
    drift: float,
    diffusion: float,
    boundary: float = 1.0,
) -> np.ndarray:
    """Inverse-Gaussian first-passage density."""

    _validate(drift, diffusion, boundary)
    t = np.asarray(time, dtype=float)
    density = np.zeros_like(t)
    positive = t > 0
    tp = t[positive]
    density[positive] = (
        boundary
        / np.sqrt(4.0 * np.pi * diffusion * tp**3)
        * np.exp(-((boundary - drift * tp) ** 2) / (4.0 * diffusion * tp))
    )
    return density


def first_passage_survival(
    time: np.ndarray | float,
    drift: float,
    diffusion: float,
    boundary: float = 1.0,
) -> np.ndarray:
    """Survival probability of the drifted-Brownian passage time."""

    t = np.asarray(time, dtype=float)
    survival = np.ones_like(t)
    positive = t > 0
    survival[positive] = _scipy_inverse_gaussian(
        drift, diffusion, boundary
    ).sf(t[positive])
    survival[t == np.inf] = 0.0
    return np.clip(survival, 0.0, 1.0)


def first_passage_hazard(
    time: np.ndarray | float,
    drift: float,
    diffusion: float,
    boundary: float = 1.0,
) -> np.ndarray:
    density = first_passage_density(time, drift, diffusion, boundary)
    survival = first_passage_survival(time, drift, diffusion, boundary)
    return np.divide(
        density,
        survival,
        out=np.zeros_like(density),
        where=survival > np.finfo(float).tiny,
    )


def sample_constant_first_passage(
    n: int,
    drift: float,
    diffusion: float,
    boundary: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Exact inverse-Gaussian samples for the constant-coefficient limit."""

    if n < 1:
        raise ValueError("n must be positive")
    moments = first_passage_moments(drift, diffusion, boundary)
    shape = boundary**2 / (2.0 * diffusion)
    return np.random.default_rng(seed).wald(moments.mean, shape, size=n)


def fit_constant_first_passage(
    durations: np.ndarray, boundary: float = 1.0
) -> dict[str, float]:
    """Closed-form inverse-Gaussian maximum-likelihood estimates."""

    x = np.asarray(durations, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2:
        raise ValueError("at least two positive durations are required")
    mean = float(np.mean(x))
    denominator = float(np.sum((x - mean) ** 2 / (mean**2 * x)))
    shape = len(x) / max(denominator, np.finfo(float).tiny)
    drift = boundary / mean
    diffusion = boundary**2 / (2.0 * shape)
    distribution = _scipy_inverse_gaussian(drift, diffusion, boundary)
    return {
        "drift": float(drift),
        "diffusion": float(diffusion),
        "mean_duration": mean,
        "log_likelihood": float(np.sum(distribution.logpdf(x))),
        "n": int(len(x)),
    }


def dimensionless_groups(
    drift: float,
    diffusion: float,
    kappa: float,
    sigma_log_speed: float,
    boundary: float = 1.0,
) -> DimensionlessGroups:
    _validate(drift, diffusion, boundary)
    if kappa < 0 or sigma_log_speed < 0:
        raise ValueError("kappa and sigma_log_speed must be nonnegative")
    passage_time = boundary / drift
    stationary_sd = (
        sigma_log_speed / np.sqrt(2.0 * kappa) if kappa > 0 else np.inf
    )
    return DimensionlessGroups(
        peclet=float(drift * boundary / diffusion),
        relaxation_ratio=float(kappa * passage_time),
        stationary_log_speed_sd=float(stationary_sd),
        deterministic_passage_time=float(passage_time),
    )


def simulate_ou_first_passage(
    n: int,
    mean_speed: float,
    kappa: float,
    sigma_log_speed: float,
    diffusion: float,
    *,
    boundary: float = 1.0,
    dt: float = 0.05,
    max_time: float = 90.0,
    seed: int = 0,
    stationary_initial: bool = True,
) -> np.ndarray:
    """Vectorized first-passage simulation with OU log progression speed.

    ``d ell = kappa(log(mean_speed)-ell)dt + sigma_log_speed dW`` and
    ``d z = exp(ell)dt + sqrt(2*diffusion)dW``. The left boundary reflects
    numerically at zero. Non-crossing trajectories are returned as ``NaN``.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if mean_speed <= 0 or kappa < 0 or sigma_log_speed < 0:
        raise ValueError("invalid OU speed parameters")
    if diffusion < 0 or boundary <= 0 or dt <= 0 or max_time <= 0:
        raise ValueError("invalid passage parameters")
    rng = np.random.default_rng(seed)
    mean_log_speed = np.log(mean_speed)
    initial_sd = (
        sigma_log_speed / np.sqrt(2.0 * kappa)
        if stationary_initial and kappa > 0
        else 0.0
    )
    ell = rng.normal(mean_log_speed, initial_sd, size=n)
    z = np.zeros(n)
    times = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    root_dt = np.sqrt(dt)
    progress_noise = np.sqrt(2.0 * diffusion * dt)
    for step in range(1, int(np.ceil(max_time / dt)) + 1):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        ell[idx] += (
            kappa * (mean_log_speed - ell[idx]) * dt
            + sigma_log_speed * root_dt * rng.normal(size=len(idx))
        )
        before = z[idx].copy()
        increment = (
            np.exp(ell[idx]) * dt
            + progress_noise * rng.normal(size=len(idx))
        )
        z[idx] = np.maximum(before + increment, 0.0)
        crossed_local = z[idx] >= boundary
        if crossed_local.any():
            crossed_idx = idx[crossed_local]
            positive_increment = np.maximum(
                z[crossed_idx] - before[crossed_local], np.finfo(float).eps
            )
            fraction = np.clip(
                (boundary - before[crossed_local]) / positive_increment,
                0.0,
                1.0,
            )
            times[crossed_idx] = (step - 1 + fraction) * dt
            active[crossed_idx] = False
    return times
