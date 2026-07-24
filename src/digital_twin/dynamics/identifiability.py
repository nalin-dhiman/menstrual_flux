from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import kstest, skew

from .first_passage import (
    fit_constant_first_passage,
    sample_constant_first_passage,
    simulate_ou_first_passage,
)


@dataclass
class IdentifiabilityResults:
    sbc_draws: pd.DataFrame
    sbc_summary: pd.DataFrame
    profile_likelihood: pd.DataFrame
    ou_degeneracy_surface: pd.DataFrame
    ou_degeneracy_summary: pd.DataFrame


def _loglikelihood(
    durations: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    boundary: float = 1.0,
) -> np.ndarray:
    x = np.asarray(durations, dtype=float)
    v = np.asarray(drift, dtype=float)[..., None]
    d = np.asarray(diffusion, dtype=float)[..., None]
    return np.sum(
        np.log(boundary)
        - 0.5 * np.log(4.0 * np.pi * d)
        - 1.5 * np.log(x)
        - ((boundary - v * x) ** 2) / (4.0 * d * x),
        axis=-1,
    )


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, probability, side="left")])


def inverse_gaussian_profile_likelihood(
    durations: np.ndarray,
    *,
    drift_points: int = 60,
    diffusion_points: int = 60,
) -> pd.DataFrame:
    x = np.asarray(durations, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    estimate = fit_constant_first_passage(x)
    v_hat = estimate["drift"]
    d_hat = estimate["diffusion"]
    rows = []
    drift_grid = np.linspace(0.55 * v_hat, 1.45 * v_hat, drift_points)
    for drift in drift_grid:
        optimum = minimize_scalar(
            lambda log_d: -float(
                _loglikelihood(x, np.array(np.array(drift)), np.exp(log_d))
            ),
            bounds=(np.log(d_hat / 20.0), np.log(d_hat * 20.0)),
            method="bounded",
        )
        rows.append(
            {
                "profile_parameter": "drift",
                "value": float(drift),
                "optimized_nuisance": float(np.exp(optimum.x)),
                "log_likelihood": float(-optimum.fun),
            }
        )
    diffusion_grid = np.exp(
        np.linspace(
            np.log(d_hat / 8.0),
            np.log(d_hat * 8.0),
            diffusion_points,
        )
    )
    for diffusion in diffusion_grid:
        optimum = minimize_scalar(
            lambda drift: -float(
                _loglikelihood(x, np.array(drift), np.array(diffusion))
            ),
            bounds=(0.4 * v_hat, 1.8 * v_hat),
            method="bounded",
        )
        rows.append(
            {
                "profile_parameter": "diffusion",
                "value": float(diffusion),
                "optimized_nuisance": float(optimum.x),
                "log_likelihood": float(-optimum.fun),
            }
        )
    result = pd.DataFrame(rows)
    result["profile_deviance"] = 2.0 * (
        result.groupby("profile_parameter")["log_likelihood"].transform("max")
        - result["log_likelihood"]
    )
    return result


def run_inverse_gaussian_sbc(
    *,
    sample_sizes: tuple[int, ...],
    replicates: int,
    drift_range: tuple[float, float],
    diffusion_range: tuple[float, float],
    grid_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grid-posterior SBC and contraction for the analytic limiting model."""

    if replicates < 2 or grid_size < 10:
        raise ValueError("SBC requires at least two replicates and grid_size >= 10")
    rng = np.random.default_rng(seed)
    drift_grid = np.linspace(*drift_range, grid_size)
    log_diffusion_grid = np.linspace(
        np.log(diffusion_range[0]),
        np.log(diffusion_range[1]),
        grid_size,
    )
    diffusion_grid = np.exp(log_diffusion_grid)
    vv, dd = np.meshgrid(drift_grid, diffusion_grid, indexing="ij")
    rows = []
    for sample_size in sample_sizes:
        for replicate in range(replicates):
            true_drift = float(rng.uniform(*drift_range))
            true_log_diffusion = float(
                rng.uniform(*np.log(np.asarray(diffusion_range)))
            )
            true_diffusion = float(np.exp(true_log_diffusion))
            data = sample_constant_first_passage(
                sample_size,
                true_drift,
                true_diffusion,
                seed=seed + 7919 * sample_size + replicate,
            )
            log_weight = _loglikelihood(data, vv, dd)
            log_weight -= np.max(log_weight)
            weight = np.exp(log_weight)
            weight /= np.sum(weight)
            drift_weight = weight.sum(axis=1)
            diffusion_weight = weight.sum(axis=0)
            drift_mean = float(np.sum(drift_grid * drift_weight))
            diffusion_mean = float(
                np.sum(diffusion_grid * diffusion_weight)
            )
            drift_sd = float(
                np.sqrt(
                    np.sum((drift_grid - drift_mean) ** 2 * drift_weight)
                )
            )
            diffusion_sd = float(
                np.sqrt(
                    np.sum(
                        (diffusion_grid - diffusion_mean) ** 2
                        * diffusion_weight
                    )
                )
            )
            v_centered = vv - drift_mean
            d_centered = dd - diffusion_mean
            covariance = float(np.sum(v_centered * d_centered * weight))
            correlation = covariance / max(
                drift_sd * diffusion_sd, np.finfo(float).tiny
            )
            drift_lower = _weighted_quantile(
                drift_grid, drift_weight, 0.025
            )
            drift_upper = _weighted_quantile(
                drift_grid, drift_weight, 0.975
            )
            diffusion_lower = _weighted_quantile(
                diffusion_grid, diffusion_weight, 0.025
            )
            diffusion_upper = _weighted_quantile(
                diffusion_grid, diffusion_weight, 0.975
            )
            rows.append(
                {
                    "sample_size": sample_size,
                    "replicate": replicate,
                    "true_drift": true_drift,
                    "posterior_mean_drift": drift_mean,
                    "posterior_sd_drift": drift_sd,
                    "drift_covered_95": (
                        drift_lower <= true_drift <= drift_upper
                    ),
                    "drift_posterior_rank": float(
                        drift_weight[drift_grid <= true_drift].sum()
                    ),
                    "true_diffusion": true_diffusion,
                    "posterior_mean_diffusion": diffusion_mean,
                    "posterior_sd_diffusion": diffusion_sd,
                    "diffusion_covered_95": (
                        diffusion_lower <= true_diffusion <= diffusion_upper
                    ),
                    "diffusion_posterior_rank": float(
                        diffusion_weight[
                            diffusion_grid <= true_diffusion
                        ].sum()
                    ),
                    "posterior_parameter_correlation": correlation,
                }
            )
    draws = pd.DataFrame(rows)
    summaries = []
    prior_drift_sd = (drift_range[1] - drift_range[0]) / np.sqrt(12.0)
    log_low, log_high = np.log(diffusion_range)
    prior_diffusion_mean = (
        diffusion_range[1] - diffusion_range[0]
    ) / (log_high - log_low)
    prior_diffusion_second = (
        diffusion_range[1] ** 2 - diffusion_range[0] ** 2
    ) / (2.0 * (log_high - log_low))
    prior_diffusion_sd = np.sqrt(
        prior_diffusion_second - prior_diffusion_mean**2
    )
    for sample_size, group in draws.groupby("sample_size"):
        drift_ks = kstest(group["drift_posterior_rank"], "uniform")
        diffusion_ks = kstest(
            group["diffusion_posterior_rank"], "uniform"
        )
        summaries.append(
            {
                "sample_size": int(sample_size),
                "replicates": int(len(group)),
                "drift_coverage_95": float(group["drift_covered_95"].mean()),
                "diffusion_coverage_95": float(
                    group["diffusion_covered_95"].mean()
                ),
                "drift_rank_uniform_ks_p": float(drift_ks.pvalue),
                "diffusion_rank_uniform_ks_p": float(
                    diffusion_ks.pvalue
                ),
                "drift_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["posterior_mean_drift"]
                                - group["true_drift"]
                            )
                            ** 2
                        )
                    )
                ),
                "diffusion_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                group["posterior_mean_diffusion"]
                                - group["true_diffusion"]
                            )
                            ** 2
                        )
                    )
                ),
                "drift_contraction": float(
                    1.0
                    - group["posterior_sd_drift"].mean() / prior_drift_sd
                ),
                "diffusion_contraction": float(
                    1.0
                    - group["posterior_sd_diffusion"].mean()
                    / prior_diffusion_sd
                ),
                "mean_abs_posterior_parameter_correlation": float(
                    group["posterior_parameter_correlation"].abs().mean()
                ),
            }
        )
    return draws, pd.DataFrame(summaries)


def ou_identifiability_surface(
    *,
    mean_speed: float,
    target_kappa: float,
    target_stationary_log_speed_sd: float,
    diffusion: float,
    kappa_values: tuple[float, ...],
    stationary_sd_values: tuple[float, ...],
    trajectories: int,
    dt: float,
    max_time: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Show which OU parameter pairs produce indistinguishable passage summaries."""

    target_sigma = target_stationary_log_speed_sd * np.sqrt(
        2.0 * target_kappa
    )
    target = simulate_ou_first_passage(
        trajectories * 2,
        mean_speed,
        target_kappa,
        target_sigma,
        diffusion,
        dt=dt,
        max_time=max_time,
        seed=seed,
    )
    target = target[np.isfinite(target)]
    target_mean = float(np.mean(target))
    target_cv = float(np.std(target, ddof=1) / target_mean)
    target_skew = float(skew(target, bias=False))
    rows = []
    for i, kappa in enumerate(kappa_values):
        for j, stationary_sd in enumerate(stationary_sd_values):
            sigma = stationary_sd * np.sqrt(2.0 * kappa)
            samples = simulate_ou_first_passage(
                trajectories,
                mean_speed,
                kappa,
                sigma,
                diffusion,
                dt=dt,
                max_time=max_time,
                seed=seed + 1009 * i + 9176 * j,
            )
            finite = samples[np.isfinite(samples)]
            mean = float(np.mean(finite))
            cv = float(np.std(finite, ddof=1) / mean)
            asymmetry = float(skew(finite, bias=False))
            distance = (
                ((mean - target_mean) / (0.05 * target_mean)) ** 2
                + ((cv - target_cv) / 0.03) ** 2
                + ((asymmetry - target_skew) / 0.30) ** 2
            )
            rows.append(
                {
                    "kappa": float(kappa),
                    "stationary_log_speed_sd": float(stationary_sd),
                    "sigma_log_speed": float(sigma),
                    "mean_days": mean,
                    "coefficient_of_variation": cv,
                    "skewness": asymmetry,
                    "noncrossing_fraction": float(
                        1.0 - len(finite) / trajectories
                    ),
                    "summary_distance": float(distance),
                }
            )
    surface = pd.DataFrame(rows)
    surface["delta_distance"] = (
        surface["summary_distance"] - surface["summary_distance"].min()
    )
    close = surface.loc[surface["delta_distance"] <= 1.0]
    correlation = (
        float(
            np.corrcoef(
                close["kappa"], close["stationary_log_speed_sd"]
            )[0, 1]
        )
        if len(close) >= 3
        and close["kappa"].std() > 0
        and close["stationary_log_speed_sd"].std() > 0
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "target_kappa": target_kappa,
                "target_stationary_log_speed_sd": (
                    target_stationary_log_speed_sd
                ),
                "target_mean_days": target_mean,
                "target_coefficient_of_variation": target_cv,
                "target_skewness": target_skew,
                "grid_candidates": int(len(surface)),
                "near_equivalent_candidates_delta_le_1": int(len(close)),
                "near_equivalent_kappa_sd_correlation": correlation,
                "identifiable_from_three_duration_summaries": bool(
                    len(close) == 1
                ),
            }
        ]
    )
    return surface, summary
