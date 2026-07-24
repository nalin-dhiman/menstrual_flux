from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from digital_twin.config import ExperimentConfig
from digital_twin.inference.particle_filter import ParticleFilterResult


@dataclass
class EventForecast:
    ovulation_samples: np.ndarray
    next_menses_samples: np.ndarray

    @staticmethod
    def summary(samples: np.ndarray) -> dict[str, float]:
        finite = np.asarray(samples)[np.isfinite(samples)]
        if not finite.size:
            return {"median": np.nan, "lower_90": np.nan, "upper_90": np.nan, "n": 0}
        return {
            "median": float(np.median(finite)),
            "lower_90": float(np.quantile(finite, 0.05)),
            "upper_90": float(np.quantile(finite, 0.95)),
            "n": int(finite.size),
        }

    @staticmethod
    def probability_mass(samples: np.ndarray) -> dict[int, float]:
        finite = np.rint(np.asarray(samples)[np.isfinite(samples)]).astype(int)
        if not finite.size:
            return {}
        values, counts = np.unique(finite, return_counts=True)
        return {int(v): float(c / finite.size) for v, c in zip(values, counts)}


def forecast_events(result: ParticleFilterResult, cfg: ExperimentConfig, seed_offset: int = 0) -> EventForecast:
    rng = np.random.default_rng(cfg.experiment.seed + 2017 + seed_offset)
    n = cfg.inference.forecast_samples
    idx = rng.choice(len(result.weights), size=n, replace=True, p=result.weights)
    stage = result.stage[idx].copy()
    z = result.progress[idx].copy()
    ell = result.log_speed[idx].copy()
    ov = result.ovulation_day[idx].copy()
    menses = result.next_menses_day[idx].copy()
    p = cfg.process
    stationary_f = p.cycle_effect_sd_f / np.sqrt(max(1 - p.cycle_effect_rho_f**2, 1e-6))
    stationary_l = p.cycle_effect_sd_l / np.sqrt(max(1 - p.cycle_effect_rho_l**2, 1e-6))
    mu_f = result.mu_log_speed_f[idx] + rng.normal(0, stationary_f, n)
    mu_l = result.mu_log_speed_l[idx] + rng.normal(0, stationary_l, n)
    # Preserve information about the currently occupied stage while retaining
    # cycle-to-cycle uncertainty for the future stage.
    mu_f[stage == 0] = 0.65 * ell[stage == 0] + 0.35 * mu_f[stage == 0]
    mu_l[stage == 1] = 0.65 * ell[stage == 1] + 0.35 * mu_l[stage == 1]
    for future in range(1, cfg.inference.forecast_horizon_days + 1):
        active = np.isnan(menses)
        if not active.any():
            break
        follicular = stage == 0
        mu = np.where(follicular, mu_f, mu_l)
        kappa = np.where(follicular, p.kappa_log_speed_f, p.kappa_log_speed_l)
        sigma_ell = np.where(follicular, p.sigma_log_speed_f, p.sigma_log_speed_l)
        sigma_z = np.where(follicular, p.sigma_progress_f, p.sigma_progress_l)
        ell[active] += kappa[active] * (mu[active] - ell[active]) + sigma_ell[active] * rng.normal(size=active.sum())
        z[active] += np.maximum(np.exp(ell[active]) + sigma_z[active] * rng.normal(size=active.sum()), 0.001)
        cross_f = active & (stage == 0) & (z >= 1)
        ov[cross_f & np.isnan(ov)] = result.final_day + future
        stage[cross_f] = 1
        z[cross_f] -= 1
        ell[cross_f] = mu_l[cross_f]
        cross_l = active & (stage == 1) & (z >= 1)
        menses[cross_l] = result.final_day + future
        stage[cross_l] = 0
        z[cross_l] -= 1
        ell[cross_l] = mu_f[cross_l]
    discrepancy = cfg.inference.forecast_model_discrepancy_sd_days
    if discrepancy > 0:
        ov_finite = np.isfinite(ov)
        menses_finite = np.isfinite(menses)
        ov[ov_finite] += rng.normal(0, discrepancy, ov_finite.sum())
        menses[menses_finite] += rng.normal(0, discrepancy, menses_finite.sum())
        ov[ov_finite] = np.maximum(ov[ov_finite], 0.0)
        menses[menses_finite] = np.maximum(menses[menses_finite], ov[menses_finite] + 1.0)
    return EventForecast(ov, menses)
