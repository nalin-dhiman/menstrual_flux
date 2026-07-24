from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ModelConfig
from .math_utils import (
    TWO_PI,
    circular_concentration,
    circular_mean,
    lognormal_logpdf,
    normal_logpdf,
    systematic_resample,
    wrap_phase,
)
from .observation_models import expected_observations


@dataclass
class ParticleFilterResult:
    summary: pd.DataFrame
    theta_particles: np.ndarray
    log_omega_particles: np.ndarray
    amplitude_particles: np.ndarray
    weights: np.ndarray


def _bernoulli_loglik(y: bool, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p if y else 1.0 - p)


def _normalize_logweights(logw: np.ndarray) -> tuple[np.ndarray, float]:
    m = float(np.max(logw))
    if not np.isfinite(m):
        w = np.full(logw.shape, 1.0 / logw.size)
        return w, -np.inf
    un = np.exp(logw - m)
    z = float(np.sum(un))
    if z <= 0 or not np.isfinite(z):
        return np.full(logw.shape, 1.0 / logw.size), -np.inf
    return un / z, m + np.log(z)


def run_particle_filter(data: pd.DataFrame, cfg: ModelConfig) -> ParticleFilterResult:
    """Bootstrap particle filter for the prototype phase-amplitude model.

    Expected columns: day_in_study, bleeding, and any subset of temperature_c,
    rhr_bpm, lh, e3g, pdg, sleep_hours, stress_event, illness_event.
    """
    required = {"day_in_study", "bleeding"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = data.sort_values("day_in_study").reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed + 17)
    n = cfg.filter.particles
    p = cfg.process
    o = cfg.observation

    # Broad prior. A production filter should initialize from participant history.
    theta = rng.uniform(0.0, TWO_PI, n)
    cycle_days = np.clip(rng.normal(p.mean_cycle_days, cfg.filter.prior_cycle_days_sd, n), 18.0, 60.0)
    log_omega = np.log(TWO_PI / cycle_days)
    amplitude = rng.normal(0.0, 0.25, n)
    weights = np.full(n, 1.0 / n)
    summaries: list[dict[str, float | int]] = []

    for _, row in df.iterrows():
        sleep = row.get("sleep_hours", np.nan)
        sleep_term = 0.0 if pd.isna(sleep) else p.sleep_effect_on_phase * (float(sleep) - 7.1)
        stress = 0.0 if pd.isna(row.get("stress_event", np.nan)) else float(row.get("stress_event", 0.0))
        illness = 0.0 if pd.isna(row.get("illness_event", np.nan)) else float(row.get("illness_event", 0.0))

        mu_log_omega = np.log(TWO_PI / p.mean_cycle_days)
        log_omega = (
            log_omega
            + p.log_omega_reversion * (mu_log_omega - log_omega)
            + p.log_omega_sd * rng.normal(size=n)
        )
        amplitude = (
            amplitude
            - p.amplitude_recovery * amplitude
            + p.stress_effect_on_amplitude * stress
            + p.illness_effect_on_amplitude * illness
            + p.amplitude_sd * rng.normal(size=n)
        )
        theta = wrap_phase(
            theta
            + np.exp(log_omega)
            + sleep_term
            + p.amplitude_to_phase * amplitude
            + p.phase_diffusion_sd * rng.normal(size=n)
        )

        expected = expected_observations(theta, amplitude, cfg)
        loglik = np.zeros(n)
        if "bleeding" in row and not pd.isna(row["bleeding"]):
            loglik += _bernoulli_loglik(bool(row["bleeding"]), expected["bleeding_prob"])
        if "temperature_c" in row and not pd.isna(row["temperature_c"]):
            loglik += normal_logpdf(float(row["temperature_c"]), expected["temperature_c"], o.temperature_sd_c)
        if "rhr_bpm" in row and not pd.isna(row["rhr_bpm"]):
            loglik += normal_logpdf(float(row["rhr_bpm"]), expected["rhr_bpm"], o.rhr_sd_bpm)
        if "lh" in row and not pd.isna(row["lh"]):
            loglik += lognormal_logpdf(float(row["lh"]), expected["lh"], o.lh_log_sd)
        if "e3g" in row and not pd.isna(row["e3g"]):
            loglik += lognormal_logpdf(float(row["e3g"]), expected["e3g"], o.e3g_log_sd)
        if "pdg" in row and not pd.isna(row["pdg"]):
            loglik += lognormal_logpdf(float(row["pdg"]), expected["pdg"], o.pdg_log_sd)

        weights, log_norm = _normalize_logweights(np.log(np.clip(weights, 1e-300, None)) + loglik)
        ess = 1.0 / float(np.sum(weights**2))

        phase_mean = circular_mean(theta, weights)
        concentration = circular_concentration(theta, weights)
        summaries.append({
            "day_in_study": int(row["day_in_study"]),
            "phase_mean_rad": phase_mean,
            "phase_mean_fraction": phase_mean / TWO_PI,
            "phase_concentration": concentration,
            "effective_sample_size": ess,
            "mean_cycle_days": float(np.sum(weights * (TWO_PI / np.exp(log_omega)))),
            "mean_amplitude": float(np.sum(weights * amplitude)),
            "incremental_log_normalizer": log_norm,
        })

        if ess < cfg.filter.resample_ess_fraction * n:
            idx = systematic_resample(weights, rng)
            theta = theta[idx]
            log_omega = log_omega[idx]
            amplitude = amplitude[idx]
            weights.fill(1.0 / n)

    return ParticleFilterResult(
        summary=pd.DataFrame(summaries),
        theta_particles=theta,
        log_omega_particles=log_omega,
        amplitude_particles=amplitude,
        weights=weights,
    )


def forecast_next_menses(result: ParticleFilterResult, cfg: ModelConfig) -> np.ndarray:
    """Simulate first-passage time to the next phase wrap from filtered particles."""
    rng = np.random.default_rng(cfg.seed + 53)
    n = result.theta_particles.size
    idx = rng.choice(n, size=n, replace=True, p=result.weights)
    theta = result.theta_particles[idx].copy()
    log_omega = result.log_omega_particles[idx].copy()
    amplitude = result.amplitude_particles[idx].copy()
    initial_theta = theta.copy()
    crossed = np.zeros(n, dtype=bool)
    times = np.full(n, np.nan)
    p = cfg.process
    mu_log_omega = np.log(TWO_PI / p.mean_cycle_days)

    unwrapped = theta.copy()
    for day in range(1, cfg.filter.forecast_horizon_days + 1):
        log_omega += p.log_omega_reversion * (mu_log_omega - log_omega) + p.log_omega_sd * rng.normal(size=n)
        amplitude += -p.amplitude_recovery * amplitude + p.amplitude_sd * rng.normal(size=n)
        unwrapped += (
            np.exp(log_omega)
            + p.amplitude_to_phase * amplitude
            + p.phase_diffusion_sd * rng.normal(size=n)
        )
        new_cross = (~crossed) & (unwrapped >= TWO_PI)
        times[new_cross] = day
        crossed |= new_cross
        if np.all(crossed):
            break
    return times[np.isfinite(times)]
