from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln, ndtr

from digital_twin.config import ExperimentConfig
from digital_twin.inference.diagnostics import normalize_log_weights
from digital_twin.inference.resampling import effective_sample_size, resample
from digital_twin.simulation.observation_process import expected_signal


@dataclass
class ParticleFilterResult:
    summary: pd.DataFrame
    stage: np.ndarray
    progress: np.ndarray
    log_speed: np.ndarray
    weights: np.ndarray
    ovulation_day: np.ndarray
    next_menses_day: np.ndarray
    mu_log_speed_f: np.ndarray
    mu_log_speed_l: np.ndarray
    final_day: int


def _normal_logpdf(y: float, mean: np.ndarray, sd: float) -> np.ndarray:
    return -0.5 * ((y - mean) / sd) ** 2 - np.log(sd * np.sqrt(2 * np.pi))


def _student_logpdf(y: float, mean: np.ndarray, scale: float, df: float) -> np.ndarray:
    q = ((y - mean) / scale) ** 2
    return gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale) - (df + 1) / 2 * np.log1p(q / df)


def _lognormal_logpdf(y: float, mean: np.ndarray, sd: float) -> np.ndarray:
    if y <= 0:
        return np.full_like(mean, -np.inf)
    return _normal_logpdf(np.log(y), np.log(np.maximum(mean, 1e-6)), sd) - np.log(y)


def _ordinal_loglik(y: float, expected: np.ndarray, sd: float, maximum: int = 4) -> np.ndarray:
    """Ordered-probit mass for categories 0..maximum."""
    category = int(np.clip(round(y), 0, maximum))
    lower = -np.inf if category == 0 else category - 0.5
    upper = np.inf if category == maximum else category + 0.5
    probability = ndtr((upper - expected) / sd) - ndtr((lower - expected) / sd)
    return np.log(np.maximum(probability, 1e-12))


def _daily_wide(observations: pd.DataFrame) -> pd.DataFrame:
    observed_mask = observations["is_observed"].astype(bool) if "is_observed" in observations else pd.Series(True, index=observations.index)
    observed = observations[observed_mask & observations["value"].notna()].copy()
    if observed.empty:
        return pd.DataFrame()
    observed["date"] = pd.to_datetime(observed["event_time"]).dt.normalize()
    start = pd.to_datetime(observations["event_time"]).min().normalize()
    end = pd.to_datetime(observations["event_time"]).max().normalize()
    calendar = pd.date_range(start, end, freq="D")
    return observed.pivot_table(index="date", columns="signal_name", values="value", aggfunc="last").reindex(calendar).sort_index()


def run_particle_filter(
    observations: pd.DataFrame,
    cfg: ExperimentConfig,
    seed_offset: int = 0,
    prior_follicular_days: float | None = None,
    prior_luteal_days: float | None = None,
) -> ParticleFilterResult:
    daily = _daily_wide(observations)
    if daily.empty:
        raise ValueError("at least one observed measurement is required")
    rng = np.random.default_rng(cfg.experiment.seed + 1009 + seed_offset)
    n = cfg.inference.particles
    p = cfg.process
    stage = np.zeros(n, dtype=np.int8)
    progress = np.clip(rng.normal(0.0, 0.025, n), 0, 0.08)
    f_days = float(prior_follicular_days or p.follicular_days)
    l_days = float(prior_luteal_days or p.luteal_days)
    mu_f = np.log(1 / f_days) + rng.normal(0, p.between_person_log_speed_sd, n)
    mu_l = np.log(1 / l_days) + rng.normal(0, p.between_person_log_speed_sd * 0.65, n)
    log_speed = mu_f.copy()
    weights = np.full(n, 1 / n)
    ovulation_day = np.full(n, np.nan)
    next_menses_day = np.full(n, np.nan)
    rows: list[dict[str, float | int | bool]] = []
    baseline_values = {
        "temperature": float(daily["temperature"].dropna().head(5).median()) if "temperature" in daily and daily["temperature"].notna().any() else 36.45,
        "rhr": float(daily["resting_heart_rate"].dropna().head(5).median()) if "resting_heart_rate" in daily and daily["resting_heart_rate"].notna().any() else 62.0,
        "hrv_log": float(np.log(daily["hrv"].dropna().head(5).median())) if "hrv" in daily and daily["hrv"].notna().any() else np.log(48.0),
    }
    for day_index, (_, row) in enumerate(daily.iterrows(), start=1):
        follicular = stage == 0
        mu = np.where(follicular, mu_f, mu_l)
        kappa = np.where(follicular, p.kappa_log_speed_f, p.kappa_log_speed_l)
        sigma_ell = np.where(follicular, p.sigma_log_speed_f, p.sigma_log_speed_l)
        sigma_z = np.where(follicular, p.sigma_progress_f, p.sigma_progress_l)
        log_speed += kappa * (mu - log_speed) + sigma_ell * rng.normal(size=n)
        progress += np.maximum(np.exp(log_speed) + sigma_z * rng.normal(size=n), 0.001)
        cross_f = (stage == 0) & (progress >= 1)
        ovulation_day[cross_f & np.isnan(ovulation_day)] = day_index
        stage[cross_f] = 1
        progress[cross_f] -= 1
        log_speed[cross_f] = mu_l[cross_f]
        cross_l = (stage == 1) & (progress >= 1)
        next_menses_day[cross_l & np.isnan(next_menses_day)] = day_index
        stage[cross_l] = 0
        progress[cross_l] -= 1
        log_speed[cross_l] = mu_f[cross_l]

        loglik = np.zeros(n)
        for signal, value in row.items():
            if not np.isfinite(value):
                continue
            mean = expected_signal(signal, stage, progress, baseline_values)
            likelihood_weight = 1.0
            if signal in {"temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"}:
                likelihood_weight = cfg.observation.wearable_likelihood_weight
            elif signal in {"lh", "e3g", "pdg", "estradiol", "progesterone"}:
                likelihood_weight = cfg.observation.hormone_likelihood_weight
            elif signal in {"bleeding", "symptom_severity"}:
                likelihood_weight = cfg.observation.diary_likelihood_weight
            if signal in {"lh", "e3g", "pdg", "estradiol", "progesterone", "hrv"}:
                sd = cfg.observation.log_hrv_sd if signal == "hrv" else cfg.observation.log_hormone_sd
                loglik += likelihood_weight * _lognormal_logpdf(float(value), mean, sd)
            elif signal == "temperature":
                loglik += likelihood_weight * _student_logpdf(float(value), mean, cfg.observation.temperature_sd, cfg.observation.heavy_tail_df)
            elif signal == "resting_heart_rate":
                loglik += likelihood_weight * _normal_logpdf(float(value), mean, cfg.observation.rhr_sd)
            elif signal == "sleep_duration":
                loglik += likelihood_weight * _normal_logpdf(float(value), mean, cfg.observation.sleep_duration_sd)
            elif signal == "sleep_efficiency":
                loglik += likelihood_weight * _normal_logpdf(float(value), mean, cfg.observation.sleep_efficiency_sd)
            elif signal in {"bleeding", "symptom_severity"}:
                loglik += likelihood_weight * _ordinal_loglik(float(value), mean, cfg.observation.ordinal_latent_sd)
        weights, log_normalizer = normalize_log_weights(np.log(np.maximum(weights, 1e-300)) + loglik)
        ess = effective_sample_size(weights)
        resampled = ess < cfg.inference.ess_threshold * n
        rows.append({
            "day": day_index,
            "p_follicular": float(weights[stage == 0].sum()),
            "p_luteal": float(weights[stage == 1].sum()),
            "mean_progress": float(np.sum(weights * progress)),
            "mean_log_speed": float(np.sum(weights * log_speed)),
            "effective_sample_size": ess,
            "incremental_log_normalizer": log_normalizer,
            "resampled": resampled,
        })
        if resampled:
            index = resample(weights, cfg.inference.resampling, rng)
            stage = stage[index]
            progress = progress[index]
            log_speed = log_speed[index]
            mu_f = mu_f[index]
            mu_l = mu_l[index]
            ovulation_day = ovulation_day[index]
            next_menses_day = next_menses_day[index]
            weights.fill(1 / n)
    return ParticleFilterResult(pd.DataFrame(rows), stage, progress, log_speed, weights, ovulation_day, next_menses_day, mu_f, mu_l, len(daily))
