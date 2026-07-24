from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ModelConfig
from .math_utils import TWO_PI, wrap_phase
from .observation_models import expected_observations


def simulate_participant(
    participant_id: str,
    n_days: int,
    cfg: ModelConfig,
    rng: np.random.Generator | None = None,
    mean_cycle_days: float | None = None,
) -> pd.DataFrame:
    rng = rng or np.random.default_rng(cfg.seed)
    p = cfg.process
    o = cfg.observation
    mean_cycle = float(mean_cycle_days or p.mean_cycle_days)
    mu_log_omega = np.log(TWO_PI / mean_cycle)

    theta_unwrapped = rng.uniform(0.0, TWO_PI)
    log_omega = rng.normal(mu_log_omega, 0.03)
    amplitude = rng.normal(0.0, 0.10)
    prev_cycle_index = int(np.floor(theta_unwrapped / TWO_PI))

    rows: list[dict[str, float | int | str | bool]] = []
    for day in range(n_days):
        sleep_hours = float(np.clip(rng.normal(7.1, 0.85), 3.5, 10.5))
        stress = int(rng.random() < 0.10)
        illness = int(rng.random() < 0.025)

        log_omega += p.log_omega_reversion * (mu_log_omega - log_omega) + p.log_omega_sd * rng.normal()
        amplitude += (
            -p.amplitude_recovery * amplitude
            + p.stress_effect_on_amplitude * stress
            + p.illness_effect_on_amplitude * illness
            + p.amplitude_sd * rng.normal()
        )
        omega = float(np.exp(log_omega))
        theta_increment = (
            omega
            + p.sleep_effect_on_phase * (sleep_hours - 7.1)
            + p.amplitude_to_phase * amplitude
            + p.phase_diffusion_sd * rng.normal()
        )
        theta_unwrapped += max(theta_increment, 0.01)
        cycle_index = int(np.floor(theta_unwrapped / TWO_PI))
        onset_event = cycle_index > prev_cycle_index
        prev_cycle_index = cycle_index
        theta = float(wrap_phase(theta_unwrapped))

        exp_obs = expected_observations(np.array([theta]), np.array([amplitude]), cfg)
        temperature = float(exp_obs["temperature_c"][0] + o.temperature_sd_c * rng.normal())
        rhr = float(exp_obs["rhr_bpm"][0] + o.rhr_sd_bpm * rng.normal())
        lh = float(np.exp(rng.normal(np.log(exp_obs["lh"][0]), o.lh_log_sd)))
        e3g = float(np.exp(rng.normal(np.log(exp_obs["e3g"][0]), o.e3g_log_sd)))
        pdg = float(np.exp(rng.normal(np.log(exp_obs["pdg"][0]), o.pdg_log_sd)))
        bleeding = bool(rng.random() < exp_obs["bleeding_prob"][0] or onset_event)

        row: dict[str, float | int | str | bool] = {
            "participant_id": participant_id,
            "day_in_study": day + 1,
            "timestamp_local": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).isoformat(),
            "sleep_hours": sleep_hours,
            "stress_event": stress,
            "illness_event": illness,
            "bleeding": bleeding,
            "temperature_c": temperature,
            "rhr_bpm": rhr,
            "lh": lh,
            "e3g": e3g,
            "pdg": pdg,
            "true_phase_rad": theta,
            "true_phase_fraction": theta / TWO_PI,
            "true_log_omega": log_omega,
            "true_amplitude": amplitude,
            "true_cycle_index": cycle_index,
            "true_onset_event": onset_event,
        }
        # Apply independent missingness as a simple prototype. Real work should model
        # device- and behavior-dependent missingness explicitly.
        for col in ["temperature_c", "rhr_bpm", "lh", "e3g", "pdg", "sleep_hours"]:
            if rng.random() < o.missing_probability:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_cohort(
    n_participants: int,
    n_days: int,
    cfg: ModelConfig,
    cycle_day_means: Iterable[float] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    means = list(cycle_day_means) if cycle_day_means is not None else list(rng.normal(cfg.process.mean_cycle_days, 2.0, n_participants))
    if len(means) != n_participants:
        raise ValueError("cycle_day_means length must equal n_participants")
    frames = [
        simulate_participant(f"P{i+1:03d}", n_days, cfg, rng=rng, mean_cycle_days=max(20.0, float(means[i])))
        for i in range(n_participants)
    ]
    return pd.concat(frames, ignore_index=True)
