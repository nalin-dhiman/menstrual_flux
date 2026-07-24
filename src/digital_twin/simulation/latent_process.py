from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from digital_twin.config import ExperimentConfig


class Stage(IntEnum):
    FOLLICULAR = 0
    LUTEAL = 1


@dataclass(frozen=True)
class ParticipantParameters:
    participant_id: str
    mu_log_speed_f: float
    mu_log_speed_l: float
    temperature_baseline: float
    rhr_baseline: float
    hrv_baseline_log: float
    temperature_amplitude: float
    hormone_amplitude: float
    scenario: str


@dataclass
class LatentSimulation:
    states: pd.DataFrame
    events: pd.DataFrame
    parameters: pd.DataFrame


def draw_participant_parameters(
    participant_id: str,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    scenario: str = "regular",
) -> ParticipantParameters:
    p = cfg.process
    f_shift = rng.normal(0, p.between_person_log_speed_sd)
    l_shift = rng.normal(0, p.between_person_log_speed_sd * 0.65)
    if scenario == "short_cycle":
        f_shift += 0.20
    elif scenario == "long_cycle":
        f_shift -= 0.20
    elif scenario in {"highly_variable", "follicular_dominant", "luteal_dominant"}:
        f_shift += rng.normal(0, 0.08)
        l_shift += rng.normal(0, 0.06)
    return ParticipantParameters(
        participant_id=participant_id,
        mu_log_speed_f=float(np.log(1 / p.follicular_days) + f_shift),
        mu_log_speed_l=float(np.log(1 / p.luteal_days) + l_shift),
        temperature_baseline=float(rng.normal(36.45, 0.18)),
        rhr_baseline=float(rng.normal(62.0, 5.0)),
        hrv_baseline_log=float(rng.normal(np.log(48.0), 0.22)),
        temperature_amplitude=float(np.clip(rng.normal(0.24, 0.055), 0.10, 0.42)),
        hormone_amplitude=float(np.clip(rng.lognormal(0, 0.18), 0.55, 1.65)),
        scenario=scenario,
    )


def simulate_latent_participant(
    parameters: ParticipantParameters,
    n_cycles: int,
    start_date: str,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> LatentSimulation:
    """Euler--Maruyama simulation of the reduced two-stage first-passage model."""
    p = cfg.process
    dt = p.dt_days
    stage = Stage.FOLLICULAR
    z = 0.0
    ell = parameters.mu_log_speed_f
    cycle_id = 1
    day = 0
    delta_f = 0.0
    delta_l = 0.0
    stage_elapsed = 0.0
    anovulatory_like_cycle = bool(parameters.scenario == "anovulatory_like" or rng.random() < p.anovulatory_like_probability)
    stalled_transition_cycle = parameters.scenario == "stalled_transition"
    states: list[dict[str, object]] = []
    events: list[dict[str, object]] = [{
        "participant_id": parameters.participant_id,
        "cycle_id": 1,
        "event_type": "menstruation_onset",
        "event_time": pd.Timestamp(start_date),
        "event_time_lower": pd.Timestamp(start_date),
        "event_time_upper": pd.Timestamp(start_date),
        "event_source": "synthetic_truth",
        "certainty": "exact_simulated",
        "availability_time": pd.Timestamp(start_date),
    }]
    # This is a fail-loud numerical guard, not an administrative study cutoff.
    # Earlier versions used a nominal-duration cap and silently dropped long
    # later cycles, which selected easier cases into evaluation.
    max_days = int(n_cycles * p.max_cycle_days)
    while cycle_id <= n_cycles and day < max_days:
        timestamp = pd.Timestamp(start_date) + pd.Timedelta(days=day)
        mu = (parameters.mu_log_speed_f + delta_f) if stage == Stage.FOLLICULAR else (parameters.mu_log_speed_l + delta_l)
        kappa = p.kappa_log_speed_f if stage == Stage.FOLLICULAR else p.kappa_log_speed_l
        sigma_ell = p.sigma_log_speed_f if stage == Stage.FOLLICULAR else p.sigma_log_speed_l
        sigma_z = p.sigma_progress_f if stage == Stage.FOLLICULAR else p.sigma_progress_l
        ell += kappa * (mu - ell) * dt + sigma_ell * np.sqrt(dt) * rng.normal()
        increment = max(np.exp(ell) * dt + sigma_z * np.sqrt(dt) * rng.normal(), 0.001)
        z_before = z
        z += increment
        stage_elapsed += dt
        if stage == Stage.FOLLICULAR and parameters.scenario == "stalled_transition" and stage_elapsed < p.stalled_follicular_days:
            z = min(z, 0.92)
        transition = "none"
        transition_fraction = np.nan
        anovulatory_like = False
        if z >= 1.0:
            transition_fraction = float(np.clip((1.0 - z_before) / increment, 0.0, 1.0))
            event_time = timestamp + pd.Timedelta(days=transition_fraction)
            if stage == Stage.FOLLICULAR:
                transition = "ovulation"
                anovulatory_like = anovulatory_like_cycle
                events.append({
                    "participant_id": parameters.participant_id,
                    "cycle_id": cycle_id,
                    "event_type": "latent_ovulation_transition",
                    "event_time": event_time,
                    "event_time_lower": event_time,
                    "event_time_upper": event_time,
                    "event_source": "synthetic_truth",
                    "certainty": "anovulatory_like_regime" if anovulatory_like else "exact_simulated",
                    "availability_time": event_time,
                })
                stage = Stage.LUTEAL
                stage_elapsed = 0.0
                ell = parameters.mu_log_speed_l + delta_l
            else:
                transition = "menstruation"
                next_cycle = cycle_id + 1
                events.append({
                    "participant_id": parameters.participant_id,
                    "cycle_id": next_cycle,
                    "event_type": "menstruation_onset",
                    "event_time": event_time,
                    "event_time_lower": event_time,
                    "event_time_upper": event_time,
                    "event_source": "synthetic_truth",
                    "certainty": "exact_simulated",
                    "availability_time": event_time,
                })
                cycle_id = next_cycle
                stage = Stage.FOLLICULAR
                stage_elapsed = 0.0
                anovulatory_like_cycle = bool(parameters.scenario == "anovulatory_like" or rng.random() < p.anovulatory_like_probability)
                delta_f = p.cycle_effect_rho_f * delta_f + rng.normal(0, p.cycle_effect_sd_f)
                delta_l = p.cycle_effect_rho_l * delta_l + rng.normal(0, p.cycle_effect_sd_l)
                drift = rng.normal(0, p.drift_per_cycle_sd)
                delta_f += drift
                delta_l += drift
                if parameters.scenario == "follicular_dominant":
                    delta_f += rng.normal(0, p.cycle_effect_sd_f)
                if parameters.scenario == "luteal_dominant":
                    delta_l += rng.normal(0, p.cycle_effect_sd_l * 2)
                if parameters.scenario == "temporary_disruption" and cycle_id == max(2, n_cycles // 2):
                    delta_f -= 0.25
                if parameters.scenario == "regime_switch" and cycle_id >= max(2, n_cycles // 2):
                    delta_f -= 0.12
                ell = parameters.mu_log_speed_f + delta_f
            z = max(z - 1.0, 0.0)
        states.append({
            "participant_id": parameters.participant_id,
            "cycle_id": min(cycle_id, n_cycles),
            "day_in_study": day + 1,
            "event_time": timestamp,
            "stage": "F" if stage == Stage.FOLLICULAR else "L",
            "stage_code": int(stage),
            "progress": float(z),
            "log_speed": float(ell),
            "speed_per_day": float(np.exp(ell)),
            "transition": transition,
            "transition_fraction": transition_fraction,
            "cycle_effect_f": float(delta_f),
            "cycle_effect_l": float(delta_l),
            "anovulatory_like": anovulatory_like,
            "anovulatory_like_cycle": anovulatory_like_cycle,
            "stalled_transition_cycle": stalled_transition_cycle,
            "stage_elapsed_days": stage_elapsed,
        })
        day += 1
    if cycle_id <= n_cycles:
        raise RuntimeError(
            f"simulation did not complete {n_cycles} cycles for {parameters.participant_id} "
            f"within {max_days} days; increase process.max_cycle_days or inspect the process"
        )
    state_df = pd.DataFrame(states)
    state_df = state_df[state_df["cycle_id"] <= n_cycles].reset_index(drop=True)
    param_df = pd.DataFrame([parameters.__dict__])
    return LatentSimulation(state_df, pd.DataFrame(events), param_df)
