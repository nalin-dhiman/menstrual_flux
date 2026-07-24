from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from digital_twin.config import ExperimentConfig
from digital_twin.simulation.latent_process import draw_participant_parameters, simulate_latent_participant
from digital_twin.simulation.missingness import apply_missingness
from digital_twin.simulation.observation_process import simulate_observations
from digital_twin.simulation.scenarios import scenario_for_index


@dataclass
class SyntheticCohort:
    raw_observations: pd.DataFrame
    observed: pd.DataFrame
    truth: pd.DataFrame
    events: pd.DataFrame
    parameters: pd.DataFrame
    participants: pd.DataFrame


def simulate_cohort(cfg: ExperimentConfig) -> SyntheticCohort:
    rng = np.random.default_rng(cfg.experiment.seed)
    raw_frames: list[pd.DataFrame] = []
    observed_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    participant_rows: list[dict[str, object]] = []
    for idx in range(cfg.data.participants):
        participant_id = f"P{idx + 1:04d}"
        scenario = scenario_for_index(idx, cfg.data.scenario)
        parameters = draw_participant_parameters(participant_id, cfg, rng, scenario)
        latent = simulate_latent_participant(parameters, cfg.data.cycles_per_participant, cfg.data.start_date, cfg, rng)
        raw = simulate_observations(latent.states, latent.events, parameters, cfg, rng)
        observed = apply_missingness(raw, cfg, rng)
        raw_frames.append(raw)
        observed_frames.append(observed)
        truth_frames.append(latent.states)
        event_frames.append(latent.events)
        parameter_frames.append(latent.parameters)
        participant_rows.append({
            "participant_id": participant_id,
            "source_dataset": "synthetic_milestone_1",
            "age": np.nan,
            "age_unit": "years",
            "reproductive_stage": "synthetic_natural_cycling",
            "contraceptive_status": "synthetic_none",
            "medication_status": "not_modelled",
            "device_type": "synthetic_wearable_v1",
            "timezone": cfg.data.timezone,
            "enrollment_date": cfg.data.start_date,
            "withdrawal_date": pd.NaT,
            "available_covariates": "scenario",
            "scenario": scenario,
        })
    return SyntheticCohort(
        raw_observations=pd.concat(raw_frames, ignore_index=True),
        observed=pd.concat(observed_frames, ignore_index=True),
        truth=pd.concat(truth_frames, ignore_index=True),
        events=pd.concat(event_frames, ignore_index=True),
        parameters=pd.concat(parameter_frames, ignore_index=True),
        participants=pd.DataFrame(participant_rows),
    )
