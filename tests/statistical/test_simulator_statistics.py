from dataclasses import replace

import numpy as np

from digital_twin.config import ExperimentConfig
from digital_twin.simulation import simulate_cohort


def test_simulator_stage_durations_are_plausible_and_transitions_alternate():
    cfg = ExperimentConfig()
    cfg = replace(cfg, data=replace(cfg.data, participants=12, cycles_per_participant=5, scenario="stable"), process=replace(cfg.process, between_person_log_speed_sd=0.02, cycle_effect_sd_f=0.02, cycle_effect_sd_l=0.01, anovulatory_like_probability=0))
    cohort = simulate_cohort(cfg)
    events = cohort.events.sort_values(["participant_id", "event_time"])
    for _, group in events.groupby("participant_id"):
        types = group["event_type"].tolist()
        assert all(a != b for a, b in zip(types, types[1:]))
    menses = events[events.event_type.eq("menstruation_onset")]
    lengths = menses.groupby("participant_id")["event_time"].apply(lambda x: np.diff(np.array(x, dtype="datetime64[s]")).astype("timedelta64[s]").astype(float) / 86400)
    values = np.concatenate(lengths.to_list())
    assert 25 < values.mean() < 31


def test_truth_and_observations_are_separate():
    cfg = ExperimentConfig()
    cfg = replace(cfg, data=replace(cfg.data, participants=2, cycles_per_participant=3))
    cohort = simulate_cohort(cfg)
    assert "progress" in cohort.truth and "progress" not in cohort.observed
    assert "signal_name" in cohort.observed and "signal_name" not in cohort.truth
