from dataclasses import replace

import numpy as np
import pandas as pd

from digital_twin.benchmark import BenchmarkConfig, _eligibility_ledger, _participant_split, load_benchmark_config
from digital_twin.config import ExperimentConfig
from digital_twin.inference.particle_filter import _daily_wide
from digital_twin.simulation import simulate_cohort


def test_locked_participant_splits_are_disjoint_and_deterministic():
    participants = np.array([f"P{i:03d}" for i in range(20)])
    first = _participant_split(participants, 11, 0.5, 0.25)
    second = _participant_split(participants, 11, 0.5, 0.25)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"population_train", "calibration", "locked_test"}
    assert first["participant_id"].is_unique


def test_fast_and_full_benchmark_configs_are_valid():
    fast = load_benchmark_config("configs/benchmarks/fast.yaml")
    full = load_benchmark_config("configs/benchmarks/full.yaml")
    assert set(fast.safety_scenarios) <= set(fast.scenarios)
    assert len(full.seeds) == 20
    assert full.resume
    validation = load_benchmark_config("configs/benchmarks/v04_validation.yaml")
    assert set(validation.seeds).isdisjoint(full.seeds)
    assert set(validation.seeds).isdisjoint(fast.seeds)


def test_missing_calendar_days_are_preserved():
    rows = pd.DataFrame({
        "event_time": pd.date_range("2026-01-01", periods=3),
        "signal_name": ["temperature"] * 3,
        "value": [36.4, np.nan, 36.5],
        "is_observed": [True, False, True],
    })
    daily = _daily_wide(rows)
    assert len(daily) == 3
    assert np.isnan(daily.iloc[1]["temperature"])


def test_anovulatory_like_suppresses_transition_hormones_for_cycle():
    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        data=replace(cfg.data, participants=1, cycles_per_participant=2, scenario="anovulatory_like"),
        process=replace(cfg.process, anovulatory_like_probability=1.0),
        missingness=replace(cfg.missingness, mechanism="none", hormone_schedule_days=1),
    )
    cohort = simulate_cohort(cfg)
    lh = cohort.raw_observations.loc[cohort.raw_observations["signal_name"] == "lh", "value"]
    assert float(lh.max()) < 10
    assert cohort.truth["anovulatory_like_cycle"].any()


def test_stalled_transition_delays_first_passage():
    cfg = ExperimentConfig()
    cfg = replace(cfg, data=replace(cfg.data, participants=1, cycles_per_participant=2, scenario="stalled_transition"))
    cohort = simulate_cohort(cfg)
    events = cohort.events[cohort.events["event_type"] == "latent_ovulation_transition"]
    onset = cohort.events[cohort.events["event_type"] == "menstruation_onset"].iloc[0]["event_time"]
    assert (pd.Timestamp(events.iloc[0]["event_time"]) - pd.Timestamp(onset)).days >= 40
    menses = cohort.events[cohort.events["event_type"] == "menstruation_onset"]
    assert len(menses) == 3


def test_eligibility_ledger_accounts_for_every_planned_forecast():
    cfg = load_benchmark_config("configs/benchmarks/fast.yaml")
    cfg = BenchmarkConfig(**{**cfg.__dict__, "cycles_per_participant": 5, "adaptation_cycles": 2})
    cycles = pd.DataFrame({
        "participant_id": ["P001"] * 3,
        "cycle_id": [3, 4, 5],
        "cycle_length": [28.0, np.nan, 9.0],
        "follicular_duration": [14.0, np.nan, 5.0],
        "luteal_duration": [14.0, np.nan, 4.0],
    })
    ledger = _eligibility_ledger(cycles, {"P001"}, cfg, 1, "test")
    assert len(ledger) == 3
    assert ledger["exclusion_reason"].tolist() == ["eligible", "administratively_censored", "event_before_issue_day"]
