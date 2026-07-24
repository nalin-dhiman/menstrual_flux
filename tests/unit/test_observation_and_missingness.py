from dataclasses import replace

import numpy as np
import pandas as pd

from digital_twin.config import ExperimentConfig
from digital_twin.simulation.missingness import apply_missingness, available_snapshot
from digital_twin.simulation.observation_process import expected_signal


def test_luteal_temperature_and_pdg_are_higher():
    f = np.array([0]); l = np.array([1]); z = np.array([0.5])
    assert expected_signal("temperature", l, z)[0] > expected_signal("temperature", f, z)[0]
    assert expected_signal("pdg", l, z)[0] > expected_signal("pdg", f, z)[0]


def test_lh_is_localized_near_follicular_boundary():
    stage = np.array([0, 0])
    values = expected_signal("lh", stage, np.array([0.3, 0.96]))
    assert values[1] > values[0] * 5


def test_missingness_does_not_use_zero_and_leakage_snapshot_is_safe():
    rows = []
    for day in range(6):
        rows.append({"participant_id": "P1", "signal_name": "temperature", "value": 36.5, "event_time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day), "availability_time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=8), "report_time": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day, hours=8), "is_observed": True, "missingness_reason": ""})
    cfg = ExperimentConfig()
    cfg = replace(cfg, missingness=replace(cfg.missingness, mechanism="mcar", mcar_probability=1.0))
    missing = apply_missingness(pd.DataFrame(rows), cfg, np.random.default_rng(1))
    assert missing["value"].isna().all()
    assert missing["missingness_reason"].ne("").all()
    original = pd.DataFrame(rows)
    issue = pd.Timestamp("2026-01-03 12:00")
    snap = available_snapshot(original, issue)
    assert (snap["availability_time"] <= issue).all()
