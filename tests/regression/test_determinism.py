from dataclasses import replace

import pandas as pd

from digital_twin.config import ExperimentConfig
from digital_twin.simulation import simulate_cohort


def test_simulation_is_deterministic():
    cfg = ExperimentConfig()
    cfg = replace(cfg, data=replace(cfg.data, participants=1, cycles_per_participant=2))
    first = simulate_cohort(cfg)
    second = simulate_cohort(cfg)
    pd.testing.assert_frame_equal(first.truth, second.truth)
    pd.testing.assert_frame_equal(first.observed, second.observed)
