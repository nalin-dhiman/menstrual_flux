from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.baselines.hsmm import HSMMBaseline
from digital_twin.real_data.mcphases_benchmark import _allocate_stratum


def test_outcome_blind_stratum_allocation_is_deterministic_and_complete() -> None:
    participant_ids = [f"p-{index}" for index in range(20)]
    first = _allocate_stratum(
        participant_ids,
        train_fraction=0.60,
        calibration_fraction=0.20,
        rng=np.random.default_rng(123),
    )
    second = _allocate_stratum(
        participant_ids,
        train_fraction=0.60,
        calibration_fraction=0.20,
        rng=np.random.default_rng(123),
    )
    assert first == second
    assert set(first) == set(participant_ids)
    assert list(first.values()).count("population_train") == 12
    assert list(first.values()).count("calibration") == 4
    assert list(first.values()).count("locked_test") == 4


def test_hsmm_transition_samples_are_forward_or_at_issue_day() -> None:
    daily = pd.DataFrame(
        {
            "bleeding": [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temperature": [36.4, 36.4, 36.42, 36.43, 36.44, 36.45, 36.46],
            "lh": [3.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0],
        }
    )
    samples = HSMMBaseline().predictive_transition_samples(
        daily,
        n=500,
        rng=np.random.default_rng(77),
    )
    assert np.isfinite(samples).all()
    assert (samples >= len(daily)).all()
