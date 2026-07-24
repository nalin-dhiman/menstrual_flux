import numpy as np
import pytest

from digital_twin.evaluation.metrics import coverage, crps_ensemble, posterior_mass_in_interval, weighted_interval_score
from digital_twin.evaluation.variability import variability_decomposition


def test_crps_is_zero_for_perfect_point_forecast():
    assert crps_ensemble(np.array([3.0, 3.0]), 3.0) == pytest.approx(0)


def test_crps_and_wis_penalize_bad_forecast():
    good = np.array([4, 5, 6])
    bad = np.array([10, 11, 12])
    assert crps_ensemble(good, 5) < crps_ensemble(bad, 5)
    assert weighted_interval_score(good, 5) < weighted_interval_score(bad, 5)


def test_coverage_and_interval_mass():
    samples = np.arange(1, 11)
    assert coverage(samples, 5, 0.8)[0]
    assert posterior_mass_in_interval(samples, 4, 6) == pytest.approx(0.3)


def test_variability_decomposition_identity():
    import pandas as pd
    cycles = pd.DataFrame({
        "participant_id": ["a"] * 4 + ["b"] * 4,
        "cycle_id": list(range(1, 5)) * 2,
        "follicular_duration": [13, 14, 15, 14, 16, 17, 15, 16],
        "luteal_duration": [14, 14, 13, 14, 13, 14, 14, 13],
    })
    cycles["cycle_length"] = cycles.follicular_duration + cycles.luteal_duration
    result = variability_decomposition(cycles, bootstrap_replicates=20, seed=1).set_index("component")
    total = result.loc["total_variance", "estimate_days_squared"]
    stages = result.loc["follicular_variance", "estimate_days_squared"] + result.loc["luteal_variance", "estimate_days_squared"] + result.loc["twice_stage_covariance", "estimate_days_squared"]
    assert stages == pytest.approx(total)
