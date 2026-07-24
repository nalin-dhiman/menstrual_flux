import numpy as np

from digital_twin.baselines.renewal import HierarchicalRenewal, calendar_forecast, rolling_mean_forecast


def test_calendar_and_rolling_baselines():
    assert calendar_forecast(100, 28) == 128
    assert rolling_mean_forecast(100, np.array([25, 27, 29]), 2) == 128


def test_hierarchical_renewal_partially_pools():
    model = HierarchicalRenewal(prior_strength=3).fit(np.array([27, 28, 29, 30]))
    participant_mean, participant_sd = model.participant_parameters(np.array([35]))
    assert 28 < participant_mean < 35
    assert participant_sd > 0
