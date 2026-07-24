import numpy as np

from digital_twin.inference.personalization import StageSpeedPosterior


def test_stage_speed_posterior_uses_observation_durations_and_reports_reliability():
    posterior = StageSpeedPosterior.from_durations(
        np.array([11.5, 12.0, 12.5, 12.0]),
        population_log_speed=-np.log(14.0),
        between_person_sd=0.15,
    )
    assert posterior.observed_cycles == 4
    assert 0 < posterior.reliability < 1
    assert posterior.mean_log_speed > -np.log(14.0)
