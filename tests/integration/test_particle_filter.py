from dataclasses import replace

import numpy as np

from digital_twin.config import ExperimentConfig
from digital_twin.inference import forecast_events, run_particle_filter
from digital_twin.simulation import simulate_cohort
from digital_twin.simulation.missingness import available_snapshot


def test_filter_and_first_passage_forecasts_execute():
    cfg = ExperimentConfig()
    cfg = replace(cfg, data=replace(cfg.data, participants=1, cycles_per_participant=3), missingness=replace(cfg.missingness, mechanism="none", mcar_probability=0, mar_probability=0, informative_probability=0), inference=replace(cfg.inference, particles=180, forecast_samples=180))
    cohort = simulate_cohort(cfg)
    subset = cohort.observed[cohort.observed.cycle_id.eq("P0001-C002")]
    issue = subset.event_time.min() + np.timedelta64(10, "D")
    subset = available_snapshot(subset, issue)
    result = run_particle_filter(subset, cfg)
    forecast = forecast_events(result, cfg)
    assert np.isclose(result.summary.p_follicular + result.summary.p_luteal, 1).all()
    assert np.isfinite(forecast.ovulation_samples).mean() > 0.95
    assert np.isfinite(forecast.next_menses_samples).mean() > 0.95
    assert np.nanmedian(forecast.ovulation_samples) < np.nanmedian(forecast.next_menses_samples)
