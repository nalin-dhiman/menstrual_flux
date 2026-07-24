import numpy as np
import pandas as pd
import pytest

from digital_twin.dynamics.coupled_fokker_planck import solve_coupled_cycle
from digital_twin.dynamics.lifespan import simulate_reproductive_lifespan
from digital_twin.gui.analytics import (
    analytic_stage_table,
    build_research_config,
    cycle_duration_table,
    lifespan_scenarios,
    validate_uploaded_frame,
)
from digital_twin.simulation.cohort import simulate_cohort
from digital_twin.simulation.scenarios import SCENARIOS


def test_gui_config_builder_returns_valid_reproducible_config():
    config = build_research_config(
        participants=3,
        cycles=4,
        seed=17,
        scenario="stable",
        follicular_days=15.0,
        luteal_days=13.0,
        progress_noise=0.005,
        between_person_sd=0.10,
        missingness_mechanism="mixed",
        missing_probability=0.08,
    )
    assert config.data.participants == 3
    assert config.data.cycles_per_participant == 4
    assert config.experiment.seed == 17
    assert config.process.follicular_days == 15.0
    assert config.missingness.mechanism == "mixed"


def test_analytic_stage_table_is_finite_and_probabilistic():
    table, metrics = analytic_stage_table(
        drift=1 / 14,
        diffusion=0.00015,
        kappa=0.35,
        sigma_log_speed=0.035,
    )
    assert len(table) == 600
    assert np.isfinite(table.to_numpy()).all()
    assert table["density"].ge(0).all()
    assert table["survival"].between(0, 1).all()
    assert metrics["mean_days"] == 14.0
    assert metrics["peclet"] > 0


def test_cycle_duration_table_uses_consecutive_menses():
    events = pd.DataFrame(
        {
            "participant_id": ["P1", "P1", "P1"],
            "cycle_id": [1, 2, 3],
            "event_type": ["menstruation_onset"] * 3,
            "event_time": ["2026-01-01", "2026-01-30", "2026-02-27"],
        }
    )
    result = cycle_duration_table(events)
    assert result["cycle_length_days"].tolist() == [29.0, 28.0]


def test_upload_validator_accepts_main_example():
    frame = pd.read_csv("data/example_common_observations.csv")
    validation = validate_uploaded_frame(frame)
    assert validation.valid
    assert validation.layout == "common long format"
    assert validation.participants == 1


@pytest.mark.parametrize("scenario", ("mixed",) + SCENARIOS)
def test_every_synthetic_gui_scenario_completes(scenario):
    config = build_research_config(
        participants=1,
        cycles=2,
        seed=101,
        scenario=scenario,
        follicular_days=14.0,
        luteal_days=14.0,
        progress_noise=0.006,
        between_person_sd=0.12,
        missingness_mechanism="mixed",
        missing_probability=0.08,
    )
    cohort = simulate_cohort(config)
    assert cohort.events["event_type"].eq("menstruation_onset").sum() == 3
    assert cohort.observed["participant_id"].nunique() == 1


@pytest.mark.parametrize(
    "mechanism", ("none", "mcar", "mar", "informative", "mixed")
)
def test_every_gui_missingness_option_completes(mechanism):
    config = build_research_config(
        participants=1,
        cycles=2,
        seed=202,
        scenario="stable",
        follicular_days=14.0,
        luteal_days=14.0,
        progress_noise=0.006,
        between_person_sd=0.12,
        missingness_mechanism=mechanism,
        missing_probability=0.08 if mechanism != "none" else 0.0,
    )
    cohort = simulate_cohort(config)
    assert len(cohort.observed) > 0
    assert cohort.observed["is_observed"].dtype == bool


def test_flux_solver_accepts_gui_parameter_extremes():
    result = solve_coupled_cycle(
        follicular_drift=1 / 8.0,
        luteal_drift=1 / 8.0,
        follicular_diffusion=0.001,
        luteal_diffusion=0.001,
        dz=0.02,
        dt=0.01,
        max_time=12.0,
        store_every=10,
    )
    assert np.max(np.abs(result.total_mass - 1.0)) < 1e-8


@pytest.mark.parametrize("scenario_name", tuple(lifespan_scenarios()))
def test_every_lifespan_gui_scenario_completes(scenario_name):
    aggregate = pd.read_csv("data/theory/menstrual_age_aggregate_2024.csv")
    result = simulate_reproductive_lifespan(
        aggregate,
        participants=2,
        seed=303,
        interruption_windows=lifespan_scenarios()[scenario_name],
    )
    assert len(result.participants) == 2
    assert not result.age_summary.empty


def test_upload_validator_accepts_wide_example_and_rejects_invalid_table():
    wide = pd.read_csv("data/example_daily_observations.csv")
    assert validate_uploaded_frame(wide).valid

    invalid = pd.DataFrame({"participant_id": ["P1"], "value": [1.0]})
    report = validate_uploaded_frame(invalid)
    assert not report.valid
    assert report.errors
