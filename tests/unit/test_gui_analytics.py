import numpy as np
import pandas as pd

from digital_twin.gui.analytics import (
    analytic_stage_table,
    build_research_config,
    cycle_duration_table,
    validate_uploaded_frame,
)


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
