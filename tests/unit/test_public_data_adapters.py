from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from digital_twin.data.schemas import (
    validate_events,
    validate_observations,
    validate_participants,
)
from digital_twin.data_adapters.salzburg_hormones import SalzburgHormoneAdapter
from digital_twin.data_adapters.soochow_heart_rate import (
    SoochowHeartRateAdapter,
    _menstruation_onsets,
)
from digital_twin.real_data.open_benchmark import (
    _apply_calibrator,
    _fit_ridge,
    _model_observations,
)


def _salzburg_fixture(root: Path) -> Path:
    target = root / "optimal_cycle_prediction" / "data" / "raw_data"
    target.mkdir(parents=True)
    rows = []
    for day in range(1, 22):
        rows.append(
            {
                "VP": 1,
                "Period": float(day == 1),
                "LHOvulatoryKit": float(day == 12),
                "Estradiol": 1.0 + 0.02 * day,
                "Progesterone": 10.0 + day,
                "cycle": 1.0,
                "cycle_day_forward": float(day),
                "cycle_day_backward": float(day - 22),
                "expected_backward_count": float(day - 22),
                "expected_cycle_length": 28.0,
            }
        )
    rows.append(
        {
            "VP": 1,
            "Period": 1.0,
            "LHOvulatoryKit": 0.0,
            "Estradiol": np.nan,
            "Progesterone": np.nan,
            "cycle": 2.0,
            "cycle_day_forward": 1.0,
            "cycle_day_backward": np.nan,
            "expected_backward_count": -27.0,
            "expected_cycle_length": 28.0,
        }
    )
    pd.DataFrame(rows).to_csv(
        target / "DD_Females_Hormones_anonymized.csv", index=False
    )
    return root


def test_salzburg_conversion_excludes_future_features(tmp_path: Path) -> None:
    source = _salzburg_fixture(tmp_path / "release")
    adapter = SalzburgHormoneAdapter()
    inspection = adapter.inspect(source)
    assert inspection["complete_cycle_intervals_from_raw"] == 1
    result = adapter.convert(source)
    assert len(result.participants) == 1
    assert len(result.events) == 2
    assert len(result.reference_intervals) == 1
    assert result.cycles["eligible_for_primary_evaluation"].sum() == 1
    assert "cycle_day_backward" not in result.daily_observations.columns
    assert "expected_backward_count" not in result.daily_observations.columns
    assert not validate_participants(result.participants)
    assert not validate_observations(result.daily_observations)
    assert not validate_events(result.events)
    manifest = adapter.write(result, tmp_path / "curated", file_format="csv")
    assert manifest["access_classification"] == "public_deidentified_participant_data"
    assert manifest["shareable"] is False


def test_soochow_event_rule_and_raw_daily_aggregation(tmp_path: Path) -> None:
    states = np.ones(70)
    states[2:7] = 2
    states[10:12] = 2
    states[40:44] = 2
    assert _menstruation_onsets(states).tolist() == [2, 40]

    start = pd.Timestamp("2020-01-01", tz="UTC")
    timestamps = [
        (start + pd.Timedelta(hours=hour)).timestamp() * 1000
        for hour in range(24)
    ]
    values = np.c_[timestamps, np.linspace(55, 78, 24)]
    path = tmp_path / "data_1.mat"
    savemat(path, {"D": values})
    daily = SoochowHeartRateAdapter._load_raw_features(path)
    assert len(daily) in {1, 2}  # UTC+8 moves late UTC records to the next local date.
    assert daily["daily_record_count"].sum() == 24
    assert daily["daily_heart_rate_mean"].notna().all()


def test_ridge_uses_only_supplied_past_case_features() -> None:
    cases = []
    for index in range(8):
        observations = pd.DataFrame(
            {
                "signal_name": ["bleeding_reported"] * 3,
                "value": [1.0, 0.0, 0.0],
                "day_in_study": [1, 2, 3],
            }
        )
        cases.append(
            {
                "history": np.array([26.0 + index]),
                "target": 27.0 + index,
                "observations": observations,
                "cycle_start_day": 1,
                "issue_day": 3,
            }
        )
    model = _fit_ridge(cases, ["bleeding_reported"], penalty=10.0)
    assert model["available"]
    assert model["training_cases"] == 8
    samples = np.array([20.0, 25.0, 30.0])
    calibrated = _apply_calibrator(
        samples, {"bias_days": 2.0, "spread_scale": 1.5}
    )
    assert np.median(calibrated) == 27.0


def test_model_observations_normalize_mixed_daily_timestamps() -> None:
    case = {
        "observations": pd.DataFrame(
            {
                "event_time": ["2020-01-01", "2020-01-02 12:00:00"],
                "signal_name": [
                    "night_heart_rate_median",
                    "bleeding_reported",
                ],
                "value": [61.0, 1.0],
            }
        )
    }
    result = _model_observations(
        case, {"resting_heart_rate", "bleeding"}, profiles={}
    )
    assert result["event_time"].dt.hour.eq(0).all()
