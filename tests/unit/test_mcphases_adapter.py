from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from digital_twin.data.schemas import (
    validate_events,
    validate_observations,
    validate_participants,
)
from digital_twin.data_adapters.mcphases import McPhasesAdapter
from digital_twin.real_data.mcphases_audit import run_mcphases_audit


def _write_release(root: Path) -> Path:
    root.mkdir()
    participant_ids = ["synthetic-a", "synthetic-b"]
    subjects = pd.DataFrame(
        {
            "id": participant_ids,
            "birth_year": [1990, 1995],
            "gender": ["synthetic", "synthetic"],
            "ethnicity": ["synthetic", "synthetic"],
            "education": ["synthetic", "synthetic"],
            "sexually_active": ["synthetic", "synthetic"],
            "self_report_menstrual_health_literacy": ["synthetic", "synthetic"],
            "age_of_first_menarche": [12, 13],
        }
    )
    subjects.to_csv(root / "subject-info.csv", index=False)
    pd.DataFrame(
        {
            "id": participant_ids,
            "height_2022": [160, 165],
            "weight_2022": [60, 65],
            "height_2024": [np.nan, np.nan],
            "weight_2024": [np.nan, np.nan],
        }
    ).to_csv(root / "height_and_weight.csv", index=False)

    symptom_columns = [
        "appetite",
        "exerciselevel",
        "headaches",
        "cramps",
        "sorebreasts",
        "fatigue",
        "sleepissue",
        "moodswing",
        "stress",
        "foodcravings",
        "indigestion",
        "bloating",
    ]
    rows: list[dict[str, object]] = []
    for participant in participant_ids:
        for day in range(1, 41):
            if day in {2, 21}:
                phase = "Menstrual"
            elif day in {3, 22}:
                phase = "Menstrual"
            elif day in {9, 10, 28, 29}:
                phase = "Fertility"
            elif 11 <= day <= 20 or 30 <= day <= 40:
                phase = "Luteal"
            else:
                phase = "Follicular"
            row: dict[str, object] = {
                "id": participant,
                "study_interval": 2022,
                "is_weekend": False,
                "day_in_study": day,
                "phase": phase,
                "lh": 5.0,
                "estrogen": 100.0,
                "pdg": np.nan,
                "flow_volume": "Light" if phase == "Menstrual" else "Not at all",
                "flow_color": "Bright Red" if phase == "Menstrual" else "Not at all",
            }
            row.update({column: "Moderate" for column in symptom_columns})
            rows.append(row)
    hormones = pd.DataFrame(rows)
    hormones.to_csv(root / "hormones_and_selfreport.csv", index=False)

    keys = hormones[["id", "study_interval", "is_weekend", "day_in_study"]].copy()
    pd.DataFrame(
        {
            "id": keys["id"],
            "study_interval": keys["study_interval"],
            "is_weekend": keys["is_weekend"],
            "sleep_start_day_in_study": keys["day_in_study"],
            "sleep_start_timestamp": "11:00:00 PM",
            "sleep_end_day_in_study": keys["day_in_study"],
            "sleep_end_timestamp": "07:00:00 AM",
            "type": "skin",
            "temperature_samples": 100,
            "nightly_temperature": 34.0,
            "baseline_relative_sample_sum": 0.0,
            "baseline_relative_sample_sum_of_squares": 0.0,
            "baseline_relative_nightly_standard_deviation": 0.1,
            "baseline_relative_sample_standard_deviation": 0.1,
        }
    ).to_csv(root / "computed_temperature.csv", index=False)
    keys.assign(value=60.0, error=1.0).to_csv(root / "resting_heart_rate.csv", index=False)
    keys.assign(
        timestamp="01:00:00 AM",
        rmssd=40.0,
        coverage=0.9,
        low_frequency=100.0,
        high_frequency=100.0,
    ).to_csv(root / "heart_rate_variability_details.csv", index=False)
    keys.assign(
        timestamp="07:00:00 AM",
        overall_score=80.0,
        composition_score=80.0,
        revitalization_score=80.0,
        duration_score=80.0,
        deep_sleep_in_minutes=60.0,
        resting_heart_rate=60.0,
        restlessness=5.0,
    ).to_csv(root / "sleep_score.csv", index=False)
    respiratory = keys.assign(
        timestamp="07:00:00 AM",
        full_sleep_breathing_rate=15.0,
        full_sleep_standard_deviation=1.0,
        full_sleep_signal_to_noise=10.0,
        deep_sleep_breathing_rate=14.0,
        deep_sleep_standard_deviation=1.0,
        deep_sleep_signal_to_noise=10.0,
        light_sleep_breathing_rate=15.0,
        light_sleep_standard_deviation=1.0,
        light_sleep_signal_to_noise=10.0,
        rem_sleep_breathing_rate=16.0,
        rem_sleep_standard_deviation=1.0,
        rem_sleep_signal_to_noise=10.0,
    )
    respiratory.to_csv(root / "respiratory_rate_summary.csv", index=False)
    keys.assign(sedentary=600, lightly=60, moderately=30, very=10).to_csv(
        root / "active_minutes.csv", index=False
    )
    (root / "README.txt").write_text("Synthetic test fixture; not participant data.\n", encoding="utf-8")
    (root / "LICENSE.txt").write_text("Synthetic test fixture.\n", encoding="utf-8")

    files = sorted(path for path in root.iterdir() if path.is_file())
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in files
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return root


def test_mcphases_release_conversion_is_valid_and_nonclinical(tmp_path: Path) -> None:
    source = _write_release(tmp_path / "release")
    adapter = McPhasesAdapter()
    inspection = adapter.inspect(source)
    assert inspection["participants"] == 2
    assert inspection["absolute_dates_available"] is False

    result = adapter.convert(source)
    assert len(result.participants) == 2
    assert set(result.participants["participant_id"]).isdisjoint({"synthetic-a", "synthetic-b"})
    assert not validate_participants(result.participants)
    assert not validate_observations(result.daily_observations)
    assert not validate_events(result.events)
    assert not validate_events(result.reference_intervals)
    assert result.daily_observations["source_record_id"].is_unique

    pdg = result.daily_observations[result.daily_observations["signal_name"].eq("pdg")]
    assert pdg["value"].isna().all()
    assert set(pdg["missingness_reason"]) == {"structurally_unavailable_interval_1"}
    assert set(result.events["event_type"]) == {"menstruation_onset"}
    assert not result.reference_intervals["event_type"].str.contains("clinical|ultrasound", case=False).any()
    assert result.reference_intervals["certainty"].str.contains("not_clinical").all()
    assert result.cycles["left_censored"].any()
    assert result.cycles["right_censored"].any()
    assert result.cycles["eligible_for_primary_evaluation"].sum() == 2

    manifest = adapter.write(result, tmp_path / "curated")
    assert manifest["access_classification"] == "restricted_health_data"
    assert manifest["shareable"] is False
    assert (tmp_path / "curated" / "cycles.parquet").exists()
    report = run_mcphases_audit(tmp_path / "curated", tmp_path / "audit")
    text = report.read_text(encoding="utf-8")
    assert "synthetic-a" not in text
    assert "synthetic-b" not in text
    assert "strict rolling-K3 comparison is underpowered" in text
    with pytest.raises(FileExistsError):
        adapter.write(result, tmp_path / "curated")


def test_mcphases_inspection_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = _write_release(tmp_path / "release")
    with (source / "hormones_and_selfreport.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")
    with pytest.raises(ValueError, match="Checksum verification failed"):
        McPhasesAdapter().inspect(source)
