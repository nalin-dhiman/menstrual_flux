from dataclasses import replace

import pandas as pd
import pytest

from digital_twin.config import ExperimentConfig, validate_config
from digital_twin.data.schemas import validate_observations


def test_config_rejects_invalid_probability():
    cfg = ExperimentConfig()
    with pytest.raises(ValueError):
        validate_config(replace(cfg, missingness=replace(cfg.missingness, mcar_probability=1.2)))


def test_schema_requires_explicit_missingness_reason():
    row = {
        "participant_id": "P1", "source_dataset": "synthetic", "source_record_id": "r1", "cycle_id": "c1",
        "signal_name": "temperature", "value": None, "unit": "degC", "event_time": "2026-01-01",
        "measurement_time": "2026-01-01 07:00", "report_time": "2026-01-01 08:00",
        "availability_time": "2026-01-01 08:00", "device": "d", "assay": "none", "quality_flag": "pass",
        "missingness_reason": "", "raw_column": "temperature", "transformation_version": "v1",
    }
    issues = validate_observations(pd.DataFrame([row]))
    assert any(issue.rule == "missingness_reason" for issue in issues)
