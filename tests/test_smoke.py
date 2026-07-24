from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from menstrual_twin.config import load_config
from menstrual_twin.data_validation import validate_daily_observations
from menstrual_twin.particle_filter import forecast_next_menses, run_particle_filter
from menstrual_twin.simulator import simulate_cohort


def test_synthetic_pipeline_smoke():
    cfg = load_config(ROOT / "config" / "model_v1.yaml")
    cohort = simulate_cohort(2, 80, cfg)
    report = validate_daily_observations(cohort)
    assert report.valid
    one = cohort[cohort.participant_id == "P001"]
    result = run_particle_filter(one, cfg)
    assert len(result.summary) == len(one)
    samples = forecast_next_menses(result, cfg)
    assert samples.size > 0
