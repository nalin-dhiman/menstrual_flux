from dataclasses import replace

from digital_twin.config import ExperimentConfig
from digital_twin.experiments import run_milestone_1


def test_small_reproducible_experiment(tmp_path):
    cfg = ExperimentConfig()
    cfg = replace(cfg, experiment=replace(cfg.experiment, output_dir="experiment"), data=replace(cfg.data, participants=3, cycles_per_participant=4), inference=replace(cfg.inference, particles=100, forecast_samples=100))
    output = run_milestone_1(cfg, tmp_path)
    assert (output / "REPORT.md").exists()
    assert (output / "experiment_card.json").exists()
    assert (output / "tables" / "main_results.csv").exists()
    assert (output / "figures" / "calibration_curve.pdf").exists()
