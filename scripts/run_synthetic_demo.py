#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from menstrual_twin.config import load_config
from menstrual_twin.data_validation import validate_daily_observations
from menstrual_twin.metrics import circular_mae_days
from menstrual_twin.particle_filter import forecast_next_menses, run_particle_filter
from menstrual_twin.simulator import simulate_cohort


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic end-to-end digital-twin prototype.")
    parser.add_argument("--config", default=str(ROOT / "config" / "model_v1.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results" / "synthetic_demo"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    cohort = simulate_cohort(n_participants=5, n_days=210, cfg=cfg)
    cohort.to_csv(out / "synthetic_cohort.csv", index=False)
    report = validate_daily_observations(cohort)
    if not report.valid:
        raise RuntimeError("Synthetic data failed validation: " + "; ".join(report.errors))

    df = cohort[cohort["participant_id"] == "P001"].copy()
    train = df.iloc[:170].copy()
    result = run_particle_filter(train, cfg)
    result.summary.to_csv(out / "filtered_state_summary.csv", index=False)

    phase_error = circular_mae_days(
        train["true_phase_rad"].to_numpy(),
        result.summary["phase_mean_rad"].to_numpy(),
        cycle_days=cfg.process.mean_cycle_days,
    )
    samples = forecast_next_menses(result, cfg)
    np.savetxt(out / "next_menses_forecast_samples_days.csv", samples, delimiter=",")

    # Approximate future truth from the simulation.
    future = df.iloc[170:].copy()
    onset_rows = future.index[future["true_onset_event"].astype(bool)].to_list()
    truth = float(onset_rows[0] - train.index[-1]) if onset_rows else float("nan")
    summary = pd.DataFrame([{
        "phase_circular_mae_days": phase_error,
        "forecast_median_days": float(np.median(samples)) if samples.size else np.nan,
        "forecast_p05_days": float(np.quantile(samples, 0.05)) if samples.size else np.nan,
        "forecast_p95_days": float(np.quantile(samples, 0.95)) if samples.size else np.nan,
        "synthetic_truth_days": truth,
        "validation_warnings": " | ".join(report.warnings),
    }])
    summary.to_csv(out / "demo_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(train["day_in_study"], train["true_phase_fraction"], label="True phase fraction", linewidth=1.5)
    ax.plot(result.summary["day_in_study"], result.summary["phase_mean_fraction"], label="Filtered mean phase", linewidth=1.2)
    ax.fill_between(
        result.summary["day_in_study"],
        np.clip(result.summary["phase_mean_fraction"] - (1.0 - result.summary["phase_concentration"]) / 2.0, 0, 1),
        np.clip(result.summary["phase_mean_fraction"] + (1.0 - result.summary["phase_concentration"]) / 2.0, 0, 1),
        alpha=0.18,
        label="Qualitative uncertainty band",
    )
    ax.set_xlabel("Day in study")
    ax.set_ylabel("Cycle phase fraction")
    ax.set_title("Synthetic prototype: latent phase filtering")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "phase_filter_demo.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
