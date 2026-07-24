from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from digital_twin.baselines.hsmm import HSMMBaseline
from digital_twin.baselines.renewal import HierarchicalRenewal, calendar_forecast, rolling_mean_forecast
from digital_twin.config import ExperimentConfig, save_resolved_config
from digital_twin.data.schemas import assert_valid, validate_events, validate_observations, validate_participants
from digital_twin.evaluation.calibration import coverage_table
from digital_twin.evaluation.metrics import event_metrics
from digital_twin.evaluation.variability import variability_decomposition
from digital_twin.fokker_planck.solver import monte_carlo_first_passage, solve_constant_coefficients
from digital_twin.inference.forecast import forecast_events
from digital_twin.inference.particle_filter import run_particle_filter
from digital_twin.simulation.cohort import SyntheticCohort, simulate_cohort
from digital_twin.simulation.missingness import available_snapshot
from digital_twin.visualization.figures import generate_milestone_figures


def _write_table(df: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.md").write_text(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_csv(index=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _cycle_table(cohort: SyntheticCohort) -> pd.DataFrame:
    events = cohort.events.copy()
    events["event_time"] = pd.to_datetime(events["event_time"])
    menses = events[events["event_type"].eq("menstruation_onset")][["participant_id", "cycle_id", "event_time"]].rename(columns={"event_time": "cycle_start"})
    ovulation = events[events["event_type"].eq("latent_ovulation_transition")][["participant_id", "cycle_id", "event_time", "certainty"]].rename(columns={"event_time": "ovulation_time", "certainty": "ovulation_regime"})
    table = menses.merge(ovulation, on=["participant_id", "cycle_id"], how="left").sort_values(["participant_id", "cycle_id"])
    table["next_menses_time"] = table.groupby("participant_id")["cycle_start"].shift(-1)
    table["cycle_length"] = (table["next_menses_time"] - table["cycle_start"]).dt.total_seconds() / 86400
    table["follicular_duration"] = (table["ovulation_time"] - table["cycle_start"]).dt.total_seconds() / 86400
    table["luteal_duration"] = (table["next_menses_time"] - table["ovulation_time"]).dt.total_seconds() / 86400
    return table


def _filter_cycle(
    observations: pd.DataFrame,
    cycle: pd.Series,
    cfg: ExperimentConfig,
    seed_offset: int,
    signals: set[str] | None = None,
    prior_cycle_lengths: np.ndarray | None = None,
):
    cycle_label = f"{cycle['participant_id']}-C{int(cycle['cycle_id']):03d}"
    issue_time = pd.Timestamp(cycle["cycle_start"]) + pd.Timedelta(days=cfg.evaluation.issue_day)
    subset = observations[(observations["participant_id"] == cycle["participant_id"]) & (observations["cycle_id"] == cycle_label)]
    subset = available_snapshot(subset, issue_time, observed_only=False)
    if signals is not None:
        subset = subset[subset["signal_name"].isin(signals)]
    history = np.asarray(prior_cycle_lengths if prior_cycle_lengths is not None else [], dtype=float)
    history = history[np.isfinite(history)]
    prior_f = None if not history.size else float(np.clip(np.mean(history) - cfg.process.luteal_days, 6.0, 60.0))
    result = run_particle_filter(subset, cfg, seed_offset=seed_offset, prior_follicular_days=prior_f)
    return result, forecast_events(result, cfg, seed_offset=seed_offset)


def _parameter_recovery(cycles: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    realized = cycles.groupby("participant_id").agg(
        recovered_follicular_log_speed=("follicular_duration", lambda x: -np.log(np.nanmean(x))),
        recovered_luteal_log_speed=("luteal_duration", lambda x: -np.log(np.nanmean(x))),
        follicular_duration_sd=("follicular_duration", "std"),
        luteal_duration_sd=("luteal_duration", "std"),
    ).reset_index()
    return parameters[["participant_id", "mu_log_speed_f", "mu_log_speed_l"]].rename(columns={
        "mu_log_speed_f": "true_follicular_log_speed", "mu_log_speed_l": "true_luteal_log_speed"
    }).merge(realized, on="participant_id")


def _fp_comparison(seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    fp = solve_constant_coefficients()
    mc = monte_carlo_first_passage(12000, seed=seed)
    finite = mc[np.isfinite(mc)]
    edges = np.append(fp.time, fp.time[-1] + (fp.time[1] - fp.time[0]))
    hist, _ = np.histogram(finite, bins=edges, density=True)
    table = pd.DataFrame({"time": fp.time, "pde_event_density": fp.event_density, "mc_event_density": hist})
    pde_mass = fp.event_density * (fp.time[1] - fp.time[0])
    pde_mean = float(np.sum(fp.time * pde_mass) / max(pde_mass.sum(), 1e-12))
    return table, {"pde_mean_days": pde_mean, "mc_mean_days": float(np.mean(finite)), "mean_difference_days": abs(pde_mean - float(np.mean(finite))), "mc_crossing_fraction": float(len(finite) / len(mc))}


def run_milestone_1(cfg: ExperimentConfig, project_root: Path) -> Path:
    started = datetime.now(UTC)
    output_dir = (project_root / cfg.experiment.output_dir).resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output_dir / "resolved_config.yaml")
    log_lines = [f"started={started.isoformat()}", f"seed={cfg.experiment.seed}"]

    cohort = simulate_cohort(cfg)
    assert_valid(validate_observations(cohort.observed), validate_events(cohort.events), validate_participants(cohort.participants))
    synthetic_root = project_root / "data" / "synthetic"
    for sub in ("raw", "observed", "truth"):
        (synthetic_root / sub).mkdir(parents=True, exist_ok=True)
    cohort.raw_observations.to_csv(synthetic_root / "raw" / "measurements_before_missingness.csv", index=False)
    cohort.observed.to_csv(synthetic_root / "observed" / "common_observations.csv", index=False)
    cohort.truth.to_csv(synthetic_root / "truth" / "latent_states.csv", index=False)
    cohort.events.to_csv(synthetic_root / "truth" / "events.csv", index=False)
    cohort.parameters.to_csv(synthetic_root / "truth" / "participant_parameters.csv", index=False)
    cohort.participants.to_csv(synthetic_root / "observed" / "participants.csv", index=False)
    cycles = _cycle_table(cohort)
    cycles.to_csv(output_dir / "cycle_summary.csv", index=False)

    complete_cycles = cycles.dropna(subset=["cycle_length", "ovulation_time"]).copy()
    training = complete_cycles[complete_cycles["cycle_id"] <= 2]
    testing = complete_cycles[complete_cycles["cycle_id"] > 2]
    renewal = HierarchicalRenewal().fit(training["cycle_length"].to_numpy())
    population_mean = float(training["cycle_length"].mean())
    population_sd = float(max(training["cycle_length"].std(), 1.0))
    metric_rows: list[dict[str, Any]] = []
    sample_sets: list[np.ndarray] = []
    truths: list[float] = []
    forecast_rows: list[dict[str, Any]] = []
    example_filter = None
    example_forecast = None
    evaluated = 0
    for _, cycle in testing.iterrows():
        history = complete_cycles[(complete_cycles["participant_id"] == cycle["participant_id"]) & (complete_cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].to_numpy()
        result, forecast = _filter_cycle(cohort.observed, cycle, cfg, evaluated, prior_cycle_lengths=history)
        truth_menses = float(cycle["cycle_length"])
        truth_ovulation = float(cycle["follicular_duration"])
        twin_metrics = event_metrics(forecast.next_menses_samples, truth_menses, cfg.evaluation.interval_levels)
        metric_rows.append({"participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "model": "digital_twin", "target": "next_menses", **twin_metrics})
        ov_metrics = event_metrics(forecast.ovulation_samples, truth_ovulation, cfg.evaluation.interval_levels)
        metric_rows.append({"participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "model": "digital_twin", "target": "latent_ovulation_transition", **ov_metrics})
        sample_sets.append(forecast.next_menses_samples)
        truths.append(truth_menses)
        baseline_samples = {
            "population_calendar": np.random.default_rng(cfg.experiment.seed + evaluated).normal(calendar_forecast(0, population_mean), population_sd, cfg.inference.forecast_samples),
            "rolling_mean": np.random.default_rng(cfg.experiment.seed + 100 + evaluated).normal(rolling_mean_forecast(0, history, 3), max(float(np.std(history)) if len(history) > 1 else population_sd, 1), cfg.inference.forecast_samples),
            "hierarchical_renewal": renewal.predictive_samples(0, history, cfg.inference.forecast_samples, np.random.default_rng(cfg.experiment.seed + 200 + evaluated)),
        }
        for model, samples in baseline_samples.items():
            metric_rows.append({"participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "model": model, "target": "next_menses", **event_metrics(samples, truth_menses, cfg.evaluation.interval_levels)})
        if example_filter is None:
            example_filter, example_forecast = result, forecast
            for event_type, samples in (("ovulation", forecast.ovulation_samples), ("next_menses", forecast.next_menses_samples)):
                forecast_rows.extend({"event_type": event_type, "event_day": float(x)} for x in samples if np.isfinite(x))
        evaluated += 1

    if not metric_rows or example_filter is None or example_forecast is None:
        raise RuntimeError("No held-out cycles were available for evaluation")
    metrics = pd.DataFrame(metric_rows)
    calibration = coverage_table(sample_sets, np.asarray(truths), cfg.evaluation.interval_levels)
    parameter_recovery = _parameter_recovery(complete_cycles, cohort.parameters)
    variability = variability_decomposition(complete_cycles, seed=cfg.experiment.seed)

    first_cycle = testing.iloc[0]
    robustness_rows = []
    for idx, (_, cycle) in enumerate(testing.head(min(8, len(testing))).iterrows()):
        history = complete_cycles[(complete_cycles["participant_id"] == cycle["participant_id"]) & (complete_cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].to_numpy()
        raw_result, raw_forecast = _filter_cycle(cohort.raw_observations, cycle, cfg, 500 + idx, prior_cycle_lengths=history)
        missing_result, missing_forecast = _filter_cycle(cohort.observed, cycle, cfg, 700 + idx, prior_cycle_lengths=history)
        raw_metrics = event_metrics(raw_forecast.next_menses_samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels)
        missing_metrics = event_metrics(missing_forecast.next_menses_samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels)
        robustness_rows.append({
            "participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]),
            "complete_width_90": raw_metrics["width_90"], "missing_width_90": missing_metrics["width_90"],
            "complete_crps": raw_metrics["crps"], "missing_crps": missing_metrics["crps"],
        })
    robustness = pd.DataFrame(robustness_rows)

    modality_sets = {
        "bleeding_only": {"bleeding"},
        "bleeding_temperature": {"bleeding", "temperature"},
        "bleeding_wearables": {"bleeding", "temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"},
        "bleeding_hormones": {"bleeding", "lh", "e3g", "pdg", "estradiol", "progesterone"},
        "all_modalities": None,
    }
    ablation_rows = []
    first_history = complete_cycles[(complete_cycles["participant_id"] == first_cycle["participant_id"]) & (complete_cycles["cycle_id"] < first_cycle["cycle_id"])]["cycle_length"].to_numpy()
    for modality, signals in modality_sets.items():
        result, forecast = _filter_cycle(cohort.observed, first_cycle, cfg, 900 + len(ablation_rows), signals, first_history)
        ablation_rows.append({"modality": modality, **event_metrics(forecast.next_menses_samples, float(first_cycle["cycle_length"]), cfg.evaluation.interval_levels)})
    ablation = pd.DataFrame(ablation_rows)

    cycle_label = f"{first_cycle['participant_id']}-C{int(first_cycle['cycle_id']):03d}"
    issue_time = pd.Timestamp(first_cycle["cycle_start"]) + pd.Timedelta(days=cfg.evaluation.issue_day)
    daily = available_snapshot(cohort.observed[(cohort.observed["participant_id"] == first_cycle["participant_id"]) & (cohort.observed["cycle_id"] == cycle_label)], issue_time)
    daily["date"] = pd.to_datetime(daily["event_time"]).dt.normalize()
    daily_wide = daily.pivot_table(index="date", columns="signal_name", values="value", aggfunc="last").reset_index()
    hsmm_summary = HSMMBaseline().filter(daily_wide)

    fp_table, fp_metrics = _fp_comparison(cfg.experiment.seed)
    forecast_samples = pd.DataFrame(forecast_rows)
    _write_table(metrics, tables_dir, "predictive_results")
    _write_table(calibration, tables_dir, "calibration")
    _write_table(parameter_recovery, tables_dir, "parameter_recovery")
    _write_table(variability, tables_dir, "variability_decomposition")
    _write_table(robustness, tables_dir, "missingness_robustness")
    _write_table(ablation, tables_dir, "modality_ablation")
    _write_table(hsmm_summary, tables_dir, "hsmm_example")
    _write_table(fp_table, tables_dir, "fokker_planck_comparison")
    example_filter.summary.to_csv(output_dir / "filtered_state_summary.csv", index=False)
    forecast_samples.to_csv(output_dir / "event_forecast_samples.csv", index=False)
    generate_milestone_figures(cohort.truth, cohort.observed, example_filter.summary, forecast_samples, calibration, metrics[metrics["target"].eq("next_menses")], parameter_recovery, variability, robustness, fp_table, figures_dir)

    summary = metrics.groupby(["model", "target"], as_index=False).agg(
        n=("absolute_error", "count"), mae=("absolute_error", "mean"), rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))), crps=("crps", "mean"), wis=("wis", "mean"), coverage_90=("coverage_90", "mean"), width_90=("width_90", "mean")
    )
    _write_table(summary, tables_dir, "main_results")
    recovery_correlations = {
        "follicular": float(parameter_recovery[["true_follicular_log_speed", "recovered_follicular_log_speed"]].corr().iloc[0, 1]),
        "luteal": float(parameter_recovery[["true_luteal_log_speed", "recovered_luteal_log_speed"]].corr().iloc[0, 1]),
    }
    report = f"""# Milestone 1 automatic report

Generated: {datetime.now(UTC).isoformat()}

## Scope

This report contains synthetic, model-development evidence only. The latent ovulation transition is known because it was simulated. It is not biological or clinical ovulation validation.

## Execution

- Participants: {cfg.data.participants}
- Cycles configured per participant: {cfg.data.cycles_per_participant}
- Held-out forecast cases: {len(sample_sets)}
- Particle count: {cfg.inference.particles}
- Forecast issue day: {cfg.evaluation.issue_day}

## Main results

{summary.to_markdown(index=False)}

## Calibration

{calibration.to_markdown(index=False)}

## Parameter recovery

- Follicular participant log-speed recovery correlation: {recovery_correlations['follicular']:.3f}
- Luteal participant log-speed recovery correlation: {recovery_correlations['luteal']:.3f}

These event-time estimators use simulated transition truth and therefore test recoverability under the generator, not inference from a public cohort.

## Variability decomposition

{variability.to_markdown(index=False)}

This decomposition uses exact simulated event times. It demonstrates an interpretable output unavailable from a point calendar baseline, but does not establish recoverability from real observations.

## Fokker--Planck verification

- PDE mean first-passage time: {fp_metrics['pde_mean_days']:.3f} days
- Monte Carlo mean first-passage time: {fp_metrics['mc_mean_days']:.3f} days
- Absolute difference: {fp_metrics['mean_difference_days']:.3f} days

## Scientific gates

- Simulator correctness: evaluated by unit/statistical tests and event summaries.
- Oracle inference: exercised on held-out synthetic cycles.
- Identifiability: core stage speeds have recovery summaries; a complete Bayesian SBC study remains future work.
- Missingness robustness: uncertainty-width comparison is reported, but failure of intervals to widen in every case is a model-development warning, not evidence of robustness.
- Baseline value: the configured twin has a synthetic calibration/interpretability advantage if its WIS/coverage and decomposition support it; point-forecast superiority is not assumed.
- Public-data, external-validity, and resilience gates: not passed.

## Limitations

The generator and filter share observation templates, hormone sampling is synthetic, baseline estimation is deliberately simple, and the run is too small to support biomedical conclusions. Anovulatory-like cycles are neutral simulations rather than diagnoses. No real participant data were used.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    finished = datetime.now(UTC)
    files = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "experiment_card.json"]
    card = {
        "experiment_id": cfg.experiment.name,
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "seed": cfg.experiment.seed,
        "git_commit": _git_commit(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__},
        "config": cfg.to_dict(),
        "input_checksums": {"resolved_config.yaml": _sha256(output_dir / "resolved_config.yaml")},
        "output_checksums": {str(path.relative_to(output_dir)): _sha256(path) for path in sorted(files)},
        "warnings": ["synthetic evidence only", "no public-data validation", "git commit unavailable" if _git_commit(project_root) is None else ""],
        "exclusions": [],
        "convergence": {"particle_filter_runs": evaluated, "parameter_recovery_correlations": recovery_correlations},
        "fokker_planck": fp_metrics,
    }
    (output_dir / "experiment_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    log_lines.extend([f"finished={finished.isoformat()}", f"evaluated_cycles={evaluated}", "status=completed"])
    (output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return output_dir
