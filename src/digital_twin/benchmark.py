from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from digital_twin.baselines.hsmm import HSMMBaseline
from digital_twin.baselines.renewal import HierarchicalRenewal, rolling_mean_forecast
from digital_twin.config import ExperimentConfig, load_experiment_config
from digital_twin.evaluation.calibration import coverage_table
from digital_twin.evaluation.gates import evaluate_scientific_gates
from digital_twin.evaluation.metrics import event_metrics
from digital_twin.experiments import _cycle_table, _write_table
from digital_twin.inference.forecast import forecast_events
from digital_twin.inference.particle_filter import _daily_wide, run_particle_filter
from digital_twin.inference.personalization import ParticipantDurationPosterior, StageSpeedPosterior
from digital_twin.simulation.cohort import SyntheticCohort, simulate_cohort
from digital_twin.simulation.missingness import apply_missingness, available_snapshot


MODALITIES: dict[str, set[str] | None] = {
    "bleeding_only": {"bleeding"},
    "bleeding_temperature": {"bleeding", "temperature"},
    "wearables": {"bleeding", "temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"},
    "hormones": {"lh", "e3g", "pdg", "estradiol", "progesterone"},
    "all_modalities": None,
}

FUSION_COMPONENTS: tuple[str, ...] = (
    "joint_filter",
    "hormones",
    "wearables",
    "bleeding_temperature",
    "bleeding_only",
    "duration_prior",
    "rolling_k3_prior",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    base_config: str
    output_dir: str
    seeds: tuple[int, ...]
    scenarios: tuple[str, ...]
    safety_scenarios: tuple[str, ...]
    participants: int
    cycles_per_participant: int
    particles: int
    forecast_samples: int
    issue_day: int
    abstention_issue_day: int
    adaptation_cycles: int
    train_fraction: float
    calibration_fraction: float
    discrepancy_grid: tuple[float, ...]
    fusion_temperature_grid: tuple[float, ...]
    fusion_missingness_floor_grid: tuple[float, ...]
    fusion_spread_grid: tuple[float, ...]
    robustness_cycles_per_run: int
    ablation_cycles_per_run: int
    minimum_eligible_fraction: float
    resume: bool = True


@dataclass(frozen=True)
class FusionStrategy:
    weights: dict[str, float]
    missingness_floor: float
    temperature: float
    spread_scale: float
    discrepancy_sd_days: float


@dataclass(frozen=True)
class MarginalCalibration:
    spread_scale: float
    discrepancy_sd_days: float


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    section = raw.get("benchmark", raw)
    required = {"name", "base_config", "output_dir", "seeds", "scenarios"}
    missing = required - set(section)
    if missing:
        raise ValueError(f"Missing benchmark keys: {sorted(missing)}")
    defaults: dict[str, Any] = {
        "participants": 24,
        "safety_scenarios": ["anovulatory_like", "stalled_transition"],
        "cycles_per_participant": 6,
        "particles": 300,
        "forecast_samples": 300,
        "issue_day": 10,
        "abstention_issue_day": 21,
        "adaptation_cycles": 2,
        "train_fraction": 0.50,
        "calibration_fraction": 0.25,
        "discrepancy_grid": [0.0, 0.8, 1.2, 1.6, 2.0, 2.5],
        "fusion_temperature_grid": [0.5, 1.0, 2.0, 4.0, 8.0],
        "fusion_missingness_floor_grid": [0.0, 0.1, 0.25, 0.5, 1.0],
        "fusion_spread_grid": [1.0, 1.25, 1.5, 2.0],
        "robustness_cycles_per_run": 8,
        "ablation_cycles_per_run": 0,
        "minimum_eligible_fraction": 0.99,
        "resume": True,
    }
    defaults.update(section)
    defaults["seeds"] = tuple(int(x) for x in defaults["seeds"])
    defaults["scenarios"] = tuple(str(x) for x in defaults["scenarios"])
    defaults["safety_scenarios"] = tuple(str(x) for x in defaults["safety_scenarios"])
    defaults["discrepancy_grid"] = tuple(float(x) for x in defaults["discrepancy_grid"])
    defaults["fusion_temperature_grid"] = tuple(float(x) for x in defaults["fusion_temperature_grid"])
    defaults["fusion_missingness_floor_grid"] = tuple(float(x) for x in defaults["fusion_missingness_floor_grid"])
    defaults["fusion_spread_grid"] = tuple(float(x) for x in defaults["fusion_spread_grid"])
    cfg = BenchmarkConfig(**defaults)
    if cfg.train_fraction <= 0 or cfg.calibration_fraction <= 0 or cfg.train_fraction + cfg.calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave a non-empty locked test fraction")
    if cfg.adaptation_cycles < 1 or cfg.cycles_per_participant < cfg.adaptation_cycles + 2:
        raise ValueError("cycles_per_participant must leave later cycles after adaptation")
    if not 0 < cfg.minimum_eligible_fraction <= 1:
        raise ValueError("minimum_eligible_fraction must be in (0, 1]")
    if not cfg.fusion_temperature_grid or any(x <= 0 for x in cfg.fusion_temperature_grid):
        raise ValueError("fusion_temperature_grid must contain positive values")
    if not cfg.fusion_missingness_floor_grid or any(x < 0 or x > 1 for x in cfg.fusion_missingness_floor_grid):
        raise ValueError("fusion_missingness_floor_grid must be in [0, 1]")
    if not cfg.fusion_spread_grid or any(x < 1 for x in cfg.fusion_spread_grid):
        raise ValueError("fusion_spread_grid must contain values >= 1")
    return cfg


def _scenario_config(base: ExperimentConfig, benchmark: BenchmarkConfig, seed: int, scenario: str) -> ExperimentConfig:
    cfg = replace(
        base,
        experiment=replace(base.experiment, seed=seed, name=f"{scenario}_seed_{seed}"),
        data=replace(base.data, participants=benchmark.participants, cycles_per_participant=benchmark.cycles_per_participant, scenario=scenario),
        inference=replace(base.inference, particles=benchmark.particles, forecast_samples=benchmark.forecast_samples, forecast_model_discrepancy_sd_days=0.0),
        evaluation=replace(base.evaluation, issue_day=benchmark.issue_day),
    )
    if scenario == "high_noise":
        cfg = replace(cfg, data=replace(cfg.data, scenario="mixed"), observation=replace(cfg.observation, temperature_sd=0.16, rhr_sd=3.0, log_hormone_sd=0.35))
    elif scenario == "high_missingness":
        cfg = replace(cfg, data=replace(cfg.data, scenario="mixed"), missingness=replace(cfg.missingness, mcar_probability=0.22, mar_probability=0.18, informative_probability=0.12, block_probability=0.35, hormone_schedule_days=5))
    elif scenario == "heterogeneous":
        cfg = replace(cfg, data=replace(cfg.data, scenario="mixed"), process=replace(cfg.process, between_person_log_speed_sd=0.24))
    elif scenario == "drift":
        cfg = replace(cfg, data=replace(cfg.data, scenario="slow_drift"), process=replace(cfg.process, drift_per_cycle_sd=0.04))
    elif scenario in {"regular", "stable"}:
        cfg = replace(cfg, data=replace(cfg.data, scenario="stable"), process=replace(cfg.process, anovulatory_like_probability=0.0))
    elif scenario == "anovulatory_like":
        cfg = replace(cfg, process=replace(cfg.process, anovulatory_like_probability=1.0))
    return cfg


def _participant_split(participants: np.ndarray, seed: int, train_fraction: float, calibration_fraction: float) -> pd.DataFrame:
    ids = np.array(sorted(participants.astype(str)))
    rng = np.random.default_rng(seed + 771)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_fraction)))
    n_calibration = max(1, int(round(len(ids) * calibration_fraction)))
    n_train = min(n_train, len(ids) - 2)
    n_calibration = min(n_calibration, len(ids) - n_train - 1)
    labels = ["population_train"] * n_train + ["calibration"] * n_calibration + ["locked_test"] * (len(ids) - n_train - n_calibration)
    return pd.DataFrame({"participant_id": ids, "split": labels})


def _eligibility_ledger(
    cycles: pd.DataFrame,
    participant_ids: set[str],
    benchmark: BenchmarkConfig,
    seed: int,
    scenario: str,
) -> pd.DataFrame:
    """Account for every planned locked-test forecast before scoring."""
    planned = pd.MultiIndex.from_product(
        [sorted(participant_ids), range(benchmark.adaptation_cycles + 1, benchmark.cycles_per_participant + 1)],
        names=["participant_id", "cycle_id"],
    ).to_frame(index=False)
    truth = cycles[["participant_id", "cycle_id", "cycle_length", "follicular_duration", "luteal_duration"]].copy()
    truth = truth[truth["cycle_id"].between(1, benchmark.cycles_per_participant)]
    ledger = planned.merge(truth, on=["participant_id", "cycle_id"], how="left", validate="one_to_one")
    ledger["has_complete_truth"] = ledger["cycle_length"].notna()
    ledger["event_before_issue_day"] = ledger["cycle_length"].le(benchmark.issue_day).fillna(False)
    ledger["eligible"] = ledger["has_complete_truth"] & ~ledger["event_before_issue_day"]
    ledger["exclusion_reason"] = np.select(
        [~ledger["has_complete_truth"], ledger["event_before_issue_day"]],
        ["administratively_censored", "event_before_issue_day"],
        default="eligible",
    )
    ledger["seed"] = seed
    ledger["scenario"] = scenario
    return ledger


def _cycle_observations(observations: pd.DataFrame, cycle: pd.Series, issue_day: int, signals: set[str] | None = None) -> pd.DataFrame:
    label = f"{cycle['participant_id']}-C{int(cycle['cycle_id']):03d}"
    issue_time = pd.Timestamp(cycle["cycle_start"]) + pd.Timedelta(days=issue_day)
    subset = observations[(observations["participant_id"] == cycle["participant_id"]) & (observations["cycle_id"] == label)]
    subset = available_snapshot(subset, issue_time, observed_only=False)
    if signals is not None:
        subset = subset[subset["signal_name"].isin(signals)]
    return subset


def _proxy_stage_history(
    observations: pd.DataFrame,
    cycles: pd.DataFrame,
    participant_id: str,
    current_cycle_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate earlier stage durations using only timestamped model-visible signals."""
    follicular: list[float] = []
    luteal: list[float] = []
    current = cycles[(cycles["participant_id"] == participant_id) & (cycles["cycle_id"] == current_cycle_id)]
    as_of_time = pd.Timestamp(current.iloc[0]["cycle_start"]) if not current.empty else pd.Timestamp.max
    prior_cycles = cycles[(cycles["participant_id"] == participant_id) & (cycles["cycle_id"] < current_cycle_id)].dropna(subset=["cycle_length"])
    for _, cycle in prior_cycles.iterrows():
        label = f"{participant_id}-C{int(cycle['cycle_id']):03d}"
        frame = observations[(observations["participant_id"] == participant_id) & (observations["cycle_id"] == label)]
        frame = frame[pd.to_datetime(frame["availability_time"]) <= as_of_time]
        frame = frame[frame["is_observed"].astype(bool) & frame["value"].notna()].copy()
        if frame.empty:
            continue
        frame["cycle_day"] = (pd.to_datetime(frame["event_time"]) - pd.Timestamp(cycle["cycle_start"])).dt.total_seconds() / 86400
        candidates: list[float] = []
        lh = frame[frame["signal_name"] == "lh"]
        if not lh.empty and float(lh["value"].max()) >= 8:
            candidates.append(float(lh.loc[lh["value"].idxmax(), "cycle_day"]))
        pdg = frame[(frame["signal_name"] == "pdg") & (frame["value"] >= 4)].sort_values("cycle_day")
        if not pdg.empty:
            candidates.append(max(float(pdg.iloc[0]["cycle_day"]) - 2.0, 1.0))
        temperature = frame[frame["signal_name"] == "temperature"].sort_values("cycle_day")
        if len(temperature) >= 6:
            baseline = float(temperature.head(5)["value"].median())
            days = temperature["cycle_day"].to_numpy(dtype=float)
            values = temperature["value"].to_numpy(dtype=float)
            for index in range(len(temperature) - 2):
                if days[index + 2] - days[index] <= 4 and np.all(values[index:index + 3] >= baseline + 0.10):
                    candidates.append(max(float(days[index]) - 1.0, 1.0))
                    break
        if candidates:
            f_days = float(np.clip(np.median(candidates), 5.0, float(cycle["cycle_length"]) - 5.0))
            follicular.append(f_days)
            luteal.append(float(cycle["cycle_length"]) - f_days)
    return np.asarray(follicular), np.asarray(luteal)


def _pooled_stage_prior(values: np.ndarray, population_value: float, strength: float = 2.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return float((strength * population_value + finite.sum()) / (strength + finite.size))


def _training_stage_population(
    observations: pd.DataFrame,
    cycles: pd.DataFrame,
    participant_ids: set[str],
    final_cycle_id: int,
) -> tuple[float, float, float, float]:
    participant_estimates: list[tuple[float, float]] = []
    for participant_id in sorted(participant_ids):
        follicular, luteal = _proxy_stage_history(observations, cycles, participant_id, final_cycle_id + 1)
        if follicular.size and luteal.size:
            participant_estimates.append((float(np.median(-np.log(follicular))), float(np.median(-np.log(luteal)))))
    if not participant_estimates:
        return -np.log(14.0), -np.log(14.0), 0.12, 0.12
    estimates = np.asarray(participant_estimates)
    return (
        float(np.median(estimates[:, 0])),
        float(np.median(estimates[:, 1])),
        float(max(np.std(estimates[:, 0], ddof=1) if len(estimates) > 1 else 0.12, 0.05)),
        float(max(np.std(estimates[:, 1], ddof=1) if len(estimates) > 1 else 0.12, 0.05)),
    )


def _participant_stage_signal_quality(observations: pd.DataFrame, participant_id: str) -> float:
    frame = observations[observations["participant_id"] == participant_id]
    if frame.empty:
        return 0.0
    observed = frame["is_observed"].astype(bool) & frame["value"].notna()
    temperature = frame["signal_name"] == "temperature"
    hormones = frame["signal_name"].isin({"lh", "e3g", "pdg", "estradiol", "progesterone"})
    temperature_quality = float(observed[temperature].mean()) if temperature.any() else 0.0
    hormone_quality = float(observed[hormones].mean()) if hormones.any() else 0.0
    return float(np.clip(0.6 * temperature_quality + 0.4 * hormone_quality, 0.05, 0.95))


def _run_twin(
    observations: pd.DataFrame,
    cycle: pd.Series,
    cfg: ExperimentConfig,
    issue_day: int,
    history: np.ndarray,
    population_mean: float,
    population_sd: float,
    discrepancy: float,
    seed_offset: int,
    signals: set[str] | None = None,
    stage_history: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    participant = ParticipantDurationPosterior.from_history(history, population_mean, population_sd)
    f_history, l_history = stage_history if stage_history is not None else (np.array([]), np.array([]))
    prior_f = _pooled_stage_prior(f_history, participant.follicular_prior_days(cfg.process.luteal_days))
    prior_l = _pooled_stage_prior(l_history, cfg.process.luteal_days)
    subset = _cycle_observations(observations, cycle, issue_day, signals)
    result = run_particle_filter(
        subset,
        cfg,
        seed_offset=seed_offset,
        prior_follicular_days=prior_f,
        prior_luteal_days=prior_l,
    )
    raw_cfg = replace(cfg, inference=replace(cfg.inference, forecast_model_discrepancy_sd_days=0.0))
    forecast = forecast_events(result, raw_cfg, seed_offset=seed_offset)
    menses_samples = forecast.next_menses_samples.copy()
    ovulation_samples = forecast.ovulation_samples.copy()
    ovulation_samples = _calibrated_copy(ovulation_samples, discrepancy, cfg.experiment.seed + 89001 + seed_offset)
    menses_samples = _calibrated_copy(menses_samples, discrepancy, cfg.experiment.seed + 90001 + seed_offset)
    finite = np.isfinite(ovulation_samples) & np.isfinite(menses_samples)
    menses_samples[finite] = np.maximum(menses_samples[finite], ovulation_samples[finite] + 1.0)
    return ovulation_samples, menses_samples, subset


def _calibrated_copy(samples: np.ndarray, discrepancy: float, seed: int) -> np.ndarray:
    copied = np.asarray(samples, dtype=float).copy()
    finite = np.isfinite(copied)
    if discrepancy > 0:
        copied[finite] += np.random.default_rng(seed).normal(0, discrepancy, finite.sum())
    return copied


def _spread_samples(samples: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    median = float(np.nanmedian(values))
    return median + scale * (values - median)


def _duration_prior_samples(
    history: np.ndarray,
    population_mean: float,
    population_sd: float,
    n: int,
    seed: int,
) -> np.ndarray:
    posterior = ParticipantDurationPosterior.from_history(history, population_mean, population_sd)
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(posterior.mean_cycle_days, posterior.predictive_sd_days, n), 15, 90)


def _availability_quality(observations: pd.DataFrame, component: str, issue_day: int) -> float:
    if component in {"duration_prior", "rolling_k3_prior"}:
        return 1.0
    observed = observations[observations["is_observed"].astype(bool) & observations["value"].notna()]
    n_days = max(issue_day + 1, 1)
    if component == "hormones":
        count = observed["signal_name"].isin({"lh", "e3g", "pdg", "estradiol", "progesterone"}).sum()
        return float(np.clip(count / 6, 0, 1))
    if component == "wearables":
        days = observed.loc[observed["signal_name"].isin({"temperature", "resting_heart_rate", "hrv", "sleep_duration", "sleep_efficiency"}), "event_time"].nunique()
        return float(np.clip(days / n_days, 0, 1))
    if component == "bleeding_temperature":
        days = observed.loc[observed["signal_name"].isin({"bleeding", "temperature"}), "event_time"].nunique()
        return float(np.clip(days / n_days, 0, 1))
    if component == "bleeding_only":
        days = observed.loc[observed["signal_name"] == "bleeding", "event_time"].nunique()
        return float(np.clip(days / n_days, 0, 1))
    signal_days = observed["event_time"].nunique()
    signal_types = observed["signal_name"].nunique()
    return float(np.clip(0.5 * signal_days / n_days + 0.5 * signal_types / 10, 0, 1))


def _component_forecasts(
    observations: pd.DataFrame,
    cycle: pd.Series,
    cfg: ExperimentConfig,
    issue_day: int,
    history: np.ndarray,
    population_mean: float,
    population_sd: float,
    seed_offset: int,
    stage_history: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], pd.DataFrame]:
    duration_samples = _duration_prior_samples(
        history,
        population_mean,
        population_sd,
        cfg.inference.forecast_samples,
        cfg.experiment.seed + 97001 + seed_offset,
    )
    rolling_rng = np.random.default_rng(cfg.experiment.seed + 98001 + seed_offset)
    rolling_sd = max(float(np.std(history)) if len(history) > 1 else population_sd, 1.0)
    rolling_samples = rolling_rng.normal(
        rolling_mean_forecast(0, history, 3),
        rolling_sd,
        cfg.inference.forecast_samples,
    )
    samples: dict[str, np.ndarray] = {
        "duration_prior": duration_samples,
        "rolling_k3_prior": rolling_samples,
    }
    qualities: dict[str, float] = {"duration_prior": 1.0, "rolling_k3_prior": 1.0}
    joint_ovulation: np.ndarray | None = None
    joint_subset: pd.DataFrame | None = None
    component_signals = {
        "joint_filter": None,
        "hormones": MODALITIES["hormones"],
        "wearables": MODALITIES["wearables"],
        "bleeding_temperature": MODALITIES["bleeding_temperature"],
        "bleeding_only": MODALITIES["bleeding_only"],
    }
    for index, (component, signals) in enumerate(component_signals.items()):
        snapshot = _cycle_observations(observations, cycle, issue_day, signals)
        observed = snapshot[snapshot["is_observed"].astype(bool) & snapshot["value"].notna()]
        if observed.empty:
            samples[component] = duration_samples.copy()
            qualities[component] = 0.0
            if component == "joint_filter":
                posterior = ParticipantDurationPosterior.from_history(history, population_mean, population_sd)
                rng = np.random.default_rng(cfg.experiment.seed + 96001 + seed_offset)
                joint_ovulation = np.clip(
                    rng.normal(posterior.follicular_prior_days(cfg.process.luteal_days), posterior.predictive_sd_days, cfg.inference.forecast_samples),
                    1,
                    80,
                )
                joint_subset = snapshot
            continue
        ovulation, menses, subset = _run_twin(
            observations,
            cycle,
            cfg,
            issue_day,
            history,
            population_mean,
            population_sd,
            0.0,
            seed_offset + index * 1009,
            signals,
            stage_history,
        )
        finite_fraction = float(np.isfinite(menses).mean())
        samples[component] = np.where(np.isfinite(menses), menses, duration_samples)
        qualities[component] = _availability_quality(subset, component, issue_day) * finite_fraction
        if component == "joint_filter":
            joint_ovulation = ovulation
            joint_subset = subset
    assert joint_ovulation is not None and joint_subset is not None
    return joint_ovulation, samples, qualities, joint_subset


def _fusion_weights(strategy: FusionStrategy, qualities: dict[str, float]) -> dict[str, float]:
    adjusted = {
        name: strategy.weights.get(name, 0.0)
        * (strategy.missingness_floor + (1 - strategy.missingness_floor) * qualities.get(name, 0.0))
        for name in FUSION_COMPONENTS
    }
    total = sum(adjusted.values())
    if total <= 0:
        return {name: (1.0 if name == "duration_prior" else 0.0) for name in FUSION_COMPONENTS}
    return {name: value / total for name, value in adjusted.items()}


def _fuse_samples(
    samples: dict[str, np.ndarray],
    qualities: dict[str, float],
    strategy: FusionStrategy,
) -> np.ndarray:
    weights = _fusion_weights(strategy, qualities)
    fallback = np.sort(np.asarray(samples["duration_prior"], dtype=float))
    ordered = []
    for name in FUSION_COMPONENTS:
        values = np.sort(np.asarray(samples[name], dtype=float))
        values = np.where(np.isfinite(values), values, fallback)
        ordered.append(values)
    fused = np.sum([weights[name] * values for name, values in zip(FUSION_COMPONENTS, ordered)], axis=0)
    return _spread_samples(fused, strategy.spread_scale)


def _calibrate_fusion(
    cohort: SyntheticCohort,
    cycles: pd.DataFrame,
    participant_ids: set[str],
    cfg: ExperimentConfig,
    benchmark: BenchmarkConfig,
    population_mean: float,
    population_sd: float,
) -> tuple[FusionStrategy, pd.DataFrame, pd.DataFrame, MarginalCalibration, pd.DataFrame]:
    records: list[tuple[dict[str, np.ndarray], dict[str, float], float, int]] = []
    eligible = cycles[
        (cycles["participant_id"].isin(participant_ids))
        & (cycles["cycle_id"] > benchmark.adaptation_cycles)
        & (cycles["cycle_id"] <= benchmark.cycles_per_participant)
        & (cycles["cycle_length"] > benchmark.issue_day)
    ].dropna(subset=["cycle_length"])
    for offset, (_, cycle) in enumerate(eligible.iterrows()):
        history = cycles[(cycles["participant_id"] == cycle["participant_id"]) & (cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].dropna().to_numpy()
        stage_history = _proxy_stage_history(cohort.observed, cycles, str(cycle["participant_id"]), int(cycle["cycle_id"]))
        _, components, qualities, _ = _component_forecasts(
            cohort.observed,
            cycle,
            cfg,
            benchmark.issue_day,
            history,
            population_mean,
            population_sd,
            10000 + offset * 10000,
            stage_history,
        )
        records.append((components, qualities, float(cycle["cycle_length"]), offset))
    if not records:
        raise RuntimeError("no calibration forecasts were eligible")

    component_rows = []
    losses: dict[str, float] = {}
    for component in FUSION_COMPONENTS:
        metrics = [event_metrics(samples[component], truth, cfg.evaluation.interval_levels) for samples, _, truth, _ in records]
        frame = pd.DataFrame(metrics)
        losses[component] = float(frame["wis"].mean())
        component_rows.append({
            "component": component,
            "calibration_wis": losses[component],
            "calibration_crps": float(frame["crps"].mean()),
            "coverage_90": float(frame["coverage_90"].mean()),
            "n": len(frame),
        })

    minimum_loss = min(losses.values())
    conformal_coverage_target = float(min(1.0, np.ceil((len(records) + 1) * 0.90) / len(records)))
    rows: list[dict[str, float | int]] = []
    strategies: list[FusionStrategy] = []
    for temperature in benchmark.fusion_temperature_grid:
        raw = {name: float(np.exp(-temperature * (losses[name] - minimum_loss))) for name in FUSION_COMPONENTS}
        normalizer = sum(raw.values())
        base_weights = {name: raw[name] / normalizer for name in FUSION_COMPONENTS}
        for floor in benchmark.fusion_missingness_floor_grid:
            for spread in benchmark.fusion_spread_grid:
                uncalibrated = FusionStrategy(base_weights, floor, temperature, spread, 0.0)
                fused_records = [(_fuse_samples(samples, qualities, uncalibrated), truth, offset) for samples, qualities, truth, offset in records]
                for discrepancy in benchmark.discrepancy_grid:
                    metrics = []
                    for samples, truth, offset in fused_records:
                        calibrated = _calibrated_copy(samples, discrepancy, cfg.experiment.seed + 20000 + offset)
                        metrics.append(event_metrics(calibrated, truth, cfg.evaluation.interval_levels))
                    frame = pd.DataFrame(metrics)
                    coverage90 = float(frame["coverage_90"].mean())
                    coverage_error = abs(coverage90 - 0.90)
                    coverage_shortfall = max(conformal_coverage_target - coverage90, 0.0)
                    objective = float(frame["wis"].mean() + 5 * coverage_error)
                    strategy = FusionStrategy(base_weights, floor, temperature, spread, discrepancy)
                    strategies.append(strategy)
                    rows.append({
                        "temperature": temperature,
                        "missingness_floor": floor,
                        "spread_scale": spread,
                        "discrepancy_sd_days": discrepancy,
                        "calibration_wis": float(frame["wis"].mean()),
                        "coverage_90": coverage90,
                        "coverage_error_90": coverage_error,
                        "coverage_target_90": conformal_coverage_target,
                        "coverage_shortfall_90": coverage_shortfall,
                        "coverage_feasible": coverage_shortfall <= 1e-12,
                        "objective": objective,
                        "n": len(frame),
                        **{f"weight_{name}": base_weights[name] for name in FUSION_COMPONENTS},
                    })
    table = pd.DataFrame(rows)
    table["_strategy_index"] = np.arange(len(table))
    table = table.sort_values(
        ["coverage_feasible", "coverage_shortfall_90", "calibration_wis", "coverage_error_90", "discrepancy_sd_days"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    selected = strategies[int(table.iloc[0]["_strategy_index"])]
    table = table.drop(columns="_strategy_index")
    rolling_rows: list[dict[str, float | int | bool]] = []
    rolling_calibrations: list[MarginalCalibration] = []
    for spread in benchmark.fusion_spread_grid:
        for discrepancy in benchmark.discrepancy_grid:
            metrics = []
            for samples, _, truth, offset in records:
                calibrated = _calibrated_copy(
                    _spread_samples(samples["rolling_k3_prior"], spread),
                    discrepancy,
                    cfg.experiment.seed + 26000 + offset,
                )
                metrics.append(event_metrics(calibrated, truth, cfg.evaluation.interval_levels))
            frame = pd.DataFrame(metrics)
            coverage90 = float(frame["coverage_90"].mean())
            shortfall = max(conformal_coverage_target - coverage90, 0.0)
            rolling_calibrations.append(MarginalCalibration(spread, discrepancy))
            rolling_rows.append({
                "spread_scale": spread,
                "discrepancy_sd_days": discrepancy,
                "calibration_wis": float(frame["wis"].mean()),
                "coverage_90": coverage90,
                "coverage_target_90": conformal_coverage_target,
                "coverage_shortfall_90": shortfall,
                "coverage_feasible": shortfall <= 1e-12,
                "n": len(frame),
            })
    rolling_table = pd.DataFrame(rolling_rows)
    rolling_table["_calibration_index"] = np.arange(len(rolling_table))
    rolling_table = rolling_table.sort_values(
        ["coverage_feasible", "coverage_shortfall_90", "calibration_wis", "discrepancy_sd_days"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    rolling_selected = rolling_calibrations[int(rolling_table.iloc[0]["_calibration_index"])]
    rolling_table = rolling_table.drop(columns="_calibration_index")
    return selected, table, pd.DataFrame(component_rows).sort_values("calibration_wis").reset_index(drop=True), rolling_selected, rolling_table


def _renewal_samples(model: str, posterior: ParticipantDurationPosterior, population_sd: float, n: int, rng: np.random.Generator) -> np.ndarray:
    mean = posterior.mean_cycle_days
    sd = max(posterior.predictive_sd_days, 1.0)
    if model == "robust_renewal":
        return np.clip(mean + sd * rng.standard_t(4, n) / np.sqrt(2), 15, 90)
    if model == "lognormal_renewal":
        variance = sd**2
        sigma2 = np.log1p(variance / mean**2)
        mu = np.log(mean) - sigma2 / 2
        return np.clip(rng.lognormal(mu, np.sqrt(sigma2), n), 15, 90)
    if model == "gamma_renewal":
        shape = max(mean**2 / sd**2, 0.5)
        return np.clip(rng.gamma(shape, mean / shape, n), 15, 90)
    raise ValueError(model)


def _abstain(observations: pd.DataFrame, ovulation_samples: np.ndarray) -> tuple[bool, str]:
    observed = observations[observations["is_observed"].astype(bool) & observations["value"].notna()]
    lh = observed.loc[observed["signal_name"] == "lh", "value"].to_numpy(dtype=float)
    pdg = observed.loc[observed["signal_name"] == "pdg", "value"].to_numpy(dtype=float)
    missing_confirmation = lh.size > 0 and pdg.size > 0 and np.nanmax(lh) < 10 and np.nanmax(pdg) < 4
    finite = ovulation_samples[np.isfinite(ovulation_samples)]
    diffuse = finite.size < 0.9 * len(ovulation_samples) or (finite.size and np.quantile(finite, 0.95) - np.quantile(finite, 0.05) > 12)
    if missing_confirmation:
        return True, "hormone_transition_not_confirmed"
    if diffuse:
        return True, "diffuse_transition_posterior"
    return False, "sufficient_synthetic_evidence"


def _run_one(base: ExperimentConfig, benchmark: BenchmarkConfig, seed: int, scenario: str, run_dir: Path) -> None:
    cfg = _scenario_config(base, benchmark, seed, scenario)
    cohort = simulate_cohort(cfg)
    all_cycles = _cycle_table(cohort).copy()
    cycles = all_cycles.dropna(subset=["cycle_length"]).copy()
    split = _participant_split(cohort.participants["participant_id"].to_numpy(), seed, benchmark.train_fraction, benchmark.calibration_fraction)
    split["seed"] = seed
    split["scenario"] = scenario
    split.to_csv(run_dir / "locked_splits.csv", index=False)
    split_map = split.set_index("participant_id")["split"]
    cycles["split"] = cycles["participant_id"].map(split_map)
    training = cycles[cycles["split"] == "population_train"]
    training_ids = set(split.loc[split["split"] == "population_train", "participant_id"])
    population_mean = float(training["cycle_length"].mean())
    population_sd = float(max(training["cycle_length"].std(), 1.0))
    stage_population_f, stage_population_l, stage_between_f, stage_between_l = _training_stage_population(
        cohort.observed,
        cycles,
        training_ids,
        benchmark.cycles_per_participant,
    )
    renewal = HierarchicalRenewal().fit(training["cycle_length"].to_numpy())
    calibration_ids = set(split.loc[split["split"] == "calibration", "participant_id"])
    fusion, selection, component_selection, rolling_calibration, rolling_selection = _calibrate_fusion(
        cohort, cycles, calibration_ids, cfg, benchmark, population_mean, population_sd
    )
    discrepancy = fusion.discrepancy_sd_days
    selection.to_csv(run_dir / "calibration_selection.csv", index=False)
    component_selection.to_csv(run_dir / "calibration_components.csv", index=False)
    rolling_selection.to_csv(run_dir / "rolling_calibration_selection.csv", index=False)

    test_ids = set(split.loc[split["split"] == "locked_test", "participant_id"])
    eligibility = _eligibility_ledger(all_cycles, test_ids, benchmark, seed, scenario)
    eligibility.to_csv(run_dir / "eligibility.csv", index=False)
    eligible_keys = eligibility.loc[eligibility["eligible"], ["participant_id", "cycle_id"]]
    test_cycles = cycles.merge(eligible_keys, on=["participant_id", "cycle_id"], how="inner", validate="one_to_one")
    predictions: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    ablation_cycles = test_cycles if benchmark.ablation_cycles_per_run <= 0 else test_cycles.head(benchmark.ablation_cycles_per_run)
    ablation_keys = set(zip(ablation_cycles["participant_id"].astype(str), ablation_cycles["cycle_id"].astype(int)))
    for offset, (_, cycle) in enumerate(test_cycles.iterrows()):
        history = cycles[(cycles["participant_id"] == cycle["participant_id"]) & (cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].dropna().to_numpy()
        stage_history = _proxy_stage_history(cohort.observed, cycles, str(cycle["participant_id"]), int(cycle["cycle_id"]))
        posterior = ParticipantDurationPosterior.from_history(history, population_mean, population_sd)
        raw_ovulation, components, qualities, subset = _component_forecasts(
            cohort.observed,
            cycle,
            cfg,
            benchmark.issue_day,
            history,
            population_mean,
            population_sd,
            offset * 10000,
            stage_history,
        )
        ov_samples = _calibrated_copy(raw_ovulation, discrepancy, seed + 89001 + offset)
        fused_raw = _fuse_samples(components, qualities, fusion)
        menses_samples = _calibrated_copy(fused_raw, discrepancy, seed + 90001 + offset)
        finite = np.isfinite(ov_samples) & np.isfinite(menses_samples)
        menses_samples[finite] = np.maximum(menses_samples[finite], ov_samples[finite] + 1.0)
        realized_weights = _fusion_weights(fusion, qualities)
        identity = {"seed": seed, "scenario": scenario, "participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "history_cycles": len(history)}
        predictions.append({
            **identity,
            "model": "digital_twin",
            "target": "next_menses",
            **{f"fusion_weight_{name}": realized_weights[name] for name in FUSION_COMPONENTS},
            **event_metrics(menses_samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels),
        })
        predictions.append({**identity, "model": "digital_twin", "target": "latent_ovulation_transition", **event_metrics(ov_samples, float(cycle["follicular_duration"]), cfg.evaluation.interval_levels)})
        rng = np.random.default_rng(seed * 100000 + offset)
        baseline_samples = {
            "rolling_mean_k1": rng.normal(rolling_mean_forecast(0, history, 1), max(float(np.std(history)) if len(history) > 1 else population_sd, 1), benchmark.forecast_samples),
            "rolling_mean_k2": rng.normal(rolling_mean_forecast(0, history, 2), max(float(np.std(history)) if len(history) > 1 else population_sd, 1), benchmark.forecast_samples),
            "rolling_mean_k3": rng.normal(rolling_mean_forecast(0, history, 3), max(float(np.std(history)) if len(history) > 1 else population_sd, 1), benchmark.forecast_samples),
            "hierarchical_renewal": renewal.predictive_samples(0, history, benchmark.forecast_samples, rng),
            "robust_renewal": _renewal_samples("robust_renewal", posterior, population_sd, benchmark.forecast_samples, rng),
            "lognormal_renewal": _renewal_samples("lognormal_renewal", posterior, population_sd, benchmark.forecast_samples, rng),
            "gamma_renewal": _renewal_samples("gamma_renewal", posterior, population_sd, benchmark.forecast_samples, rng),
        }
        baseline_samples["calibrated_rolling_k3"] = _calibrated_copy(
            _spread_samples(components["rolling_k3_prior"], rolling_calibration.spread_scale),
            rolling_calibration.discrepancy_sd_days,
            seed + 28000 + offset,
        )
        daily = _daily_wide(subset)
        hsmm = HSMMBaseline(follicular_mean=posterior.follicular_prior_days(cfg.process.luteal_days), follicular_sd=max(posterior.predictive_sd_days, 1.5), luteal_mean=cfg.process.luteal_days, luteal_sd=2.0)
        baseline_samples["hsmm"] = hsmm.predictive_samples(daily, benchmark.forecast_samples, rng, discrepancy_sd_days=discrepancy)
        for model, samples in baseline_samples.items():
            predictions.append({**identity, "model": model, "target": "next_menses", **event_metrics(samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels)})
        if (str(cycle["participant_id"]), int(cycle["cycle_id"])) in ablation_keys:
            for modality in MODALITIES:
                raw = fused_raw if modality == "all_modalities" else components[modality]
                calibrated = _calibrated_copy(raw, discrepancy, seed + 30000 + offset * 10 + list(MODALITIES).index(modality))
                ablations.append({
                    **identity,
                    "modality": modality,
                    **event_metrics(calibrated, float(cycle["cycle_length"]), cfg.evaluation.interval_levels),
                })

    low_cfg = replace(cfg, missingness=replace(cfg.missingness, mechanism="mcar", mcar_probability=0.05, mar_probability=0, informative_probability=0, block_probability=0, hormone_schedule_days=2))
    high_cfg = replace(cfg, missingness=replace(cfg.missingness, mechanism="mixed", mcar_probability=0.35, mar_probability=0.25, informative_probability=0.18, block_probability=0.45, hormone_schedule_days=6))
    low_observed = apply_missingness(cohort.raw_observations, low_cfg, np.random.default_rng(seed + 40001))
    high_observed = apply_missingness(cohort.raw_observations, high_cfg, np.random.default_rng(seed + 40002))
    robustness = []
    for rob_offset, (_, cycle) in enumerate(test_cycles.head(benchmark.robustness_cycles_per_run).iterrows()):
        history = cycles[(cycles["participant_id"] == cycle["participant_id"]) & (cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].dropna().to_numpy()
        low_stage_history = _proxy_stage_history(low_observed, cycles, str(cycle["participant_id"]), int(cycle["cycle_id"]))
        high_stage_history = _proxy_stage_history(high_observed, cycles, str(cycle["participant_id"]), int(cycle["cycle_id"]))
        _, low_components, low_qualities, _ = _component_forecasts(low_observed, cycle, cfg, benchmark.issue_day, history, population_mean, population_sd, 50000 + rob_offset * 10000, low_stage_history)
        _, high_components, high_qualities, _ = _component_forecasts(high_observed, cycle, cfg, benchmark.issue_day, history, population_mean, population_sd, 60000 + rob_offset * 10000, high_stage_history)
        low_samples = _calibrated_copy(_fuse_samples(low_components, low_qualities, fusion), discrepancy, seed + 50000 + rob_offset)
        high_samples = _calibrated_copy(_fuse_samples(high_components, high_qualities, fusion), discrepancy, seed + 60000 + rob_offset)
        low_metrics = event_metrics(low_samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels)
        high_metrics = event_metrics(high_samples, float(cycle["cycle_length"]), cfg.evaluation.interval_levels)
        robustness.append({"seed": seed, "scenario": scenario, "participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "low_missing_width_90": low_metrics["width_90"], "high_missing_width_90": high_metrics["width_90"], "low_missing_crps": low_metrics["crps"], "high_missing_crps": high_metrics["crps"], "low_missing_coverage_90": low_metrics["coverage_90"], "high_missing_coverage_90": high_metrics["coverage_90"]})

    abstention = []
    for abst_offset, (_, cycle) in enumerate(test_cycles.iterrows()):
        history = cycles[(cycles["participant_id"] == cycle["participant_id"]) & (cycles["cycle_id"] < cycle["cycle_id"])]["cycle_length"].dropna().to_numpy()
        stage_history = _proxy_stage_history(cohort.observed, cycles, str(cycle["participant_id"]), int(cycle["cycle_id"]))
        ov_samples, _, subset = _run_twin(cohort.observed, cycle, cfg, benchmark.abstention_issue_day, history, population_mean, population_sd, discrepancy, 70000 + abst_offset, stage_history=stage_history)
        predicted, reason = _abstain(subset, ov_samples)
        truth_ambiguous = bool(scenario in {"anovulatory_like", "stalled_transition"} or cycle.get("ovulation_regime", "") == "anovulatory_like_regime")
        abstention.append({"seed": seed, "scenario": scenario, "participant_id": cycle["participant_id"], "cycle_id": int(cycle["cycle_id"]), "truth_ambiguous": truth_ambiguous, "abstained": predicted, "reason": reason})

    parameters = cohort.parameters.set_index("participant_id")
    recovery_rows: list[dict[str, Any]] = []
    for participant_id in sorted(test_ids):
        follicular_history, luteal_history = _proxy_stage_history(
            cohort.observed,
            cycles,
            participant_id,
            benchmark.cycles_per_participant + 1,
        )
        follicular_posterior = StageSpeedPosterior.from_durations(
            follicular_history,
            stage_population_f,
            stage_between_f,
        )
        luteal_posterior = StageSpeedPosterior.from_durations(
            luteal_history,
            stage_population_l,
            stage_between_l,
        )
        participant_cycles = cycles[
            (cycles["participant_id"] == participant_id)
            & (cycles["cycle_id"] <= benchmark.cycles_per_participant)
        ]
        # This retrospective recovery endpoint uses every completed, model-visible
        # cycle onset. It never feeds a forecast; forward forecasts above retain
        # their strict earlier-cycle histories.
        adaptation_history = participant_cycles["cycle_length"].to_numpy(dtype=float)
        duration_posterior = ParticipantDurationPosterior.from_history(
            adaptation_history,
            population_mean,
            population_sd,
        )
        fallback_follicular = np.clip(
            participant_cycles["cycle_length"].to_numpy(dtype=float) - np.exp(-stage_population_l),
            5.0,
            80.0,
        )
        length_posterior = StageSpeedPosterior.from_durations(
            fallback_follicular,
            stage_population_f,
            stage_between_f,
        )
        stage_signal_quality = _participant_stage_signal_quality(cohort.observed, participant_id)
        blended_follicular_speed = (
            stage_signal_quality * follicular_posterior.mean_log_speed
            + (1 - stage_signal_quality) * length_posterior.mean_log_speed
        )
        recovery_rows.append({
            "participant_id": participant_id,
            "estimated_follicular_log_speed": blended_follicular_speed,
            "estimated_luteal_log_speed": luteal_posterior.mean_log_speed,
            "follicular_posterior_sd": float(np.hypot(
                stage_signal_quality * follicular_posterior.posterior_sd,
                (1 - stage_signal_quality) * length_posterior.posterior_sd,
            )),
            "luteal_posterior_sd": luteal_posterior.posterior_sd,
            "follicular_proxy_cycles": follicular_posterior.observed_cycles,
            "luteal_proxy_cycles": luteal_posterior.observed_cycles,
            "follicular_reliability": follicular_posterior.reliability,
            "luteal_reliability": luteal_posterior.reliability,
            "stage_signal_quality": stage_signal_quality,
            "length_proxy_cycles": length_posterior.observed_cycles,
            "estimated_participant_cycle_days": duration_posterior.mean_cycle_days,
            "duration_history_cycles": duration_posterior.observed_cycles,
        })
    recovery = pd.DataFrame(recovery_rows)
    recovery["true_follicular_log_speed"] = recovery["participant_id"].map(parameters["mu_log_speed_f"])
    recovery["true_luteal_log_speed"] = recovery["participant_id"].map(parameters["mu_log_speed_l"])
    recovery["true_expected_cycle_days"] = np.exp(-recovery["true_follicular_log_speed"]) + np.exp(-recovery["true_luteal_log_speed"])
    recovery["seed"] = seed
    recovery["scenario"] = scenario
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(ablations).to_csv(run_dir / "ablations.csv", index=False)
    pd.DataFrame(robustness).to_csv(run_dir / "missingness.csv", index=False)
    pd.DataFrame(abstention).to_csv(run_dir / "abstention.csv", index=False)
    recovery.to_csv(run_dir / "observation_recovery.csv", index=False)
    (run_dir / "completed.json").write_text(json.dumps({
        "seed": seed,
        "scenario": scenario,
        "selected_discrepancy_sd_days": discrepancy,
        "fusion_temperature": fusion.temperature,
        "fusion_missingness_floor": fusion.missingness_floor,
        "fusion_spread_scale": fusion.spread_scale,
        "fusion_weights": fusion.weights,
        "rolling_calibration_spread_scale": rolling_calibration.spread_scale,
        "rolling_calibration_discrepancy_sd_days": rolling_calibration.discrepancy_sd_days,
        "planned_forecasts": int(len(eligibility)),
        "eligible_forecasts": int(eligibility["eligible"].sum()),
        "administratively_censored": int((eligibility["exclusion_reason"] == "administratively_censored").sum()),
        "completed_at": datetime.now(UTC).isoformat(),
    }, indent=2), encoding="utf-8")


def _summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["model", "target"], as_index=False).agg(
        n=("absolute_error", "count"),
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        crps=("crps", "mean"),
        wis=("wis", "mean"),
        coverage_90=("coverage_90", "mean"),
        width_90=("width_90", "mean"),
    )


def _seed_comparison_table(predictions: pd.DataFrame, summary: pd.DataFrame, seed: int) -> pd.DataFrame:
    next_menses = predictions[predictions["target"] == "next_menses"]
    by_seed = next_menses.groupby(["seed", "model"], as_index=False)["wis"].mean().pivot(index="seed", columns="model", values="wis")
    distributional = ["hierarchical_renewal", "robust_renewal", "lognormal_renewal", "gamma_renewal", "hsmm"]
    distributional_summary = summary[(summary["model"].isin(distributional)) & (summary["target"] == "next_menses")]
    best_model = str(distributional_summary.sort_values("wis").iloc[0]["model"])
    rolling_candidates = [name for name in ("rolling_mean_k3", "calibrated_rolling_k3") if name in by_seed]
    rolling_summary = summary[(summary["model"].isin(rolling_candidates)) & (summary["target"] == "next_menses")]
    best_rolling = str(rolling_summary.sort_values("wis").iloc[0]["model"])
    comparisons = {"best_renewal_hsmm": best_model, "rolling_mean_k3": best_rolling}
    rng = np.random.default_rng(seed + 440011)
    rows = []
    for label, comparator in comparisons.items():
        differences = (by_seed["digital_twin"] - by_seed[comparator]).dropna().to_numpy(dtype=float)
        bootstrap = np.asarray([
            rng.choice(differences, size=len(differences), replace=True).mean()
            for _ in range(20000)
        ])
        rows.append({
            "comparison": label,
            "comparator_model": comparator,
            "mean_wis_difference": float(differences.mean()),
            "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
            "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
            "seed_wins": int((differences <= 0).sum()),
            "n_seeds": int(len(differences)),
        })
    return pd.DataFrame(rows)


def run_benchmark(config_path: str | Path, project_root: Path, force: bool = False) -> Path:
    benchmark = load_benchmark_config(config_path)
    base_path = project_root / benchmark.base_config
    base = load_experiment_config(base_path)
    output = (project_root / benchmark.output_dir).resolve()
    runs_dir = output / "runs"
    tables_dir = output / "tables"
    runs_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(benchmark.scenarios) * len(benchmark.seeds)
    run_index = 0
    for scenario in benchmark.scenarios:
        for seed in benchmark.seeds:
            run_index += 1
            run_dir = runs_dir / f"{scenario}_seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if benchmark.resume and not force and (run_dir / "completed.json").exists():
                print(f"[{run_index}/{total_runs}] reuse {scenario} seed={seed}", flush=True)
                continue
            print(f"[{run_index}/{total_runs}] run {scenario} seed={seed}", flush=True)
            _run_one(base, benchmark, seed, scenario, run_dir)
            completed_now = len(list(runs_dir.glob("*/completed.json")))
            (output / "benchmark_progress.json").write_text(json.dumps({"completed_runs": completed_now, "expected_runs": total_runs, "last_completed": {"scenario": scenario, "seed": seed}, "updated_at": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")

    def combine(name: str) -> pd.DataFrame:
        files = sorted(runs_dir.glob(f"*/{name}.csv"))
        if not files:
            raise RuntimeError(f"No completed {name} outputs")
        return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    predictions = combine("predictions")
    ablations = combine("ablations")
    missingness = combine("missingness")
    abstention_cases = combine("abstention")
    recovery_cases = combine("observation_recovery")
    eligibility_files = sorted(runs_dir.glob("*/eligibility.csv"))
    eligibility = pd.concat([pd.read_csv(path) for path in eligibility_files], ignore_index=True) if eligibility_files else pd.DataFrame()
    core_predictions = predictions[~predictions["scenario"].isin(benchmark.safety_scenarios)]
    core_ablations = ablations[~ablations["scenario"].isin(benchmark.safety_scenarios)]
    core_missingness = missingness[~missingness["scenario"].isin(benchmark.safety_scenarios)]
    core_recovery = recovery_cases[~recovery_cases["scenario"].isin(benchmark.safety_scenarios)]
    core_eligibility = eligibility[~eligibility["scenario"].isin(benchmark.safety_scenarios)] if not eligibility.empty else eligibility
    summary = _summary(core_predictions)
    scenario_summary = predictions.groupby(["scenario", "model", "target"], as_index=False).agg(n=("absolute_error", "count"), mae=("absolute_error", "mean"), crps=("crps", "mean"), wis=("wis", "mean"), coverage_90=("coverage_90", "mean"), width_90=("width_90", "mean"))
    twin_predictions = core_predictions[(core_predictions["model"] == "digital_twin") & (core_predictions["target"] == "next_menses")]
    # Stored central intervals make locked-test coverage directly auditable.
    calibration = pd.DataFrame([
        {"nominal": level, "empirical": float(twin_predictions[f"coverage_{int(level * 100)}"].mean()), "calibration_error": float(twin_predictions[f"coverage_{int(level * 100)}"].mean() - level), "n": len(twin_predictions)}
        for level in base.evaluation.interval_levels
    ])
    scenario_calibration = twin_predictions.groupby("scenario", as_index=False).agg(
        empirical_90=("coverage_90", "mean"),
        n=("coverage_90", "size"),
    )
    scenario_calibration["nominal_90"] = 0.90
    scenario_calibration["absolute_error_90"] = (scenario_calibration["empirical_90"] - 0.90).abs()
    recovery = pd.DataFrame([{
        "follicular_correlation": float(core_recovery[["true_follicular_log_speed", "estimated_follicular_log_speed"]].corr().iloc[0, 1]),
        "luteal_correlation": float(core_recovery[["true_luteal_log_speed", "estimated_luteal_log_speed"]].corr().iloc[0, 1]),
        "participant_duration_correlation": float(core_recovery[["true_expected_cycle_days", "estimated_participant_cycle_days"]].corr().iloc[0, 1]) if "estimated_participant_cycle_days" in core_recovery else np.nan,
        "n_participant_runs": len(core_recovery),
    }])
    recovery_by_scenario_rows = []
    for scenario, frame in core_recovery.groupby("scenario"):
        recovery_by_scenario_rows.append({
            "scenario": scenario,
            "follicular_correlation": float(frame["estimated_follicular_log_speed"].corr(frame["true_follicular_log_speed"])),
            "luteal_correlation": float(frame["estimated_luteal_log_speed"].corr(frame["true_luteal_log_speed"])),
            "participant_duration_correlation": float(frame["estimated_participant_cycle_days"].corr(frame["true_expected_cycle_days"])) if "estimated_participant_cycle_days" in frame else np.nan,
            "mean_follicular_proxy_cycles": float(frame["follicular_proxy_cycles"].mean()) if "follicular_proxy_cycles" in frame else np.nan,
            "mean_luteal_proxy_cycles": float(frame["luteal_proxy_cycles"].mean()) if "luteal_proxy_cycles" in frame else np.nan,
            "mean_follicular_reliability": float(frame["follicular_reliability"].mean()) if "follicular_reliability" in frame else np.nan,
            "mean_luteal_reliability": float(frame["luteal_reliability"].mean()) if "luteal_reliability" in frame else np.nan,
            "n": len(frame),
        })
    recovery_by_scenario = pd.DataFrame(recovery_by_scenario_rows)
    positives = abstention_cases["truth_ambiguous"].astype(bool)
    predicted = abstention_cases["abstained"].astype(bool)
    sensitivity = float(predicted[positives].mean()) if positives.any() else np.nan
    specificity = float((~predicted[~positives]).mean()) if (~positives).any() else np.nan
    abstention = pd.DataFrame({"metric": ["sensitivity", "specificity", "positive_cases", "negative_cases"], "value": [sensitivity, specificity, int(positives.sum()), int((~positives).sum())]})
    ablation_summary = core_ablations.groupby("modality", as_index=False).agg(n=("absolute_error", "count"), mae=("absolute_error", "mean"), crps=("crps", "mean"), wis=("wis", "mean"), coverage_90=("coverage_90", "mean"), width_90=("width_90", "mean"))
    seed_comparisons = _seed_comparison_table(core_predictions, summary, base.experiment.seed)
    if not core_eligibility.empty:
        scored_forecasts = int(len(twin_predictions))
        completeness = pd.DataFrame([{
            "planned_forecasts": int(len(core_eligibility)),
            "eligible_forecasts": int(core_eligibility["eligible"].astype(bool).sum()),
            "scored_forecasts": scored_forecasts,
            "administratively_censored": int((core_eligibility["exclusion_reason"] == "administratively_censored").sum()),
            "event_before_issue_day": int((core_eligibility["exclusion_reason"] == "event_before_issue_day").sum()),
            "eligible_fraction": float(core_eligibility["eligible"].astype(bool).mean()),
            "minimum_eligible_fraction": benchmark.minimum_eligible_fraction,
        }])
    else:
        completeness = pd.DataFrame()
    weight_columns = [f"fusion_weight_{name}" for name in FUSION_COMPONENTS if f"fusion_weight_{name}" in twin_predictions]
    fusion_weights = pd.DataFrame({
        "component": [name.removeprefix("fusion_weight_") for name in weight_columns],
        "mean_weight": [float(twin_predictions[name].mean()) for name in weight_columns],
        "median_weight": [float(twin_predictions[name].median()) for name in weight_columns],
    })
    gates = evaluate_scientific_gates(
        summary,
        calibration,
        core_missingness,
        recovery,
        abstention,
        ablation_summary,
        scenario_calibration,
        seed_comparisons,
        completeness,
    )
    personalization = core_predictions[(core_predictions["model"] == "digital_twin") & (core_predictions["target"] == "next_menses")].groupby("history_cycles", as_index=False).agg(n=("absolute_error", "count"), mae=("absolute_error", "mean"), crps=("crps", "mean"), wis=("wis", "mean"), coverage_90=("coverage_90", "mean"), width_90=("width_90", "mean"))
    missingness_summary = pd.DataFrame([{
        "n": len(core_missingness),
        "low_width_90": float(core_missingness["low_missing_width_90"].mean()),
        "high_width_90": float(core_missingness["high_missing_width_90"].mean()),
        "width_ratio": float(core_missingness["high_missing_width_90"].mean() / core_missingness["low_missing_width_90"].mean()),
        "low_coverage_90": float(core_missingness["low_missing_coverage_90"].mean()),
        "high_coverage_90": float(core_missingness["high_missing_coverage_90"].mean()),
    }])
    for name, frame in {
        "locked_test_results": summary,
        "results_by_scenario": scenario_summary,
        "locked_test_calibration": calibration,
        "calibration_by_scenario": scenario_calibration,
        "cohort_modality_ablation": ablation_summary,
        "missingness_robustness": missingness_summary,
        "observation_based_recovery": recovery,
        "recovery_by_scenario": recovery_by_scenario,
        "seed_cluster_comparisons": seed_comparisons,
        "forecast_completeness": completeness,
        "fusion_weights": fusion_weights,
        "personalization_curve": personalization,
        "abstention": abstention,
        "scientific_gates": gates,
    }.items():
        _write_table(frame, tables_dir, name)
    passed = bool(gates["passed"].all())
    report = f"""# Leakage-safe multi-seed synthetic benchmark

Generated: {datetime.now(UTC).isoformat()}

- Seeds: {list(benchmark.seeds)}
- Scenarios: {list(benchmark.scenarios)}
- Safety scenarios excluded from ordinary predictive gates: {list(benchmark.safety_scenarios)}
- Participants per run: {benchmark.participants}
- Population/calibration/locked-test participants are disjoint in every run.
- Earlier cycles personalize held-out participants; later cycles are evaluated.
- Forecast discrepancy is selected on calibration participants only.

## Locked-test results

{summary.to_markdown(index=False)}

## Locked-test calibration

{calibration.to_markdown(index=False)}

## Calibration by scenario

{scenario_calibration.to_markdown(index=False)}

## Results by scenario, including safety stress tests

{scenario_summary.to_markdown(index=False)}

## Cohort-wide modality ablation

{ablation_summary.to_markdown(index=False)}

## Missingness robustness

{missingness_summary.to_markdown(index=False)}

## Observation-based recovery

{recovery.to_markdown(index=False)}

## Recovery identifiability by scenario

{recovery_by_scenario.to_markdown(index=False)}

## Seed-cluster uncertainty

{seed_comparisons.to_markdown(index=False)}

## Forecast completeness and censoring

{completeness.to_markdown(index=False) if not completeness.empty else 'Not available for this legacy benchmark output.'}

## Calibration-trained fusion weights

{fusion_weights.to_markdown(index=False) if not fusion_weights.empty else 'Not available for this legacy benchmark output.'}

## Forward-time personalization

{personalization.to_markdown(index=False)}

## Abstention

{abstention.to_markdown(index=False)}

## Scientific gates

{gates.to_markdown(index=False)}

Overall benchmark decision: **{'PASS' if passed else 'FAIL / DO NOT PROMOTE'}**.

Failure is a valid scientific result. Do not tune against these locked-test outputs; revise using new development seeds and create a new locked test.
"""
    (output / "BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")
    (output / "benchmark_status.json").write_text(json.dumps({"passed_all_gates": passed, "completed_runs": len(list(runs_dir.glob("*/completed.json"))), "expected_runs": len(benchmark.seeds) * len(benchmark.scenarios), "generated_at": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")
    (output / "benchmark_progress.json").write_text(json.dumps({"completed_runs": total_runs, "expected_runs": total_runs, "status": "aggregated", "updated_at": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")
    return output
