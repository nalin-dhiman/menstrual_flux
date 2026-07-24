from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from digital_twin.baselines.hsmm import HSMMBaseline
from digital_twin.baselines.renewal import HierarchicalRenewal
from digital_twin.config import load_experiment_config
from digital_twin.evaluation.metrics import event_metrics, posterior_mass_in_interval
from digital_twin.inference.forecast import forecast_events
from digital_twin.inference.particle_filter import run_particle_filter
from digital_twin.inference.personalization import ParticipantDurationPosterior
from digital_twin.real_data.mcphases_audit import _load_verified_tables


SUPPORTED_OBSERVATION_NAMES = {
    "nightly_skin_temperature": "temperature",
    "resting_heart_rate": "resting_heart_rate",
    "sleep_hrv_rmssd": "hrv",
    "lh": "lh",
    "e3g": "e3g",
    "pdg": "pdg",
    "bleeding_reported": "bleeding",
}
CODE_PATHS = (
    "src/digital_twin/real_data/mcphases_benchmark.py",
    "src/digital_twin/baselines/renewal.py",
    "src/digital_twin/baselines/hsmm.py",
    "src/digital_twin/inference/particle_filter.py",
    "src/digital_twin/inference/forecast.py",
    "src/digital_twin/evaluation/metrics.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"source_dataset", "seed", "split", "tasks", "evaluation", "digital_twin"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Real-data protocol is missing keys: {missing}")
    return config


def _allocate_stratum(
    participant_ids: list[str],
    train_fraction: float,
    calibration_fraction: float,
    rng: np.random.Generator,
) -> dict[str, str]:
    values = np.asarray(sorted(participant_ids), dtype=object)
    rng.shuffle(values)
    count = len(values)
    n_train = int(round(count * train_fraction))
    n_calibration = int(round(count * calibration_fraction))
    if count >= 3:
        n_train = min(max(n_train, 1), count - 2)
        n_calibration = min(max(n_calibration, 1), count - n_train - 1)
    labels = (
        ["population_train"] * n_train
        + ["calibration"] * n_calibration
        + ["locked_test"] * (count - n_train - n_calibration)
    )
    return {str(participant): label for participant, label in zip(values, labels)}


def freeze_mcphases_protocol(
    curated_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> Path:
    """Freeze outcome-blind participant splits using interval availability only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "participant_splits.parquet"
    manifest_path = output_dir / "protocol_manifest.json"
    if split_path.exists() or manifest_path.exists():
        raise FileExistsError("The mcPHASES participant protocol is already frozen")

    tables, curated_manifest = _load_verified_tables(Path(curated_dir))
    config = _load_protocol(Path(config_path))
    quality = tables["data_quality"][["participant_id", "study_interval"]].drop_duplicates()
    strata = (
        quality.assign(has_interval_2024=quality["study_interval"].eq(2024))
        .groupby("participant_id", as_index=False)["has_interval_2024"]
        .any()
    )
    split_cfg = config["split"]
    assignments: dict[str, str] = {}
    for index, (_, group) in enumerate(strata.groupby("has_interval_2024", sort=True)):
        rng = np.random.default_rng(int(config["seed"]) + 7919 * (index + 1))
        assignments.update(
            _allocate_stratum(
                group["participant_id"].astype(str).tolist(),
                float(split_cfg["train_fraction"]),
                float(split_cfg["calibration_fraction"]),
                rng,
            )
        )
    splits = strata.copy()
    splits["split"] = splits["participant_id"].map(assignments)
    splits["split_seed"] = int(config["seed"])
    splits["stratification_variables"] = "has_interval_2024"
    splits.to_parquet(split_path, index=False)

    counts = (
        splits.groupby(["has_interval_2024", "split"])
        .size()
        .rename("participants")
        .reset_index()
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_dataset": config["source_dataset"],
        "outcome_blind": True,
        "variables_used_for_split": ["participant_id", "has_interval_2024"],
        "variables_not_used": [
            "cycle_length",
            "hormone_values",
            "wearable_values",
            "phase_transition_timing",
            "forecast_outcomes",
        ],
        "config_sha256": _sha256(Path(config_path)),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "split_sha256": _sha256(split_path),
        "aggregate_split_counts": [
            {
                "has_interval_2024": bool(row["has_interval_2024"]),
                "split": str(row["split"]),
                "participants": int(row["participants"]),
            }
            for _, row in counts.iterrows()
        ],
        "locked_test_policy": "No locked-test outcome is evaluated before model_freeze.json exists.",
        "curated_manifest_created_utc": curated_manifest.get("created_utc"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path.resolve()


def _verify_protocol(
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(protocol_dir) / "protocol_manifest.json"
    split_path = Path(protocol_dir) / "participant_splits.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "config_sha256": _sha256(Path(config_path)),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "split_sha256": _sha256(split_path),
    }
    for key, actual in checks.items():
        if manifest.get(key) != actual:
            raise ValueError(f"Frozen protocol verification failed: {key}")
    return pd.read_parquet(split_path), manifest


def _transition_targets(
    references: pd.DataFrame,
    cycles: pd.DataFrame,
) -> pd.DataFrame:
    transition = references.loc[
        references["event_type"].eq("mira_fertility_to_luteal_transition")
        & ~references["cycle_id"].astype(str).str.endswith("-PRE")
    ].copy()
    counts = transition.groupby(["participant_id", "study_interval", "cycle_id"]).size()
    unique_keys = counts[counts.eq(1)].index
    transition = transition.set_index(["participant_id", "study_interval", "cycle_id"]).loc[
        unique_keys
    ].reset_index()
    starts = cycles[
        [
            "participant_id",
            "study_interval",
            "cycle_id",
            "cycle_start_day",
            "cycle_length_days",
            "eligible_for_primary_evaluation",
        ]
    ]
    transition = transition.merge(
        starts,
        on=["participant_id", "study_interval", "cycle_id"],
        how="inner",
        validate="one_to_one",
    )
    epoch = pd.Timestamp("2000-01-01 12:00:00")
    lower_study_day = (
        (pd.to_datetime(transition["event_time_lower"]) - epoch).dt.days + 1
    )
    upper_study_day = (
        (pd.to_datetime(transition["event_time_upper"]) - epoch).dt.days + 1
    )
    transition["target_lower_day"] = lower_study_day - transition["cycle_start_day"] + 1
    transition["target_upper_day"] = upper_study_day - transition["cycle_start_day"] + 1
    transition["target_midpoint_day"] = (
        transition["target_lower_day"] + transition["target_upper_day"]
    ) / 2
    return transition


def _population_parameters(
    cycles: pd.DataFrame,
    transitions: pd.DataFrame,
    training_ids: set[str],
) -> dict[str, float]:
    train_cycles = cycles.loc[
        cycles["participant_id"].isin(training_ids)
        & cycles["eligible_for_primary_evaluation"]
    ].copy()
    values = train_cycles["cycle_length_days"].to_numpy(dtype=float)
    if len(values) < 2:
        raise RuntimeError("Insufficient population-training cycles")
    train_transitions = transitions.loc[
        transitions["participant_id"].isin(training_ids)
    ].copy()
    follicular = train_transitions["target_midpoint_day"].to_numpy(dtype=float)
    matched = train_transitions.dropna(subset=["cycle_length_days"])
    luteal = (
        matched["cycle_length_days"] - matched["target_midpoint_day"]
    ).to_numpy(dtype=float)
    luteal = luteal[np.isfinite(luteal) & (luteal > 2)]
    return {
        "cycle_mean": float(np.mean(values)),
        "cycle_sd": float(max(np.std(values, ddof=1), 1.0)),
        "transition_mean": float(np.mean(follicular)),
        "transition_sd": float(max(np.std(follicular, ddof=1), 1.0)),
        "follicular_mean": float(np.mean(follicular)),
        "follicular_sd": float(max(np.std(follicular, ddof=1), 1.0)),
        "luteal_mean": float(np.mean(luteal)) if len(luteal) else 13.0,
        "luteal_sd": float(max(np.std(luteal, ddof=1), 1.0)) if len(luteal) > 1 else 2.0,
        "training_cycle_intervals": int(len(values)),
        "training_transition_intervals": int(len(follicular)),
    }


def _history(
    table: pd.DataFrame,
    participant_id: str,
    interval: int,
    start_day: float,
    value_column: str,
    start_column: str,
) -> np.ndarray:
    values = table.loc[
        table["participant_id"].eq(participant_id)
        & table["study_interval"].eq(interval)
        & table[start_column].lt(start_day),
        [start_column, value_column],
    ].sort_values(start_column)[value_column]
    return values.dropna().to_numpy(dtype=float)


def _cases(
    task: str,
    participant_ids: set[str],
    cycles: pd.DataFrame,
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if task == "next_menses":
        task_cfg = config["tasks"]["next_menses"]
        eligible = cycles.loc[
            cycles["participant_id"].isin(participant_ids)
            & cycles["eligible_for_primary_evaluation"]
            & cycles["cycle_length_days"].gt(float(task_cfg["issue_day"]))
        ].sort_values(["participant_id", "study_interval", "cycle_start_day"])
        for _, row in eligible.iterrows():
            history = _history(
                cycles.loc[cycles["eligible_for_primary_evaluation"]],
                str(row["participant_id"]),
                int(row["study_interval"]),
                float(row["cycle_start_day"]),
                "cycle_length_days",
                "cycle_start_day",
            )
            if len(history) < int(task_cfg["minimum_history_cycles"]):
                continue
            lower = upper = float(row["cycle_length_days"])
            cases.append(
                {
                    "task": task,
                    "participant_id": str(row["participant_id"]),
                    "study_interval": int(row["study_interval"]),
                    "cycle_id": str(row["cycle_id"]),
                    "cycle_start_day": int(row["cycle_start_day"]),
                    "issue_day": int(task_cfg["issue_day"]),
                    "history": history,
                    "target_lower": lower,
                    "target_upper": upper,
                    "target_midpoint": lower,
                }
            )
    elif task == "device_transition":
        task_cfg = config["tasks"]["device_transition"]
        eligible = transitions.loc[
            transitions["participant_id"].isin(participant_ids)
            & transitions["target_upper_day"].gt(float(task_cfg["issue_day"]))
        ].sort_values(["participant_id", "study_interval", "cycle_start_day"])
        for _, row in eligible.iterrows():
            history = _history(
                transitions,
                str(row["participant_id"]),
                int(row["study_interval"]),
                float(row["cycle_start_day"]),
                "target_midpoint_day",
                "cycle_start_day",
            )
            cases.append(
                {
                    "task": task,
                    "participant_id": str(row["participant_id"]),
                    "study_interval": int(row["study_interval"]),
                    "cycle_id": str(row["cycle_id"]),
                    "cycle_start_day": int(row["cycle_start_day"]),
                    "issue_day": int(task_cfg["issue_day"]),
                    "history": history,
                    "target_lower": float(row["target_lower_day"]),
                    "target_upper": float(row["target_upper_day"]),
                    "target_midpoint": float(row["target_midpoint_day"]),
                }
            )
    else:
        raise ValueError(task)

    for case in cases:
        case["observations"] = observations.loc[
            observations["participant_id"].eq(case["participant_id"])
            & observations["cycle_id"].eq(case["cycle_id"])
            & observations["day_in_study"].between(
                case["cycle_start_day"],
                case["cycle_start_day"] + case["issue_day"] - 1,
            )
        ].copy()
    return cases


def _seed(config_seed: int, case: dict[str, Any], model: str) -> int:
    key = (
        f"{config_seed}:{case['task']}:{case['participant_id']}:"
        f"{case['cycle_id']}:{model}"
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _model_observations(case: dict[str, Any], signals: set[str]) -> pd.DataFrame:
    frame = case["observations"].loc[
        case["observations"]["signal_name"].isin(SUPPORTED_OBSERVATION_NAMES)
    ].copy()
    frame["signal_name"] = frame["signal_name"].map(SUPPORTED_OBSERVATION_NAMES)
    frame = frame.loc[frame["signal_name"].isin(signals)]
    frame["is_observed"] = frame["value"].notna()
    return frame


def _hsmm_daily(case: dict[str, Any]) -> pd.DataFrame:
    frame = _model_observations(
        case, {"temperature", "lh", "bleeding"}
    )
    observed = frame.loc[frame["value"].notna()].copy()
    observed["cycle_day"] = (
        observed["day_in_study"] - case["cycle_start_day"] + 1
    ).astype(int)
    daily = observed.pivot_table(
        index="cycle_day",
        columns="signal_name",
        values="value",
        aggfunc="last",
    ).reindex(range(1, case["issue_day"] + 1))
    if "temperature" in daily and daily["temperature"].notna().any():
        baseline = float(daily["temperature"].dropna().head(5).median())
        daily["temperature"] = 36.45 + daily["temperature"] - baseline
    return daily.reset_index(drop=True)


def _runtime_twin_config(
    project_root: Path,
    config: dict[str, Any],
    parameters: dict[str, float],
):
    cfg = load_experiment_config(project_root / config["digital_twin"]["base_config"])
    twin = config["digital_twin"]
    return replace(
        cfg,
        experiment=replace(cfg.experiment, seed=int(config["seed"])),
        process=replace(
            cfg.process,
            follicular_days=parameters["follicular_mean"],
            luteal_days=parameters["luteal_mean"],
        ),
        inference=replace(
            cfg.inference,
            particles=int(twin["particles"]),
            forecast_samples=int(twin["forecast_samples"]),
            forecast_horizon_days=int(twin["forecast_horizon_days"]),
            forecast_model_discrepancy_sd_days=0.0,
        ),
    )


def _raw_forecasts(
    case: dict[str, Any],
    parameters: dict[str, float],
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, np.ndarray]:
    sample_count = int(config["evaluation"]["forecast_samples"])
    history = np.asarray(case["history"], dtype=float)
    posterior = ParticipantDurationPosterior.from_history(
        history,
        parameters["cycle_mean"],
        parameters["cycle_sd"],
    )
    task = case["task"]
    models: dict[str, np.ndarray] = {}
    if task == "next_menses":
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "population_calendar"))
        models["population_calendar"] = np.clip(
            rng.normal(parameters["cycle_mean"], parameters["cycle_sd"], sample_count),
            15,
            60,
        )
        rolling_mean = float(np.mean(history[-int(config["tasks"]["next_menses"]["rolling_window"]):]))
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "rolling_up_to_k3"))
        models["rolling_up_to_k3"] = np.clip(
            rng.normal(rolling_mean, parameters["cycle_sd"], sample_count), 15, 60
        )
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "population_personal_history"))
        models["population_personal_history"] = np.clip(
            rng.normal(
                posterior.mean_cycle_days,
                posterior.predictive_sd_days,
                sample_count,
            ),
            15,
            60,
        )
        renewal = HierarchicalRenewal(
            population_mean=parameters["cycle_mean"],
            population_sd=parameters["cycle_sd"],
        )
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "hierarchical_renewal"))
        models["hierarchical_renewal"] = renewal.predictive_samples(
            0.0, history, sample_count, rng
        )
        hsmm = HSMMBaseline(
            follicular_mean=posterior.follicular_prior_days(parameters["luteal_mean"]),
            follicular_sd=max(posterior.predictive_sd_days, 1.5),
            luteal_mean=parameters["luteal_mean"],
            luteal_sd=parameters["luteal_sd"],
        )
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "hsmm"))
        models["hsmm"] = hsmm.predictive_samples(
            _hsmm_daily(case), sample_count, rng, horizon=60
        )
    else:
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "population_phase_prior"))
        models["population_phase_prior"] = np.clip(
            rng.normal(
                parameters["transition_mean"],
                parameters["transition_sd"],
                sample_count,
            ),
            case["issue_day"],
            50,
        )
        if len(history):
            center = float(
                np.mean(
                    history[-int(config["tasks"]["device_transition"]["rolling_window"]):]
                )
            )
        else:
            center = parameters["transition_mean"]
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "rolling_transition_up_to_k3"))
        models["rolling_transition_up_to_k3"] = np.clip(
            rng.normal(center, parameters["transition_sd"], sample_count),
            case["issue_day"],
            50,
        )
        hsmm = HSMMBaseline(
            follicular_mean=parameters["follicular_mean"],
            follicular_sd=parameters["follicular_sd"],
            luteal_mean=parameters["luteal_mean"],
            luteal_sd=parameters["luteal_sd"],
        )
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "hsmm_transition"))
        models["hsmm_transition"] = hsmm.predictive_transition_samples(
            _hsmm_daily(case), sample_count, rng
        )

    twin_cfg = _runtime_twin_config(project_root, config, parameters)
    for modality, signal_names in config["digital_twin"]["modalities"].items():
        model_name = f"digital_twin_{modality}"
        model_frame = _model_observations(case, set(signal_names))
        if model_frame.loc[model_frame["value"].notna()].empty:
            models[model_name] = np.full(sample_count, np.nan)
            continue
        result = run_particle_filter(
            model_frame,
            twin_cfg,
            seed_offset=_seed(int(config["seed"]), case, model_name) % 1_000_000,
            prior_follicular_days=posterior.follicular_prior_days(
                parameters["luteal_mean"]
            ),
            prior_luteal_days=parameters["luteal_mean"],
        )
        forecast = forecast_events(
            result,
            twin_cfg,
            seed_offset=_seed(int(config["seed"]), case, model_name + "_forecast")
            % 1_000_000,
        )
        models[model_name] = (
            forecast.next_menses_samples
            if task == "next_menses"
            else forecast.ovulation_samples
        )
    return models


def _fit_calibrators(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    calibrators: dict[str, dict[str, float]] = {}
    for (task, model), group in records.groupby(["task", "model"]):
        residual = group["target_midpoint"] - group["raw_median"]
        raw_sd = group["raw_sd"].replace(0, np.nan)
        bias = float(residual.mean()) if len(group) else 0.0
        residual_sd = float(residual.std(ddof=1)) if len(group) > 1 else np.nan
        typical_raw_sd = float(raw_sd.median()) if raw_sd.notna().any() else np.nan
        if not np.isfinite(residual_sd) or not np.isfinite(typical_raw_sd):
            scale = 1.0
        else:
            scale = float(np.clip(residual_sd / max(typical_raw_sd, 0.25), 0.5, 4.0))
        calibrators[f"{task}|{model}"] = {
            "bias_days": bias,
            "spread_scale": scale,
            "calibration_forecasts": int(len(group)),
            "calibration_participants": int(group["participant_id"].nunique()),
        }
    return calibrators


def _apply_calibrator(
    samples: np.ndarray,
    calibrator: dict[str, float],
) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values
    median = float(np.median(values[finite]))
    calibrated = values.copy()
    calibrated[finite] = (
        median
        + float(calibrator["bias_days"])
        + float(calibrator["spread_scale"]) * (values[finite] - median)
    )
    return calibrated


def _metric_row(
    case: dict[str, Any],
    model: str,
    samples: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    metrics = event_metrics(samples, case["target_midpoint"], levels)
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    median = float(np.median(finite)) if len(finite) else np.nan
    lower = case["target_lower"]
    upper = case["target_upper"]
    distance = (
        max(lower - median, median - upper, 0.0) if np.isfinite(median) else np.nan
    )
    return {
        "participant_id": case["participant_id"],
        "study_interval": case["study_interval"],
        "cycle_id": case["cycle_id"],
        "task": case["task"],
        "model": model,
        "issue_day": case["issue_day"],
        "history_cycles": len(case["history"]),
        "target_lower": lower,
        "target_upper": upper,
        "target_midpoint": case["target_midpoint"],
        "interval_absolute_error": distance,
        "posterior_mass_in_reference_interval": posterior_mass_in_interval(
            finite, lower, upper
        ),
        **metrics,
    }


def _code_hashes(project_root: Path) -> dict[str, str]:
    return {
        relative: _sha256(Path(project_root) / relative)
        for relative in CODE_PATHS
    }


def run_mcphases_development(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    output_dir: Path,
) -> Path:
    """Fit population parameters and calibration without reading locked outcomes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "model_freeze.json"
    if freeze_path.exists():
        raise FileExistsError("The mcPHASES development model is already frozen")
    tables, _ = _load_verified_tables(Path(curated_dir))
    splits, protocol_manifest = _verify_protocol(
        curated_dir, config_path, protocol_dir
    )
    config = _load_protocol(config_path)
    split_map = splits.set_index("participant_id")["split"]
    training_ids = set(split_map[split_map.eq("population_train")].index.astype(str))
    calibration_ids = set(split_map[split_map.eq("calibration")].index.astype(str))

    # Locked-test identifiers are not passed to case construction and their
    # cycle/transition outcomes are never included in any development object.
    cycles = tables["cycles"].loc[
        tables["cycles"]["participant_id"].isin(training_ids | calibration_ids)
    ].copy()
    references = tables["reference_intervals"].loc[
        tables["reference_intervals"]["participant_id"].isin(
            training_ids | calibration_ids
        )
    ].copy()
    observations = tables["daily_observations"].loc[
        tables["daily_observations"]["participant_id"].isin(
            training_ids | calibration_ids
        )
    ].copy()
    transitions = _transition_targets(references, cycles)
    parameters = _population_parameters(cycles, transitions, training_ids)

    calibration_cases = (
        _cases(
            "next_menses",
            calibration_ids,
            cycles,
            transitions,
            observations,
            config,
        )
        + _cases(
            "device_transition",
            calibration_ids,
            cycles,
            transitions,
            observations,
            config,
        )
    )
    raw_rows: list[dict[str, Any]] = []
    raw_cache: list[tuple[dict[str, Any], str, np.ndarray]] = []
    for case in calibration_cases:
        for model, samples in _raw_forecasts(
            case, parameters, config, Path(project_root)
        ).items():
            finite = np.asarray(samples, dtype=float)
            finite = finite[np.isfinite(finite)]
            raw_rows.append(
                {
                    "participant_id": case["participant_id"],
                    "study_interval": case["study_interval"],
                    "cycle_id": case["cycle_id"],
                    "task": case["task"],
                    "model": model,
                    "target_midpoint": case["target_midpoint"],
                    "raw_median": float(np.median(finite)) if len(finite) else np.nan,
                    "raw_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "finite_sample_fraction": len(finite) / len(samples),
                }
            )
            raw_cache.append((case, model, samples))
    raw_frame = pd.DataFrame(raw_rows)
    calibrators = _fit_calibrators(raw_frame.dropna(subset=["raw_median"]))
    levels = tuple(map(float, config["evaluation"]["interval_levels"]))
    calibrated_rows = [
        _metric_row(
            case,
            model,
            _apply_calibrator(samples, calibrators[f"{case['task']}|{model}"]),
            levels,
        )
        for case, model, samples in raw_cache
        if f"{case['task']}|{model}" in calibrators
    ]
    calibration_predictions = pd.DataFrame(calibrated_rows)
    calibration_summary = (
        calibration_predictions.groupby(["task", "model"], as_index=False)
        .agg(
            n=("wis", "count"),
            participants=("participant_id", "nunique"),
            mae=("absolute_error", "mean"),
            interval_mae=("interval_absolute_error", "mean"),
            crps=("crps", "mean"),
            wis=("wis", "mean"),
            coverage_90=("coverage_90", "mean"),
            width_90=("width_90", "mean"),
        )
    )

    parameter_path = output_dir / "population_parameters.json"
    calibrator_path = output_dir / "calibrators.json"
    raw_path = output_dir / "calibration_raw.parquet"
    prediction_path = output_dir / "calibration_predictions.parquet"
    summary_path = output_dir / "calibration_summary.csv"
    parameter_path.write_text(json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8")
    calibrator_path.write_text(json.dumps(calibrators, indent=2, sort_keys=True), encoding="utf-8")
    raw_frame.to_parquet(raw_path, index=False)
    calibration_predictions.to_parquet(prediction_path, index=False)
    calibration_summary.to_csv(summary_path, index=False)

    freeze = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_locked_test",
        "config_sha256": _sha256(config_path),
        "protocol_manifest_sha256": _sha256(
            Path(protocol_dir) / "protocol_manifest.json"
        ),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "population_parameters_sha256": _sha256(parameter_path),
        "calibrators_sha256": _sha256(calibrator_path),
        "code_sha256": _code_hashes(Path(project_root)),
        "development_participant_counts": {
            "population_train": len(training_ids),
            "calibration": len(calibration_ids),
        },
        "locked_test_outcomes_accessed": False,
        "protocol_outcome_blind": protocol_manifest["outcome_blind"],
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    return freeze_path.resolve()


def _verify_model_freeze(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    development_dir: Path,
) -> dict[str, Any]:
    freeze_path = Path(development_dir) / "model_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checks = {
        "config_sha256": _sha256(config_path),
        "protocol_manifest_sha256": _sha256(
            Path(protocol_dir) / "protocol_manifest.json"
        ),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "population_parameters_sha256": _sha256(
            Path(development_dir) / "population_parameters.json"
        ),
        "calibrators_sha256": _sha256(Path(development_dir) / "calibrators.json"),
    }
    for key, value in checks.items():
        if freeze.get(key) != value:
            raise ValueError(f"Model freeze verification failed: {key}")
    if freeze.get("code_sha256") != _code_hashes(Path(project_root)):
        raise ValueError("Model freeze verification failed: code changed")
    return freeze


def _bootstrap_comparisons(
    predictions: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 501001)
    for task, task_frame in predictions.groupby("task"):
        pivot = task_frame.pivot_table(
            index=["participant_id", "cycle_id"],
            columns="model",
            values="wis",
        )
        if "digital_twin_all" not in pivot:
            continue
        for comparator in sorted(set(pivot.columns) - {"digital_twin_all"}):
            paired = pivot[["digital_twin_all", comparator]].dropna()
            if paired.empty:
                continue
            participant_values = (
                (paired["digital_twin_all"] - paired[comparator])
                .groupby(level="participant_id")
                .mean()
            )
            values = participant_values.to_numpy(dtype=float)
            bootstrap = np.array(
                [
                    rng.choice(values, size=len(values), replace=True).mean()
                    for _ in range(replicates)
                ]
            )
            rows.append(
                {
                    "task": task,
                    "comparison": f"digital_twin_all-minus-{comparator}",
                    "comparator": comparator,
                    "mean_wis_difference": float(values.mean()),
                    "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
                    "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
                    "participants": len(values),
                    "digital_twin_better": bool(values.mean() < 0),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_report(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> Path:
    lines = [
        "# Locked mcPHASES real-data benchmark",
        "",
        "> Aggregate results from restricted data. Human disclosure review is required before release.",
        "",
        "Negative WIS differences mean the digital twin performed better than the comparator.",
        "",
        "## Model summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Participant-bootstrap comparisons",
        "",
        comparisons.to_markdown(index=False) if not comparisons.empty else "No paired comparisons were available.",
        "",
        "## Interpretation constraints",
        "",
        "- The transition target is a proprietary Mira/device-defined interval, not clinically confirmed ovulation.",
        "- WIS for the transition task uses the interval midpoint as a secondary scoring convention; interval distance and posterior mass are also reported.",
        "- The 2022 interval lacks PdG and the 2024 interval largely lacks diary data, so modality effects are interval-confounded.",
        "- This is a small pilot and not clinical or external validation.",
        "",
    ]
    report = output_dir / "LOCKED_BENCHMARK_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def run_mcphases_locked_test(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    development_dir: Path,
    restricted_output_dir: Path,
    aggregate_output_dir: Path,
) -> Path:
    restricted_output_dir = Path(restricted_output_dir)
    aggregate_output_dir = Path(aggregate_output_dir)
    restricted_output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = restricted_output_dir / "locked_run_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Locked mcPHASES evaluation has already been started")
    _verify_model_freeze(
        project_root, curated_dir, config_path, protocol_dir, development_dir
    )
    receipt_path.write_text(
        json.dumps(
            {
                "started_utc": datetime.now(UTC).isoformat(),
                "status": "started",
                "one_shot": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tables, _ = _load_verified_tables(Path(curated_dir))
    splits, _ = _verify_protocol(curated_dir, config_path, protocol_dir)
    config = _load_protocol(config_path)
    test_ids = set(
        splits.loc[splits["split"].eq("locked_test"), "participant_id"].astype(str)
    )
    cycles = tables["cycles"].loc[
        tables["cycles"]["participant_id"].isin(test_ids)
    ].copy()
    references = tables["reference_intervals"].loc[
        tables["reference_intervals"]["participant_id"].isin(test_ids)
    ].copy()
    observations = tables["daily_observations"].loc[
        tables["daily_observations"]["participant_id"].isin(test_ids)
    ].copy()
    transitions = _transition_targets(references, cycles)
    parameters = json.loads(
        (Path(development_dir) / "population_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    calibrators = json.loads(
        (Path(development_dir) / "calibrators.json").read_text(encoding="utf-8")
    )
    cases = (
        _cases(
            "next_menses", test_ids, cycles, transitions, observations, config
        )
        + _cases(
            "device_transition", test_ids, cycles, transitions, observations, config
        )
    )
    levels = tuple(map(float, config["evaluation"]["interval_levels"]))
    rows: list[dict[str, Any]] = []
    for case in cases:
        for model, raw_samples in _raw_forecasts(
            case, parameters, config, Path(project_root)
        ).items():
            key = f"{case['task']}|{model}"
            if key not in calibrators:
                continue
            rows.append(
                _metric_row(
                    case,
                    model,
                    _apply_calibrator(raw_samples, calibrators[key]),
                    levels,
                )
            )
    predictions = pd.DataFrame(rows)
    prediction_path = restricted_output_dir / "locked_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    summary = (
        predictions.groupby(["task", "study_interval", "model"], as_index=False)
        .agg(
            n=("wis", "count"),
            participants=("participant_id", "nunique"),
            mae=("absolute_error", "mean"),
            interval_mae=("interval_absolute_error", "mean"),
            crps=("crps", "mean"),
            wis=("wis", "mean"),
            coverage_90=("coverage_90", "mean"),
            width_90=("width_90", "mean"),
            reference_interval_mass=("posterior_mass_in_reference_interval", "mean"),
        )
    )
    comparisons = _bootstrap_comparisons(
        predictions,
        int(config["evaluation"]["bootstrap_replicates"]),
        int(config["seed"]),
    )
    summary.to_csv(aggregate_output_dir / "locked_summary.csv", index=False)
    comparisons.to_csv(
        aggregate_output_dir / "participant_bootstrap_comparisons.csv", index=False
    )
    report = _aggregate_report(summary, comparisons, aggregate_output_dir)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "predictions_sha256": _sha256(prediction_path),
            "forecast_rows": len(predictions),
            "test_participants": len(test_ids),
            "aggregate_report": str(report.resolve()),
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return report.resolve()
