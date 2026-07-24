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
from digital_twin.evaluation.metrics import event_metrics
from digital_twin.inference.forecast import forecast_events
from digital_twin.inference.particle_filter import run_particle_filter
from digital_twin.inference.personalization import ParticipantDurationPosterior
from digital_twin.simulation.observation_process import expected_signal


REQUIRED_TABLES = {
    "participants",
    "daily_observations",
    "hormone_measurements",
    "events",
    "reference_intervals",
    "cycles",
    "data_quality",
}
CANONICAL_SIGNAL_MAP = {
    "salivary_estradiol": "estradiol",
    "salivary_progesterone": "progesterone",
    "bleeding_reported": "bleeding",
    "night_heart_rate_median": "resting_heart_rate",
}
CODE_PATHS = (
    "src/digital_twin/real_data/open_benchmark.py",
    "src/digital_twin/data_adapters/salzburg_hormones.py",
    "src/digital_twin/data_adapters/soochow_heart_rate.py",
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


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "source_dataset",
        "seed",
        "split",
        "tasks",
        "evaluation",
        "digital_twin",
        "feature_model",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Open-data protocol is missing keys: {missing}")
    return config


def _load_verified_tables(
    curated_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    curated_dir = Path(curated_dir)
    manifest_path = curated_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("access_classification") != "public_deidentified_participant_data":
        raise ValueError("Expected a public deidentified curated manifest")
    metadata = manifest.get("tables", {})
    missing = sorted(REQUIRED_TABLES - set(metadata))
    if missing:
        raise ValueError(f"Curated manifest is missing tables: {missing}")
    tables: dict[str, pd.DataFrame] = {}
    for name in sorted(REQUIRED_TABLES):
        details = metadata[name]
        path = curated_dir / str(details["path"])
        if _sha256(path) != details["sha256"]:
            raise ValueError(f"Curated checksum verification failed: {path.name}")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
        else:
            raise ValueError(f"Unsupported curated table format: {path}")
        if len(frame) != int(details["rows"]):
            raise ValueError(f"Curated row-count verification failed: {path.name}")
        tables[name] = frame
    return tables, manifest


def _allocate(
    ids: list[str],
    train_fraction: float,
    calibration_fraction: float,
    rng: np.random.Generator,
) -> dict[str, str]:
    values = np.asarray(sorted(ids), dtype=object)
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


def freeze_open_protocol(
    curated_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "participant_splits.csv"
    manifest_path = output_dir / "protocol_manifest.json"
    if split_path.exists() or manifest_path.exists():
        raise FileExistsError("The open-data participant protocol is already frozen")
    tables, curated_manifest = _load_verified_tables(curated_dir)
    config = _load_config(config_path)
    participants = tables["participants"].copy()
    quality = tables["data_quality"].copy()
    lh = quality[["participant_id", "urinary_lh_positive_events"]].copy()
    lh["has_lh_reference"] = lh["urinary_lh_positive_events"].gt(0)
    strata = participants[["participant_id", "device_type"]].merge(
        lh[["participant_id", "has_lh_reference"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    if "soochow" in str(config["source_dataset"]).lower():
        strata["stratum"] = strata["device_type"].fillna("unknown").astype(str)
        variables_used = ["participant_id", "device_type"]
    else:
        strata["stratum"] = np.where(
            strata["has_lh_reference"].fillna(False),
            "has_lh_reference",
            "no_lh_reference",
        )
        variables_used = ["participant_id", "has_lh_reference"]

    assignments: dict[str, str] = {}
    split_cfg = config["split"]
    for index, (_, group) in enumerate(strata.groupby("stratum", sort=True)):
        rng = np.random.default_rng(int(config["seed"]) + 7919 * (index + 1))
        assignments.update(
            _allocate(
                group["participant_id"].astype(str).tolist(),
                float(split_cfg["train_fraction"]),
                float(split_cfg["calibration_fraction"]),
                rng,
            )
        )
    splits = strata.copy()
    splits["split"] = splits["participant_id"].map(assignments)
    splits["split_seed"] = int(config["seed"])
    splits["stratification_variables"] = ",".join(variables_used[1:])
    splits.to_csv(split_path, index=False)
    counts = (
        splits.groupby(["stratum", "split"]).size().rename("participants").reset_index()
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_dataset": config["source_dataset"],
        "procedurally_locked_public_dataset": True,
        "outcome_timing_blind": True,
        "variables_used_for_split": variables_used,
        "variables_not_used": [
            "cycle_length",
            "transition_day",
            "hormone_values",
            "heart_rate_values",
            "forecast_error",
        ],
        "config_sha256": _sha256(config_path),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "split_sha256": _sha256(split_path),
        "aggregate_split_counts": counts.to_dict(orient="records"),
        "locked_test_policy": (
            "No locked-test forecast is generated before model_freeze.json exists."
        ),
        "curated_manifest_created_utc": curated_manifest.get("created_utc"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path.resolve()


def _verify_protocol(
    curated_dir: Path, config_path: Path, protocol_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    protocol_dir = Path(protocol_dir)
    manifest = json.loads(
        (protocol_dir / "protocol_manifest.json").read_text(encoding="utf-8")
    )
    split_path = protocol_dir / "participant_splits.csv"
    checks = {
        "config_sha256": _sha256(config_path),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "split_sha256": _sha256(split_path),
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise ValueError(f"Frozen protocol verification failed: {key}")
    return pd.read_csv(split_path), manifest


def _transition_targets(
    references: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    transition = references.loc[
        references["event_type"].eq("urinary_lh_positive")
        & ~references["cycle_id"].astype(str).str.endswith("-PRE")
    ].copy()
    keys = ["participant_id", "study_interval", "cycle_id"]
    counts = transition.groupby(keys).size()
    valid = counts[counts.eq(1)].index
    if len(valid) == 0:
        return pd.DataFrame()
    transition = transition.set_index(keys).loc[valid].reset_index()
    starts = cycles[
        keys
        + [
            "cycle_start_day",
            "cycle_length_days",
            "eligible_for_primary_evaluation",
        ]
    ]
    transition = transition.merge(starts, on=keys, how="inner", validate="one_to_one")
    transition["target_midpoint_day"] = (
        pd.to_numeric(transition["day_in_study"], errors="coerce")
        - pd.to_numeric(transition["cycle_start_day"], errors="coerce")
        + 1
    )
    transition["target_lower_day"] = transition["target_midpoint_day"]
    transition["target_upper_day"] = transition["target_midpoint_day"]
    return transition


def _population_parameters(
    cycles: pd.DataFrame,
    transitions: pd.DataFrame,
    training_ids: set[str],
) -> dict[str, float]:
    train_cycles = cycles.loc[
        cycles["participant_id"].isin(training_ids)
        & cycles["eligible_for_primary_evaluation"].astype(bool)
    ]
    duration = pd.to_numeric(
        train_cycles["cycle_length_days"], errors="coerce"
    ).dropna().to_numpy()
    if len(duration) < 2:
        raise RuntimeError("Insufficient population-training cycles")
    train_transition = transitions.loc[
        transitions["participant_id"].isin(training_ids)
    ]
    follicular = pd.to_numeric(
        train_transition["target_midpoint_day"], errors="coerce"
    ).dropna().to_numpy()
    matched = train_transition.dropna(
        subset=["cycle_length_days", "target_midpoint_day"]
    )
    luteal = (
        pd.to_numeric(matched["cycle_length_days"], errors="coerce")
        - pd.to_numeric(matched["target_midpoint_day"], errors="coerce")
    ).to_numpy()
    luteal = luteal[np.isfinite(luteal) & (luteal > 2)]
    return {
        "cycle_mean": float(np.mean(duration)),
        "cycle_sd": float(max(np.std(duration, ddof=1), 1.0)),
        "transition_mean": float(np.mean(follicular)) if len(follicular) else 14.0,
        "transition_sd": (
            float(max(np.std(follicular, ddof=1), 1.0))
            if len(follicular) > 1
            else 3.0
        ),
        "follicular_mean": float(np.mean(follicular)) if len(follicular) else 14.0,
        "follicular_sd": (
            float(max(np.std(follicular, ddof=1), 1.0))
            if len(follicular) > 1
            else 3.0
        ),
        "luteal_mean": float(np.mean(luteal)) if len(luteal) else 13.0,
        "luteal_sd": (
            float(max(np.std(luteal, ddof=1), 1.0)) if len(luteal) > 1 else 2.0
        ),
        "training_cycle_intervals": int(len(duration)),
        "training_transition_intervals": int(len(follicular)),
    }


def _history(
    table: pd.DataFrame,
    participant_id: str,
    start_day: float,
    value_column: str,
) -> np.ndarray:
    values = table.loc[
        table["participant_id"].eq(participant_id)
        & pd.to_numeric(table["cycle_start_day"], errors="coerce").lt(start_day),
        ["cycle_start_day", value_column],
    ].sort_values("cycle_start_day")[value_column]
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy()


def _cases(
    task: str,
    participant_ids: set[str],
    cycles: pd.DataFrame,
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    task_cfg = config["tasks"][task]
    issue_day = int(task_cfg["issue_day"])
    if task == "next_menses":
        eligible = cycles.loc[
            cycles["participant_id"].isin(participant_ids)
            & cycles["eligible_for_primary_evaluation"].astype(bool)
            & pd.to_numeric(cycles["cycle_length_days"], errors="coerce").gt(issue_day)
        ].sort_values(["participant_id", "cycle_start_day"])
        for _, row in eligible.iterrows():
            history = _history(
                cycles.loc[cycles["eligible_for_primary_evaluation"].astype(bool)],
                str(row["participant_id"]),
                float(row["cycle_start_day"]),
                "cycle_length_days",
            )
            if len(history) < int(task_cfg["minimum_history_cycles"]):
                continue
            target = float(row["cycle_length_days"])
            cases.append(
                {
                    "task": task,
                    "participant_id": str(row["participant_id"]),
                    "study_interval": int(row["study_interval"]),
                    "cycle_id": str(row["cycle_id"]),
                    "cycle_start_day": int(row["cycle_start_day"]),
                    "issue_day": issue_day,
                    "history": history,
                    "target": target,
                    "expected_cycle_length_history": float(
                        row.get("expected_cycle_length_history", np.nan)
                    ),
                }
            )
    elif task == "lh_transition":
        eligible = transitions.loc[
            transitions["participant_id"].isin(participant_ids)
            & pd.to_numeric(
                transitions["target_midpoint_day"], errors="coerce"
            ).gt(issue_day)
        ].sort_values(["participant_id", "cycle_start_day"])
        for _, row in eligible.iterrows():
            history = _history(
                transitions,
                str(row["participant_id"]),
                float(row["cycle_start_day"]),
                "target_midpoint_day",
            )
            cases.append(
                {
                    "task": task,
                    "participant_id": str(row["participant_id"]),
                    "study_interval": int(row["study_interval"]),
                    "cycle_id": str(row["cycle_id"]),
                    "cycle_start_day": int(row["cycle_start_day"]),
                    "issue_day": issue_day,
                    "history": history,
                    "target": float(row["target_midpoint_day"]),
                    "expected_cycle_length_history": np.nan,
                }
            )
    else:
        raise ValueError(task)
    for case in cases:
        upper_day = case["cycle_start_day"] + case["issue_day"] - 1
        case["observations"] = observations.loc[
            observations["participant_id"].eq(case["participant_id"])
            & observations["cycle_id"].eq(case["cycle_id"])
            & pd.to_numeric(observations["day_in_study"], errors="coerce").between(
                case["cycle_start_day"], upper_day
            )
        ].copy()
    return cases


def _seed(config_seed: int, case: dict[str, Any], model: str) -> int:
    key = (
        f"{config_seed}:{case['task']}:{case['participant_id']}:"
        f"{case['cycle_id']}:{model}"
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _fit_signal_profiles(
    observations: pd.DataFrame, training_ids: set[str]
) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    training = observations.loc[observations["participant_id"].isin(training_ids)]
    phase_progress = np.linspace(0.0, 1.0, 200)
    stage = np.r_[np.zeros(200, dtype=int), np.ones(200, dtype=int)]
    progress_grid = np.r_[phase_progress, phase_progress]
    for source_signal, canonical in CANONICAL_SIGNAL_MAP.items():
        if canonical not in {"estradiol", "progesterone"}:
            continue
        raw = pd.to_numeric(
            training.loc[
                training["signal_name"].eq(source_signal), "value"
            ],
            errors="coerce",
        ).dropna().to_numpy()
        positive = raw[raw > 0]
        if not len(positive):
            continue
        floor = max(float(np.min(positive)) / 2.0, 1e-4)
        raw_log = np.log(np.maximum(raw, floor))
        target = expected_signal(canonical, stage, progress_grid)
        target_log = np.log(np.maximum(target, 1e-6))
        profiles[source_signal] = {
            "canonical_signal": canonical,
            "raw_floor": floor,
            "raw_log_mean": float(np.mean(raw_log)),
            "raw_log_sd": float(max(np.std(raw_log, ddof=1), 0.1)),
            "target_log_mean": float(np.mean(target_log)),
            "target_log_sd": float(max(np.std(target_log, ddof=1), 0.1)),
            "training_values": int(len(raw)),
        }
    return profiles


def _model_observations(
    case: dict[str, Any],
    canonical_signals: set[str],
    profiles: dict[str, dict[str, float]],
) -> pd.DataFrame:
    frame = case["observations"].loc[
        case["observations"]["signal_name"].isin(CANONICAL_SIGNAL_MAP)
    ].copy()
    frame["source_signal_name"] = frame["signal_name"]
    frame["signal_name"] = frame["signal_name"].map(CANONICAL_SIGNAL_MAP)
    frame = frame.loc[frame["signal_name"].isin(canonical_signals)].copy()
    frame["event_time"] = pd.to_datetime(
        frame["event_time"], format="mixed"
    ).dt.normalize()
    for source_signal, profile in profiles.items():
        mask = frame["source_signal_name"].eq(source_signal) & frame["value"].notna()
        if not mask.any():
            continue
        raw = pd.to_numeric(frame.loc[mask, "value"], errors="coerce").to_numpy()
        z = (
            np.log(np.maximum(raw, float(profile["raw_floor"])))
            - float(profile["raw_log_mean"])
        ) / float(profile["raw_log_sd"])
        frame.loc[mask, "value"] = np.exp(
            float(profile["target_log_mean"])
            + float(profile["target_log_sd"]) * z
        )
    frame["is_observed"] = frame["value"].notna()
    return frame


def _feature_names(signals: list[str]) -> list[str]:
    names = ["history_count", "history_mean", "history_last", "history_sd"]
    for signal in signals:
        names.extend(
            [
                f"{signal}__coverage",
                f"{signal}__mean",
                f"{signal}__last",
                f"{signal}__slope",
            ]
        )
    return names


def _feature_vector(case: dict[str, Any], signals: list[str]) -> np.ndarray:
    history = np.asarray(case["history"], dtype=float)
    values: list[float] = [
        float(len(history)),
        float(np.mean(history)) if len(history) else np.nan,
        float(history[-1]) if len(history) else np.nan,
        float(np.std(history, ddof=1)) if len(history) > 1 else np.nan,
    ]
    observations = case["observations"]
    cycle_day = (
        pd.to_numeric(observations["day_in_study"], errors="coerce")
        - case["cycle_start_day"]
        + 1
    )
    for signal in signals:
        mask = observations["signal_name"].eq(signal)
        y = pd.to_numeric(observations.loc[mask, "value"], errors="coerce")
        x = cycle_day.loc[mask]
        valid = y.notna() & x.notna()
        observed_y = y.loc[valid].to_numpy()
        observed_x = x.loc[valid].to_numpy()
        slope = (
            float(np.polyfit(observed_x, observed_y, 1)[0])
            if len(observed_y) >= 2 and np.ptp(observed_x) > 0
            else np.nan
        )
        values.extend(
            [
                float(len(observed_y) / max(case["issue_day"], 1)),
                float(np.mean(observed_y)) if len(observed_y) else np.nan,
                float(observed_y[-1]) if len(observed_y) else np.nan,
                slope,
            ]
        )
    return np.asarray(values, dtype=float)


def _fit_ridge(
    cases: list[dict[str, Any]], signals: list[str], penalty: float
) -> dict[str, Any]:
    names = _feature_names(signals)
    if len(cases) < 3:
        return {"available": False, "reason": "fewer_than_three_training_cases"}
    matrix = np.vstack([_feature_vector(case, signals) for case in cases])
    target = np.asarray([case["target"] for case in cases], dtype=float)
    medians = np.asarray(
        [
            float(np.median(column[np.isfinite(column)]))
            if np.isfinite(column).any()
            else 0.0
            for column in matrix.T
        ]
    )
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0, ddof=1)
    scales[~np.isfinite(scales) | (scales < 1e-8)] = 1.0
    standardized = (matrix - means) / scales
    design = np.c_[np.ones(len(standardized)), standardized]
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficient = np.linalg.solve(
        design.T @ design + regularizer, design.T @ target
    )
    fitted = design @ coefficient
    residual_sd = (
        float(max(np.std(target - fitted, ddof=1), 1.0))
        if len(target) > 1
        else 3.0
    )
    return {
        "available": True,
        "signals": signals,
        "feature_names": names,
        "medians": medians.tolist(),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficient.tolist(),
        "residual_sd": residual_sd,
        "training_cases": len(cases),
        "penalty": float(penalty),
    }


def _ridge_predict(case: dict[str, Any], model: dict[str, Any]) -> float:
    vector = _feature_vector(case, list(model["signals"]))
    medians = np.asarray(model["medians"], dtype=float)
    vector = np.where(np.isfinite(vector), vector, medians)
    standardized = (
        vector - np.asarray(model["means"], dtype=float)
    ) / np.asarray(model["scales"], dtype=float)
    design = np.r_[1.0, standardized]
    return float(design @ np.asarray(model["coefficients"], dtype=float))


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


def _hsmm_daily(case: dict[str, Any]) -> pd.DataFrame:
    frame = case["observations"].loc[
        case["observations"]["signal_name"].eq("bleeding_reported")
    ].copy()
    frame["cycle_day"] = (
        pd.to_numeric(frame["day_in_study"], errors="coerce")
        - case["cycle_start_day"]
        + 1
    )
    daily = (
        frame.pivot_table(
            index="cycle_day", values="value", aggfunc="last"
        )
        .rename(columns={"value": "bleeding"})
        .reindex(range(1, case["issue_day"] + 1))
    )
    return daily.reset_index(drop=True)


def _raw_forecasts(
    case: dict[str, Any],
    parameters: dict[str, float],
    ridge_models: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, float]],
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, np.ndarray]:
    n = int(config["evaluation"]["forecast_samples"])
    history = np.asarray(case["history"], dtype=float)
    posterior = ParticipantDurationPosterior.from_history(
        history, parameters["cycle_mean"], parameters["cycle_sd"]
    )
    models: dict[str, np.ndarray] = {}
    if case["task"] == "next_menses":
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "population_calendar"))
        models["population_calendar"] = np.clip(
            rng.normal(parameters["cycle_mean"], parameters["cycle_sd"], n), 15, 60
        )
        rolling = float(
            np.mean(history[-int(config["tasks"]["next_menses"]["rolling_window"]) :])
        )
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "rolling_up_to_k3"))
        models["rolling_up_to_k3"] = np.clip(
            rng.normal(rolling, parameters["cycle_sd"], n), 15, 60
        )
        renewal = HierarchicalRenewal(
            population_mean=parameters["cycle_mean"],
            population_sd=parameters["cycle_sd"],
        )
        rng = np.random.default_rng(
            _seed(int(config["seed"]), case, "hierarchical_renewal")
        )
        models["hierarchical_renewal"] = renewal.predictive_samples(
            0.0, history, n, rng
        )
        expected = float(case["expected_cycle_length_history"])
        if np.isfinite(expected):
            rng = np.random.default_rng(
                _seed(int(config["seed"]), case, "reported_history_calendar")
            )
            models["reported_history_calendar"] = np.clip(
                rng.normal(expected, parameters["cycle_sd"], n), 15, 60
            )
    else:
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "population_phase_prior"))
        models["population_phase_prior"] = np.clip(
            rng.normal(parameters["transition_mean"], parameters["transition_sd"], n),
            case["issue_day"],
            50,
        )
        center = (
            float(
                np.mean(
                    history[
                        -int(config["tasks"]["lh_transition"]["rolling_window"]) :
                    ]
                )
            )
            if len(history)
            else parameters["transition_mean"]
        )
        rng = np.random.default_rng(
            _seed(int(config["seed"]), case, "rolling_transition_up_to_k3")
        )
        models["rolling_transition_up_to_k3"] = np.clip(
            rng.normal(center, parameters["transition_sd"], n),
            case["issue_day"],
            50,
        )

    ridge = ridge_models.get(case["task"], {})
    if ridge.get("available"):
        center = _ridge_predict(case, ridge)
        rng = np.random.default_rng(_seed(int(config["seed"]), case, "ridge_past_features"))
        models["ridge_past_features"] = np.clip(
            rng.normal(center, float(ridge["residual_sd"]), n),
            case["issue_day"] if case["task"] == "lh_transition" else 15,
            60,
        )

    hsmm = HSMMBaseline(
        follicular_mean=posterior.follicular_prior_days(parameters["luteal_mean"]),
        follicular_sd=max(parameters["follicular_sd"], 1.5),
        luteal_mean=parameters["luteal_mean"],
        luteal_sd=parameters["luteal_sd"],
    )
    rng = np.random.default_rng(_seed(int(config["seed"]), case, "hsmm"))
    if case["task"] == "next_menses":
        models["hsmm"] = hsmm.predictive_samples(_hsmm_daily(case), n, rng, horizon=60)
    else:
        models["hsmm"] = hsmm.predictive_transition_samples(
            _hsmm_daily(case), n, rng, horizon=45
        )

    twin_cfg = _runtime_twin_config(project_root, config, parameters)
    for modality, signals in config["digital_twin"]["modalities"].items():
        model_name = f"digital_twin_{modality}"
        model_frame = _model_observations(case, set(signals), profiles)
        if model_frame.loc[model_frame["value"].notna()].empty:
            models[model_name] = np.full(n, np.nan)
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
            seed_offset=(
                _seed(int(config["seed"]), case, model_name + "_forecast") % 1_000_000
            ),
        )
        models[model_name] = (
            forecast.next_menses_samples
            if case["task"] == "next_menses"
            else forecast.ovulation_samples
        )
    return models


def _fit_calibrators(records: pd.DataFrame) -> dict[str, dict[str, float]]:
    calibrators: dict[str, dict[str, float]] = {}
    for (task, model), group in records.groupby(["task", "model"]):
        group = group.dropna(subset=["raw_median"])
        if group.empty:
            continue
        residual = group["target"] - group["raw_median"]
        raw_sd = group["raw_sd"].replace(0, np.nan)
        residual_sd = float(residual.std(ddof=1)) if len(group) > 1 else np.nan
        typical_raw_sd = float(raw_sd.median()) if raw_sd.notna().any() else np.nan
        scale = (
            1.0
            if not np.isfinite(residual_sd) or not np.isfinite(typical_raw_sd)
            else float(
                np.clip(residual_sd / max(typical_raw_sd, 0.25), 0.5, 4.0)
            )
        )
        calibrators[f"{task}|{model}"] = {
            "bias_days": float(residual.mean()),
            "spread_scale": scale,
            "calibration_forecasts": int(len(group)),
            "calibration_participants": int(group["participant_id"].nunique()),
        }
    return calibrators


def _apply_calibrator(
    samples: np.ndarray, calibrator: dict[str, float]
) -> np.ndarray:
    values = np.asarray(samples, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        return values
    median = float(np.median(values[finite]))
    values[finite] = (
        median
        + float(calibrator["bias_days"])
        + float(calibrator["spread_scale"]) * (values[finite] - median)
    )
    return values


def _metric_row(
    case: dict[str, Any],
    model: str,
    samples: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "participant_id": case["participant_id"],
        "cycle_id": case["cycle_id"],
        "task": case["task"],
        "model": model,
        "issue_day": case["issue_day"],
        "history_cycles": len(case["history"]),
        "target": case["target"],
        "finite_sample_fraction": float(np.isfinite(samples).mean()),
        **event_metrics(samples, case["target"], levels),
    }


def _code_hashes(project_root: Path) -> dict[str, str]:
    return {
        path: _sha256(Path(project_root) / path)
        for path in CODE_PATHS
    }


def run_open_development(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "model_freeze.json"
    if freeze_path.exists():
        raise FileExistsError("The open-data development model is already frozen")
    tables, _ = _load_verified_tables(curated_dir)
    splits, protocol_manifest = _verify_protocol(
        curated_dir, config_path, protocol_dir
    )
    config = _load_config(config_path)
    split_map = splits.set_index("participant_id")["split"]
    training_ids = set(split_map[split_map.eq("population_train")].index.astype(str))
    calibration_ids = set(split_map[split_map.eq("calibration")].index.astype(str))
    allowed = training_ids | calibration_ids
    cycles = tables["cycles"].loc[tables["cycles"]["participant_id"].isin(allowed)]
    references = tables["reference_intervals"].loc[
        tables["reference_intervals"]["participant_id"].isin(allowed)
    ]
    observations = tables["daily_observations"].loc[
        tables["daily_observations"]["participant_id"].isin(allowed)
    ]
    transitions = _transition_targets(references, cycles)
    parameters = _population_parameters(cycles, transitions, training_ids)
    training_cases = {
        task: _cases(
            task, training_ids, cycles, transitions, observations, config
        )
        for task in ("next_menses", "lh_transition")
    }
    feature_signals = list(config["feature_model"]["signals"])
    ridge_models = {
        task: _fit_ridge(
            cases,
            feature_signals,
            float(config["feature_model"]["ridge_penalty"]),
        )
        for task, cases in training_cases.items()
    }
    profiles = _fit_signal_profiles(observations, training_ids)
    calibration_cases = [
        case
        for task in ("next_menses", "lh_transition")
        for case in _cases(
            task, calibration_ids, cycles, transitions, observations, config
        )
    ]
    raw_rows: list[dict[str, Any]] = []
    raw_cache: list[tuple[dict[str, Any], str, np.ndarray]] = []
    for case in calibration_cases:
        forecasts = _raw_forecasts(
            case,
            parameters,
            ridge_models,
            profiles,
            config,
            Path(project_root),
        )
        for model, samples in forecasts.items():
            finite = np.asarray(samples, dtype=float)
            finite = finite[np.isfinite(finite)]
            raw_rows.append(
                {
                    "participant_id": case["participant_id"],
                    "cycle_id": case["cycle_id"],
                    "task": case["task"],
                    "model": model,
                    "target": case["target"],
                    "raw_median": float(np.median(finite)) if len(finite) else np.nan,
                    "raw_sd": (
                        float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
                    ),
                    "finite_sample_fraction": len(finite) / len(samples),
                }
            )
            raw_cache.append((case, model, samples))
    raw_frame = pd.DataFrame(raw_rows)
    calibrators = _fit_calibrators(raw_frame)
    levels = tuple(map(float, config["evaluation"]["interval_levels"]))
    calibration_predictions = pd.DataFrame(
        [
            _metric_row(
                case,
                model,
                _apply_calibrator(samples, calibrators[f"{case['task']}|{model}"]),
                levels,
            )
            for case, model, samples in raw_cache
            if f"{case['task']}|{model}" in calibrators
        ]
    )
    summary = (
        calibration_predictions.groupby(["task", "model"], as_index=False)
        .agg(
            n=("wis", "count"),
            participants=("participant_id", "nunique"),
            mae=("absolute_error", "mean"),
            crps=("crps", "mean"),
            wis=("wis", "mean"),
            coverage_90=("coverage_90", "mean"),
            width_90=("width_90", "mean"),
        )
    )
    artifacts = {
        "population_parameters.json": parameters,
        "ridge_models.json": ridge_models,
        "signal_profiles.json": profiles,
        "calibrators.json": calibrators,
    }
    hashes: dict[str, str] = {}
    for name, content in artifacts.items():
        path = output_dir / name
        path.write_text(json.dumps(content, indent=2, sort_keys=True), encoding="utf-8")
        hashes[name] = _sha256(path)
    raw_frame.to_csv(output_dir / "calibration_raw.csv", index=False)
    calibration_predictions.to_csv(
        output_dir / "calibration_predictions.csv", index=False
    )
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    freeze = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "frozen_before_locked_test",
        "config_sha256": _sha256(config_path),
        "protocol_manifest_sha256": _sha256(
            Path(protocol_dir) / "protocol_manifest.json"
        ),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
        "artifact_sha256": hashes,
        "code_sha256": _code_hashes(Path(project_root)),
        "development_participant_counts": {
            "population_train": len(training_ids),
            "calibration": len(calibration_ids),
        },
        "training_case_counts": {
            task: len(cases) for task, cases in training_cases.items()
        },
        "calibration_case_count": len(calibration_cases),
        "locked_test_outcomes_accessed": False,
        "protocol_outcome_timing_blind": protocol_manifest["outcome_timing_blind"],
    }
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8"
    )
    return freeze_path.resolve()


def _verify_model_freeze(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    development_dir: Path,
) -> dict[str, Any]:
    freeze = json.loads(
        (Path(development_dir) / "model_freeze.json").read_text(encoding="utf-8")
    )
    checks = {
        "config_sha256": _sha256(config_path),
        "protocol_manifest_sha256": _sha256(
            Path(protocol_dir) / "protocol_manifest.json"
        ),
        "curated_manifest_sha256": _sha256(Path(curated_dir) / "manifest.json"),
    }
    for key, value in checks.items():
        if freeze.get(key) != value:
            raise ValueError(f"Model freeze verification failed: {key}")
    for name, expected in freeze["artifact_sha256"].items():
        if _sha256(Path(development_dir) / name) != expected:
            raise ValueError(f"Model freeze verification failed: {name}")
    if freeze.get("code_sha256") != _code_hashes(project_root):
        raise ValueError("Model freeze verification failed: code changed")
    return freeze


def _bootstrap_comparisons(
    predictions: pd.DataFrame, replicates: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 501001)
    for task, task_frame in predictions.groupby("task"):
        pivot = task_frame.pivot_table(
            index=["participant_id", "cycle_id"], columns="model", values="wis"
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
            values = participant_values.to_numpy()
            bootstrap = np.asarray(
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
                    "mean_wis_difference": float(np.mean(values)),
                    "ci_lower_95": float(np.quantile(bootstrap, 0.025)),
                    "ci_upper_95": float(np.quantile(bootstrap, 0.975)),
                    "participants": int(len(values)),
                    "digital_twin_better": bool(np.mean(values) < 0),
                }
            )
    return pd.DataFrame(rows)


def _report(
    source: str,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> Path:
    lines = [
        f"# Locked open-data benchmark: {source}",
        "",
        "Negative WIS differences mean `digital_twin_all` performed better.",
        "",
        "## Model summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Participant-bootstrap comparisons",
        "",
        comparisons.to_markdown(index=False)
        if not comparisons.empty
        else "No paired comparisons were available.",
        "",
        "## Interpretation constraints",
        "",
        "- Urinary-LH positivity is a transition surrogate, not confirmed biological ovulation.",
        "- Participant splits, transformations, calibration, and model code were frozen before this evaluation.",
        "- Different cohorts are evaluated separately and are never joined into artificial multimodal participants.",
        "- Results are methodological and do not establish clinical or fertility-use validity.",
        "",
    ]
    path = output_dir / "LOCKED_OPEN_BENCHMARK_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_open_locked_test(
    project_root: Path,
    curated_dir: Path,
    config_path: Path,
    protocol_dir: Path,
    development_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "locked_run_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("This locked open-data evaluation has already started")
    _verify_model_freeze(
        Path(project_root),
        curated_dir,
        config_path,
        protocol_dir,
        development_dir,
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
    tables, _ = _load_verified_tables(curated_dir)
    splits, _ = _verify_protocol(curated_dir, config_path, protocol_dir)
    config = _load_config(config_path)
    test_ids = set(
        splits.loc[splits["split"].eq("locked_test"), "participant_id"].astype(str)
    )
    cycles = tables["cycles"].loc[tables["cycles"]["participant_id"].isin(test_ids)]
    references = tables["reference_intervals"].loc[
        tables["reference_intervals"]["participant_id"].isin(test_ids)
    ]
    observations = tables["daily_observations"].loc[
        tables["daily_observations"]["participant_id"].isin(test_ids)
    ]
    transitions = _transition_targets(references, cycles)
    development_dir = Path(development_dir)
    parameters = json.loads(
        (development_dir / "population_parameters.json").read_text(encoding="utf-8")
    )
    ridge_models = json.loads(
        (development_dir / "ridge_models.json").read_text(encoding="utf-8")
    )
    profiles = json.loads(
        (development_dir / "signal_profiles.json").read_text(encoding="utf-8")
    )
    calibrators = json.loads(
        (development_dir / "calibrators.json").read_text(encoding="utf-8")
    )
    cases = [
        case
        for task in ("next_menses", "lh_transition")
        for case in _cases(
            task, test_ids, cycles, transitions, observations, config
        )
    ]
    levels = tuple(map(float, config["evaluation"]["interval_levels"]))
    rows: list[dict[str, Any]] = []
    for case in cases:
        forecasts = _raw_forecasts(
            case,
            parameters,
            ridge_models,
            profiles,
            config,
            Path(project_root),
        )
        for model, samples in forecasts.items():
            key = f"{case['task']}|{model}"
            if key not in calibrators:
                continue
            rows.append(
                _metric_row(
                    case,
                    model,
                    _apply_calibrator(samples, calibrators[key]),
                    levels,
                )
            )
    predictions = pd.DataFrame(rows)
    prediction_path = output_dir / "locked_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    summary = (
        predictions.groupby(["task", "model"], as_index=False)
        .agg(
            n=("wis", "count"),
            participants=("participant_id", "nunique"),
            mae=("absolute_error", "mean"),
            crps=("crps", "mean"),
            wis=("wis", "mean"),
            coverage_90=("coverage_90", "mean"),
            width_90=("width_90", "mean"),
            finite_sample_fraction=("finite_sample_fraction", "mean"),
        )
    )
    comparisons = _bootstrap_comparisons(
        predictions,
        int(config["evaluation"]["bootstrap_replicates"]),
        int(config["seed"]),
    )
    summary.to_csv(output_dir / "locked_summary.csv", index=False)
    comparisons.to_csv(
        output_dir / "participant_bootstrap_comparisons.csv", index=False
    )
    report = _report(str(config["source_dataset"]), summary, comparisons, output_dir)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "completed_utc": datetime.now(UTC).isoformat(),
            "status": "complete",
            "predictions_sha256": _sha256(prediction_path),
            "forecast_rows": int(len(predictions)),
            "forecast_cases": int(len(cases)),
            "test_participants": int(len(test_ids)),
            "aggregate_report": str(report.resolve()),
        }
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report.resolve()
