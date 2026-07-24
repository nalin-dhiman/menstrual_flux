from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSection:
    name: str = "milestone_1"
    seed: int = 20260721
    output_dir: str = "outputs/experiments/milestone_1"


@dataclass(frozen=True)
class DataSection:
    participants: int = 20
    cycles_per_participant: int = 5
    start_date: str = "2026-01-01"
    timezone: str = "UTC"
    scenario: str = "mixed"


@dataclass(frozen=True)
class ProcessSection:
    dt_days: float = 1.0
    follicular_days: float = 14.0
    luteal_days: float = 14.0
    kappa_log_speed_f: float = 0.35
    kappa_log_speed_l: float = 0.45
    sigma_log_speed_f: float = 0.035
    sigma_log_speed_l: float = 0.025
    sigma_progress_f: float = 0.006
    sigma_progress_l: float = 0.004
    between_person_log_speed_sd: float = 0.12
    cycle_effect_sd_f: float = 0.08
    cycle_effect_sd_l: float = 0.04
    cycle_effect_rho_f: float = 0.45
    cycle_effect_rho_l: float = 0.25
    drift_per_cycle_sd: float = 0.01
    anovulatory_like_probability: float = 0.04
    stalled_follicular_days: int = 45
    max_cycle_days: int = 180


@dataclass(frozen=True)
class ObservationSection:
    temperature_sd: float = 0.08
    rhr_sd: float = 1.6
    log_hrv_sd: float = 0.12
    sleep_duration_sd: float = 0.7
    sleep_efficiency_sd: float = 0.035
    log_hormone_sd: float = 0.20
    heavy_tail_df: float = 5.0
    assay_detection_limit: float = 0.5
    ordinal_latent_sd: float = 0.85
    wearable_likelihood_weight: float = 0.55
    hormone_likelihood_weight: float = 0.75
    diary_likelihood_weight: float = 1.0


@dataclass(frozen=True)
class MissingnessSection:
    mechanism: str = "mixed"
    mcar_probability: float = 0.08
    mar_probability: float = 0.08
    informative_probability: float = 0.06
    hormone_schedule_days: int = 3
    block_probability: float = 0.10
    block_min_days: int = 3
    block_max_days: int = 8
    report_delay_probability: float = 0.15
    max_report_delay_days: int = 3


@dataclass(frozen=True)
class InferenceSection:
    particles: int = 500
    ess_threshold: float = 0.50
    resampling: str = "systematic"
    forecast_horizon_days: int = 60
    forecast_samples: int = 500
    forecast_model_discrepancy_sd_days: float = 1.7


@dataclass(frozen=True)
class EvaluationSection:
    issue_day: int = 10
    interval_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    rolling_windows: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    process: ProcessSection = field(default_factory=ProcessSection)
    observation: ObservationSection = field(default_factory=ObservationSection)
    missingness: MissingnessSection = field(default_factory=MissingnessSection)
    inference: InferenceSection = field(default_factory=InferenceSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type] = {
    "experiment": ExperimentSection,
    "data": DataSection,
    "process": ProcessSection,
    "observation": ObservationSection,
    "missingness": MissingnessSection,
    "inference": InferenceSection,
    "evaluation": EvaluationSection,
}


def _construct(cls: type, values: dict[str, Any] | None):
    values = dict(values or {})
    valid = set(cls.__dataclass_fields__)
    unexpected = set(values) - valid
    if unexpected:
        raise ValueError(f"Unexpected {cls.__name__} keys: {sorted(unexpected)}")
    if cls is EvaluationSection:
        for key in ("interval_levels", "rolling_windows"):
            if key in values:
                values[key] = tuple(values[key])
    return cls(**values)


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.process.dt_days <= 0 or cfg.process.dt_days > 1:
        raise ValueError("process.dt_days must be in (0, 1]")
    if cfg.process.follicular_days <= 0 or cfg.process.luteal_days <= 0:
        raise ValueError("stage durations must be positive")
    if cfg.process.stalled_follicular_days < cfg.process.follicular_days:
        raise ValueError("stalled_follicular_days must not be shorter than the regular follicular duration")
    if cfg.process.max_cycle_days <= cfg.process.stalled_follicular_days + cfg.process.luteal_days:
        raise ValueError("max_cycle_days must exceed the longest configured nominal stage duration")
    if cfg.data.participants < 1 or cfg.data.cycles_per_participant < 2:
        raise ValueError("at least one participant and two cycles are required")
    if cfg.inference.particles < 50:
        raise ValueError("at least 50 particles are required")
    if not 0 < cfg.inference.ess_threshold <= 1:
        raise ValueError("ESS threshold must be in (0, 1]")
    probabilities = [
        cfg.missingness.mcar_probability,
        cfg.missingness.mar_probability,
        cfg.missingness.informative_probability,
        cfg.missingness.block_probability,
        cfg.missingness.report_delay_probability,
        cfg.process.anovulatory_like_probability,
    ]
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("all probabilities must be in [0, 1]")
    if cfg.missingness.mechanism not in {"none", "mcar", "mar", "informative", "mixed"}:
        raise ValueError("unsupported missingness mechanism")
    if cfg.inference.resampling not in {"systematic", "stratified", "residual", "multinomial"}:
        raise ValueError("unsupported resampling method")
    if any(weight <= 0 or weight > 1 for weight in (
        cfg.observation.wearable_likelihood_weight,
        cfg.observation.hormone_likelihood_weight,
        cfg.observation.diary_likelihood_weight,
    )):
        raise ValueError("likelihood weights must be in (0, 1]")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unexpected = set(raw) - set(_SECTIONS)
    if unexpected:
        raise ValueError(f"Unexpected top-level config keys: {sorted(unexpected)}")
    kwargs = {name: _construct(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    cfg = ExperimentConfig(**kwargs)
    validate_config(cfg)
    return cfg


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
