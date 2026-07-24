from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProcessConfig:
    dt_days: float = 1.0
    mean_cycle_days: float = 29.0
    log_omega_reversion: float = 0.06
    log_omega_sd: float = 0.012
    phase_diffusion_sd: float = 0.035
    amplitude_recovery: float = 0.22
    amplitude_sd: float = 0.08
    amplitude_to_phase: float = 0.02
    sleep_effect_on_phase: float = -0.004
    stress_effect_on_amplitude: float = 0.08
    illness_effect_on_amplitude: float = 0.25
    ovulation_phase_fraction: float = 0.56


@dataclass(frozen=True)
class ObservationConfig:
    temperature_baseline_c: float = 36.45
    temperature_luteal_shift_c: float = 0.22
    temperature_sd_c: float = 0.08
    rhr_baseline_bpm: float = 62.0
    rhr_luteal_shift_bpm: float = 2.3
    rhr_sd_bpm: float = 1.5
    lh_baseline: float = 3.0
    lh_surge_amplitude: float = 26.0
    lh_log_sd: float = 0.22
    e3g_baseline: float = 45.0
    e3g_peak_amplitude: float = 155.0
    e3g_log_sd: float = 0.18
    pdg_baseline: float = 1.5
    pdg_luteal_amplitude: float = 10.0
    pdg_log_sd: float = 0.20
    bleeding_logit_peak: float = 5.0
    bleeding_logit_floor: float = -6.0
    missing_probability: float = 0.08


@dataclass(frozen=True)
class FilterConfig:
    particles: int = 1500
    resample_ess_fraction: float = 0.50
    prior_cycle_days_sd: float = 3.5
    forecast_horizon_days: int = 60
    forecast_replicates_per_particle: int = 1


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 20260721
    process: ProcessConfig = field(default_factory=ProcessConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


def _construct(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> ModelConfig:
    """Load a YAML configuration with strict top-level sections."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    allowed = {"seed", "process", "observation", "filter"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"Unexpected config keys: {sorted(unexpected)}")
    return ModelConfig(
        seed=int(raw.get("seed", 20260721)),
        process=_construct(ProcessConfig, raw.get("process")),
        observation=_construct(ObservationConfig, raw.get("observation")),
        filter=_construct(FilterConfig, raw.get("filter")),
    )
