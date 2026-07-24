from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticipantDurationPosterior:
    """Conjugate approximation updated only with earlier reported cycle lengths."""

    mean_cycle_days: float
    mean_variance: float
    predictive_variance: float
    observed_cycles: int

    @classmethod
    def from_history(
        cls,
        history: np.ndarray,
        population_mean: float,
        population_sd: float,
        prior_strength: float = 3.0,
    ) -> "ParticipantDurationPosterior":
        values = np.asarray(history, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        n = values.size
        observation_variance = max(population_sd**2, 1.0)
        prior_variance = observation_variance / prior_strength
        posterior_precision = 1 / prior_variance + n / observation_variance
        posterior_mean = (population_mean / prior_variance + values.sum() / observation_variance) / posterior_precision
        posterior_variance = 1 / posterior_precision
        if n >= 2:
            observation_variance = max(float(np.var(values, ddof=1)), 1.0)
        return cls(
            mean_cycle_days=float(posterior_mean),
            mean_variance=float(posterior_variance),
            predictive_variance=float(posterior_variance + observation_variance),
            observed_cycles=int(n),
        )

    def follicular_prior_days(self, population_luteal_days: float) -> float:
        return float(np.clip(self.mean_cycle_days - population_luteal_days, 6.0, 60.0))

    @property
    def predictive_sd_days(self) -> float:
        return float(np.sqrt(self.predictive_variance))


@dataclass(frozen=True)
class StageSpeedPosterior:
    """Hierarchical estimate from observation-derived prior-cycle durations."""

    mean_log_speed: float
    posterior_sd: float
    observed_cycles: int
    reliability: float

    @classmethod
    def from_durations(
        cls,
        durations: np.ndarray,
        population_log_speed: float,
        between_person_sd: float,
        observation_sd_floor: float = 0.12,
    ) -> "StageSpeedPosterior":
        values = np.asarray(durations, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if not values.size:
            return cls(float(population_log_speed), float(max(between_person_sd, 0.05)), 0, 0.0)
        observations = -np.log(values)
        median = float(np.median(observations))
        mad = float(1.4826 * np.median(np.abs(observations - median)))
        robust_scale = max(mad, observation_sd_floor)
        observations = np.clip(observations, median - 3 * robust_scale, median + 3 * robust_scale)
        prior_variance = max(float(between_person_sd), 0.05) ** 2
        mean_variance = max(float(np.var(observations, ddof=1)) if observations.size > 1 else robust_scale**2, observation_sd_floor**2) / observations.size
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / mean_variance)
        posterior_mean = posterior_variance * (population_log_speed / prior_variance + float(observations.mean()) / mean_variance)
        reliability = prior_variance / (prior_variance + mean_variance)
        return cls(float(posterior_mean), float(np.sqrt(posterior_variance)), int(values.size), float(reliability))
