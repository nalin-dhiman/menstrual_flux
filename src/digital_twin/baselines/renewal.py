from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def calendar_forecast(last_onset_day: float, population_mean_days: float) -> float:
    return float(last_onset_day + population_mean_days)


def rolling_mean_forecast(last_onset_day: float, cycle_lengths: np.ndarray, k: int = 3) -> float:
    values = np.asarray(cycle_lengths, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("at least one prior cycle length is required")
    return float(last_onset_day + np.mean(values[-k:]))


@dataclass
class HierarchicalRenewal:
    population_mean: float = 28.0
    population_sd: float = 4.0
    prior_strength: float = 3.0

    def fit(self, cycle_lengths: np.ndarray) -> "HierarchicalRenewal":
        values = np.asarray(cycle_lengths, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("at least two cycles are required")
        self.population_mean = float(np.mean(values))
        self.population_sd = float(max(np.std(values, ddof=1), 1.0))
        return self

    def participant_parameters(self, history: np.ndarray) -> tuple[float, float]:
        values = np.asarray(history, dtype=float)
        values = values[np.isfinite(values)]
        n = values.size
        mean = (self.prior_strength * self.population_mean + np.sum(values)) / (self.prior_strength + n)
        within = self.population_sd**2 if n < 2 else float(np.var(values, ddof=1))
        sd = float(np.sqrt((self.prior_strength * self.population_sd**2 + max(n - 1, 0) * within) / (self.prior_strength + max(n - 1, 0))))
        return float(mean), max(sd, 1.0)

    def predictive_samples(self, last_onset_day: float, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        mean, sd = self.participant_parameters(history)
        durations = np.clip(rng.normal(mean, sd, n), 15.0, 90.0)
        return last_onset_day + durations
