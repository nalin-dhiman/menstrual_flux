from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class HSMMBaseline:
    """Two-state explicit-duration baseline implemented as an age-expanded HMM."""

    follicular_mean: float = 14.0
    follicular_sd: float = 3.0
    luteal_mean: float = 14.0
    luteal_sd: float = 2.0
    max_duration: int = 45

    def _hazard(self, mean: float, sd: float) -> np.ndarray:
        age = np.arange(1, self.max_duration + 1)
        survival = np.maximum(1 - norm.cdf(age - 0.5, mean, sd), 1e-10)
        mass = norm.cdf(age + 0.5, mean, sd) - norm.cdf(age - 0.5, mean, sd)
        hazard = np.clip(mass / survival, 1e-5, 1.0)
        hazard[-1] = 1.0
        return hazard

    def filter_state(self, daily: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        hf = self._hazard(self.follicular_mean, self.follicular_sd)
        hl = self._hazard(self.luteal_mean, self.luteal_sd)
        f = np.zeros(self.max_duration)
        l = np.zeros(self.max_duration)
        f[0] = 1.0
        rows: list[dict[str, float]] = []
        for day, row in daily.reset_index(drop=True).iterrows():
            nf = np.zeros_like(f)
            nl = np.zeros_like(l)
            nf[1:] = f[:-1] * (1 - hf[:-1])
            nl[0] = np.sum(f * hf)
            nl[1:] = l[:-1] * (1 - hl[:-1])
            nf[0] += np.sum(l * hl)
            f, l = nf, nl
            bleed = row.get("bleeding", np.nan)
            temperature = row.get("temperature", np.nan)
            lh = row.get("lh", np.nan)
            ef, el = 1.0, 1.0
            if np.isfinite(bleed):
                ef *= 0.70 if bleed >= 1 else 0.72
                el *= 0.03 if bleed >= 1 else 0.97
            if np.isfinite(temperature):
                ef *= float(norm.pdf(temperature, 36.45, 0.15) + 1e-9)
                el *= float(norm.pdf(temperature, 36.68, 0.15) + 1e-9)
            if np.isfinite(lh) and lh > 12:
                ef *= 2.0
                el *= 0.35
            f *= ef
            l *= el
            total = f.sum() + l.sum()
            if total <= 0:
                f.fill(0); l.fill(0); f[0] = 1.0
            else:
                f /= total; l /= total
            rows.append({"day": day + 1, "p_follicular": float(f.sum()), "p_luteal": float(l.sum())})
        return pd.DataFrame(rows), f, l

    def filter(self, daily: pd.DataFrame) -> pd.DataFrame:
        return self.filter_state(daily)[0]

    def predictive_samples(
        self,
        daily: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
        horizon: int = 70,
        discrepancy_sd_days: float = 0.0,
    ) -> np.ndarray:
        """Sample the next luteal-to-follicular passage from the filtered HSMM."""
        _, f, l = self.filter_state(daily)
        combined = np.concatenate([f, l])
        combined /= combined.sum()
        initial = rng.choice(combined.size, size=n, p=combined)
        stage = (initial >= self.max_duration).astype(np.int8)
        age = initial % self.max_duration
        hf = self._hazard(self.follicular_mean, self.follicular_sd)
        hl = self._hazard(self.luteal_mean, self.luteal_sd)
        samples = np.full(n, np.nan)
        issue_day = len(daily)
        for future in range(1, horizon + 1):
            active = np.isnan(samples)
            if not active.any():
                break
            hazard = np.where(stage == 0, hf[np.minimum(age, self.max_duration - 1)], hl[np.minimum(age, self.max_duration - 1)])
            transition = active & (rng.random(n) < hazard)
            menses = transition & (stage == 1)
            samples[menses] = issue_day + future
            stage[transition] = 1 - stage[transition]
            age[transition] = 0
            age[active & ~transition] = np.minimum(age[active & ~transition] + 1, self.max_duration - 1)
        finite = np.isfinite(samples)
        if discrepancy_sd_days > 0 and finite.any():
            samples[finite] += rng.normal(0, discrepancy_sd_days, finite.sum())
        return samples

    def predictive_transition_samples(
        self,
        daily: pd.DataFrame,
        n: int,
        rng: np.random.Generator,
        horizon: int = 45,
    ) -> np.ndarray:
        """Sample the next follicular-to-luteal transition from the filtered state.

        Particles already classified as luteal are recorded at the issue day,
        representing a model belief that the transition has already occurred.
        """
        _, f, l = self.filter_state(daily)
        combined = np.concatenate([f, l])
        combined /= combined.sum()
        initial = rng.choice(combined.size, size=n, p=combined)
        stage = (initial >= self.max_duration).astype(np.int8)
        age = initial % self.max_duration
        hf = self._hazard(self.follicular_mean, self.follicular_sd)
        samples = np.full(n, np.nan)
        issue_day = len(daily)
        samples[stage == 1] = issue_day
        for future in range(1, horizon + 1):
            active = np.isnan(samples)
            if not active.any():
                break
            hazard = hf[np.minimum(age, self.max_duration - 1)]
            transition = active & (rng.random(n) < hazard)
            samples[transition] = issue_day + future
            age[active & ~transition] = np.minimum(
                age[active & ~transition] + 1,
                self.max_duration - 1,
            )
        return samples
