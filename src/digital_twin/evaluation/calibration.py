from __future__ import annotations

import numpy as np
import pandas as pd

from digital_twin.evaluation.metrics import coverage


def coverage_table(sample_sets: list[np.ndarray], truths: np.ndarray, levels: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for level in levels:
        values = [coverage(samples, float(truth), level)[0] for samples, truth in zip(sample_sets, truths)]
        empirical = float(np.mean(values)) if values else np.nan
        rows.append({"nominal": level, "empirical": empirical, "calibration_error": empirical - level if values else np.nan, "n": len(values)})
    return pd.DataFrame(rows)


def pit_values(sample_sets: list[np.ndarray], truths: np.ndarray) -> np.ndarray:
    return np.array([np.mean(np.asarray(samples) <= truth) for samples, truth in zip(sample_sets, truths)], dtype=float)
