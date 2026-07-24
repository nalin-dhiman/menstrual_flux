from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


def spectral_diagnostics(residuals: np.ndarray, sampling_days: float = 1.0) -> dict[str, np.ndarray | float]:
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        raise ValueError("at least four residuals are required")
    centered = values - values.mean()
    autocorrelation = np.correlate(centered, centered, mode="full")[values.size - 1 :]
    autocorrelation /= max(autocorrelation[0], 1e-12)
    frequencies, power = periodogram(centered, fs=1 / sampling_days)
    positive = frequencies > 0
    peak_frequency = float(frequencies[positive][np.argmax(power[positive])]) if positive.any() else np.nan
    return {"autocorrelation": autocorrelation, "frequencies": frequencies, "power": power, "peak_frequency": peak_frequency}
