import numpy as np
import pandas as pd

from digital_twin.baselines.hsmm import HSMMBaseline


def test_hsmm_produces_first_passage_samples():
    daily = pd.DataFrame({
        "bleeding": [3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        "temperature": [36.4, 36.4, 36.45, 36.42, 36.48, 36.5, 36.51, 36.55, 36.55, 36.6],
        "lh": [3, 3, 3, 3, 4, 5, 9, 15, 6, 3],
    })
    samples = HSMMBaseline().predictive_samples(daily, 300, np.random.default_rng(4))
    assert np.isfinite(samples).mean() > 0.95
    assert np.nanmedian(samples) > len(daily)
