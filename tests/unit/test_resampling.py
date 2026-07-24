import numpy as np
import pytest

from digital_twin.inference.diagnostics import normalize_log_weights
from digital_twin.inference.resampling import RESAMPLERS, effective_sample_size


def test_log_weight_normalization_is_stable():
    weights, normalizer = normalize_log_weights(np.array([-10000.0, -10001.0, -10002.0]))
    assert np.isclose(weights.sum(), 1)
    assert np.isfinite(normalizer)
    assert weights[0] > weights[1] > weights[2]


def test_ess_extremes():
    assert effective_sample_size(np.full(10, 0.1)) == pytest.approx(10)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)


@pytest.mark.parametrize("name", sorted(RESAMPLERS))
def test_resamplers_return_valid_indices(name):
    rng = np.random.default_rng(4)
    indices = RESAMPLERS[name](np.array([0.05, 0.15, 0.30, 0.50]), rng)
    assert indices.shape == (4,)
    assert indices.min() >= 0 and indices.max() < 4
