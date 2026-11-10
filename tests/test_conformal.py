"""Tests for RAPS conformal prediction."""

import numpy as np
import pytest
from yolo_uncertainty.calibration.conformal import RAPS


def _make_data(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    confs = rng.uniform(0.3, 1.0, size=n)
    labels = (rng.random(n) < confs).astype(float)
    return confs, labels


class TestRAPS:
    def test_coverage_at_least_1_minus_alpha(self):
        confs, labels = _make_data(n=1000)
        alpha = 0.05
        raps = RAPS(alpha=alpha)
        raps.fit(confs[:800], labels[:800])
        cov = raps.empirical_coverage(confs[800:], labels[800:])
        # Should achieve at least 1-alpha coverage (with small tolerance)
        assert cov >= 1 - alpha - 0.05, (
            f"Coverage {cov:.3f} below target {1-alpha:.2f}"
        )

    def test_threshold_in_range(self):
        confs, labels = _make_data()
        raps = RAPS(alpha=0.10)
        raps.fit(confs, labels)
        assert 0.0 <= raps.threshold <= 1.0

    def test_predict_set_shape(self):
        confs, labels = _make_data()
        raps = RAPS()
        raps.fit(confs, labels)
        mask = raps.predict_set(confs[:20])
        assert mask.shape == (20,)
        assert mask.dtype == bool

    def test_unfitted_raises(self):
        raps = RAPS()
        with pytest.raises(RuntimeError):
            raps.predict_set(np.array([0.8, 0.9]))
