"""Tests for Temperature Scaling calibration."""

import numpy as np
import pytest
from yolo_uncertainty.calibration.temperature import TemperatureScaling


def _make_overconfident_data(n: int = 500, seed: int = 42):
    """Synthetic overconfident detector: high raw confidence, ~60% accuracy."""
    rng = np.random.default_rng(seed)
    # Raw logits biased toward high confidence
    logits = rng.normal(loc=2.0, scale=1.0, size=n)
    probs = 1 / (1 + np.exp(-logits))
    # Ground truth: noisy (accuracy ~60%)
    labels = (rng.random(n) < 0.60).astype(float)
    return logits, probs, labels


class TestTemperatureScaling:
    def test_fit_returns_result(self):
        logits, _, labels = _make_overconfident_data()
        ts = TemperatureScaling()
        result = ts.fit(logits, labels)
        assert result.temperature > 0
        assert result.n_samples == 500

    def test_temperature_above_one_for_overconfident(self):
        """Overconfident model → T > 1 (softening)."""
        logits, _, labels = _make_overconfident_data()
        ts = TemperatureScaling()
        result = ts.fit(logits, labels)
        assert result.temperature > 1.0, (
            f"Expected T > 1 for overconfident data, got T={result.temperature:.4f}"
        )

    def test_ece_decreases_after_calibration(self):
        logits, _, labels = _make_overconfident_data()
        ts = TemperatureScaling()
        result = ts.fit(logits, labels)
        assert result.ece_after < result.ece_before, (
            f"ECE should decrease: before={result.ece_before:.4f}, after={result.ece_after:.4f}"
        )

    def test_calibrate_shape(self):
        logits, _, labels = _make_overconfident_data()
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        cal = ts.calibrate(logits[:50])
        assert cal.shape == (50,)
        assert np.all((cal >= 0) & (cal <= 1))

    def test_state_dict_roundtrip(self):
        logits, _, labels = _make_overconfident_data()
        ts = TemperatureScaling()
        ts.fit(logits, labels)
        T_original = ts.temperature

        ts2 = TemperatureScaling()
        ts2.load_state_dict(ts.state_dict())
        assert ts2.temperature == pytest.approx(T_original)

    def test_uncalibrated_raises(self):
        ts = TemperatureScaling()
        with pytest.raises(RuntimeError):
            ts.calibrate(np.array([0.5, 1.0]))
