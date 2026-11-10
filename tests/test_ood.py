"""Tests for OOD detectors: Energy Score and Mahalanobis Distance."""

import numpy as np
import pytest
from yolo_uncertainty.ood.energy import EnergyScore
from yolo_uncertainty.ood.mahalanobis import MahalanobisDetector


def _make_logits(n: int = 300, C: int = 10, seed: int = 1):
    rng = np.random.default_rng(seed)
    # In-distribution: logits centered near one class
    in_logits = rng.normal(loc=0, scale=1, size=(n, C))
    in_logits[:, 0] += 2.0
    # OOD: uniform logits (no confident class)
    ood_logits = rng.uniform(-3, 3, size=(50, C))
    return in_logits, ood_logits


class TestEnergyScore:
    def test_fit_returns_threshold(self):
        in_logits, _ = _make_logits()
        es = EnergyScore()
        t = es.fit(in_logits)
        assert isinstance(t, float)

    def test_in_dist_mostly_not_ood(self):
        in_logits, _ = _make_logits(n=500)
        es = EnergyScore(fpr_target=0.05)
        es.fit(in_logits[:400])
        results = [es.score(lg) for lg in in_logits[400:]]
        fpr = sum(r.is_ood for r in results) / len(results)
        assert fpr <= 0.15, f"FPR too high on in-dist: {fpr:.2f}"

    def test_ood_higher_energy_than_in(self):
        in_logits, ood_logits = _make_logits()
        es = EnergyScore()
        es.fit(in_logits)
        in_scores  = [es.score(lg).score for lg in in_logits[:20]]
        ood_scores = [es.score(lg).score for lg in ood_logits[:20]]
        # OOD should have higher (less negative) energy on average
        assert np.mean(ood_scores) > np.mean(in_scores)

    def test_unfitted_raises(self):
        es = EnergyScore()
        with pytest.raises(RuntimeError):
            es.score(np.zeros(10))


class TestMahalanobisDetector:
    def _make_features(self, seed: int = 2):
        rng = np.random.default_rng(seed)
        D = 32
        n_classes = 3
        features, labels = [], []
        for c in range(n_classes):
            mu = rng.normal(0, 2, D)
            feat = rng.normal(mu, 0.5, (100, D))
            features.append(feat)
            labels.extend([c] * 100)
        X = np.vstack(features)
        y = np.array(labels)
        # OOD: far from all class centres
        ood = rng.normal(10, 0.5, (30, D))
        return X, y, ood

    def test_fit_returns_threshold(self):
        X, y, _ = self._make_features()
        md = MahalanobisDetector()
        t = md.fit(X, y)
        assert isinstance(t, float) and t > 0

    def test_in_dist_low_score(self):
        X, y, _ = self._make_features()
        md = MahalanobisDetector(fpr_target=0.05)
        md.fit(X[:270], y[:270])
        scores = [md.score(x).score for x in X[270:]]
        flags = [md.score(x).is_ood for x in X[270:]]
        fpr = sum(flags) / len(flags)
        assert fpr <= 0.20

    def test_ood_flagged(self):
        X, y, ood = self._make_features()
        md = MahalanobisDetector()
        md.fit(X, y)
        tpr_ood = sum(md.score(x).is_ood for x in ood) / len(ood)
        assert tpr_ood >= 0.70, f"OOD detection rate too low: {tpr_ood:.2f}"

    def test_unfitted_raises(self):
        md = MahalanobisDetector()
        with pytest.raises(RuntimeError):
            md.score(np.zeros(10))
