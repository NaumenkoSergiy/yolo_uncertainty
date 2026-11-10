"""Mahalanobis Distance OOD detection.

Reference:
    Lee, K. et al. "A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks."
    NeurIPS 2018. https://arxiv.org/abs/1807.03888

Theory
------
For each class c, fit a Gaussian N(μ_c, Σ) on the penultimate-layer
feature vectors of training/calibration samples.

At test time:

    M(x) = max_c  (f(x) − μ_c)^T Σ^{-1} (f(x) − μ_c)

Low M → close to a known class distribution → in-distribution.
High M → far from all class centres → OOD.

The shared covariance Σ is estimated via tied covariance across classes.

Practical note
--------------
For YOLOv8 we hook the final backbone (C2f) layer output, average
the spatial dimensions to get a feature vector of shape (C_feat,).
See `scripts/extract_features.py` for the hook setup.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MahalanobisResult:
    score: float          # M(x); lower = more in-distribution
    nearest_class: int    # class with minimum Mahalanobis distance
    is_ood: bool
    threshold: float


class MahalanobisDetector:
    """Class-conditional Mahalanobis distance OOD detector.

    Parameters
    ----------
    fpr_target : float
        Target FPR for threshold selection on calibration set (default 0.05).

    Examples
    --------
    >>> md = MahalanobisDetector()
    >>> md.fit(features, labels)    # features: (N, D), labels: (N,) int class ids
    >>> result = md.score(test_feat)
    >>> result.is_ood
    False
    """

    def __init__(self, fpr_target: float = 0.05):
        self.fpr_target = fpr_target
        self._means: dict[int, np.ndarray] = {}
        self._precision: np.ndarray | None = None   # Σ^{-1}
        self._threshold: float | None = None
        self._classes: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray,   # (N, D)
        labels: np.ndarray,     # (N,) int — class ids
        cal_features: np.ndarray | None = None,
    ) -> float:
        """Fit class means and shared covariance.

        Parameters
        ----------
        features : (N, D) — in-distribution training features
        labels   : (N,)   — class assignments
        cal_features : (M, D) optional held-out features for threshold fitting.
            If None, uses training features (optimistic threshold).

        Returns
        -------
        threshold : float
        """
        self._classes = sorted(set(labels.tolist()))
        D = features.shape[1]

        # Class-conditional means
        for c in self._classes:
            self._means[c] = features[labels == c].mean(axis=0)

        # Tied covariance
        cov = np.zeros((D, D))
        for c in self._classes:
            diff = features[labels == c] - self._means[c]
            cov += diff.T @ diff
        cov /= max(len(features) - len(self._classes), 1)
        cov += 1e-6 * np.eye(D)   # regularisation
        self._precision = np.linalg.inv(cov)

        # Threshold on calibration set
        ref = cal_features if cal_features is not None else features
        scores = np.array([self._mahal(f) for f in ref])
        self._threshold = float(np.quantile(scores, 1.0 - self.fpr_target))
        return self._threshold

    def score(self, feature: np.ndarray) -> MahalanobisResult:
        """OOD score for a single feature vector.

        Parameters
        ----------
        feature : ndarray (D,)

        Returns
        -------
        MahalanobisResult
        """
        if self._precision is None:
            raise RuntimeError("Call .fit() first.")
        m, nearest = self._mahal_with_class(feature)
        return MahalanobisResult(
            score=m,
            nearest_class=nearest,
            is_ood=m > self._threshold,
            threshold=self._threshold,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mahal(self, f: np.ndarray) -> float:
        return self._mahal_with_class(f)[0]

    def _mahal_with_class(self, f: np.ndarray) -> tuple[float, int]:
        best_score = np.inf
        best_class = -1
        for c, mu in self._means.items():
            diff = f - mu
            d = float(diff @ self._precision @ diff)
            if d < best_score:
                best_score = d
                best_class = c
        return best_score, best_class

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Not fitted yet.")
        return self._threshold
