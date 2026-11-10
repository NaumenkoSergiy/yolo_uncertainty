"""RAPS — Regularized Adaptive Prediction Sets for detection confidence.

Reference:
    Romano, Y. et al. "Classification with Valid and Adaptive Coverage."
    NeurIPS 2020. https://arxiv.org/abs/2009.14193

Theory
------
Conformal prediction gives a *coverage guarantee*:
    P(label ∈ Ĉ(x)) ≥ 1 − α

for any user-chosen miscoverage level α (e.g. α=0.05 → 95% coverage).

For object detection we treat each bounding box independently:
    1. Compute non-conformity score s_i = 1 − conf_i  for calibration boxes.
    2. Find quantile q̂ at level ⌈(n+1)(1−α)⌉/n of sorted scores.
    3. At test time: box is included in prediction set if s_test ≤ q̂,
       i.e. conf_test ≥ 1 − q̂.

RAPS adds a regularisation penalty on set size to avoid trivially large sets.

Usage
-----
>>> raps = RAPS(alpha=0.05)
>>> raps.fit(val_confs, val_labels)      # calibrate on validation set
>>> threshold = raps.threshold
>>> accepted = test_confs >= threshold   # True = "in prediction set"
>>> coverage = raps.empirical_coverage(test_confs, test_labels)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RAPSResult:
    alpha: float
    threshold: float          # conf ≥ threshold → included in prediction set
    empirical_coverage: float
    n_calibration: int
    quantile_index: int


class RAPS:
    """Regularized Adaptive Prediction Sets for detection boxes.

    Parameters
    ----------
    alpha : float
        Miscoverage level, e.g. 0.05 for 95% coverage (default 0.05).
    kreg : int
        Regularisation onset — penalise sets larger than kreg (default 1).
    lamda : float
        Regularisation strength (default 0.01).
    """

    def __init__(self, alpha: float = 0.05, kreg: int = 1, lamda: float = 0.01):
        self.alpha = alpha
        self.kreg = kreg
        self.lamda = lamda
        self._threshold: float | None = None
        self._result: RAPSResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, cal_confs: np.ndarray, cal_labels: np.ndarray) -> RAPSResult:
        """Calibrate on held-out validation boxes.

        Parameters
        ----------
        cal_confs : ndarray (N,)
            Confidence scores for calibration boxes.
        cal_labels : ndarray (N,)
            Binary labels (1 = TP, 0 = FP).

        Returns
        -------
        RAPSResult with threshold and empirical coverage.
        """
        # Non-conformity scores (higher = less confident)
        scores = 1.0 - cal_confs

        # RAPS regularisation: penalise low-confidence boxes
        penalties = self.lamda * np.maximum(0, np.arange(len(scores)) - self.kreg + 1)
        scores_reg = scores + penalties[: len(scores)]

        n = len(scores_reg)
        idx = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        idx = min(idx, n - 1)

        sorted_scores = np.sort(scores_reg)
        q_hat = float(sorted_scores[idx])
        self._threshold = 1.0 - q_hat

        # Empirical coverage on calibration set
        predicted = cal_confs >= self._threshold
        cov = float(predicted[cal_labels == 1].mean()) if cal_labels.sum() > 0 else 0.0

        self._result = RAPSResult(
            alpha=self.alpha,
            threshold=self._threshold,
            empirical_coverage=cov,
            n_calibration=n,
            quantile_index=idx,
        )
        return self._result

    def predict_set(self, confs: np.ndarray) -> np.ndarray:
        """Return boolean mask: True = box included in prediction set.

        Parameters
        ----------
        confs : ndarray (N,)

        Returns
        -------
        ndarray (N,) bool
        """
        if self._threshold is None:
            raise RuntimeError("Call .fit() before .predict_set()")
        return confs >= self._threshold

    def empirical_coverage(self, confs: np.ndarray, labels: np.ndarray) -> float:
        """Fraction of TP boxes that are included in the prediction set."""
        in_set = self.predict_set(confs)
        tp_mask = labels == 1
        if tp_mask.sum() == 0:
            return 0.0
        return float(in_set[tp_mask].mean())

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Not fitted yet.")
        return self._threshold

    @property
    def result(self) -> RAPSResult | None:
        return self._result
