"""Energy-based OOD detection.

Reference:
    Liu, W. et al. "Energy-based Out-of-distribution Detection."
    NeurIPS 2020. https://arxiv.org/abs/2010.03759

Theory
------
For a classifier with logits f(x) ∈ ℝ^C, the energy score is:

    E(x) = −T · log Σ_c exp(f_c(x) / T)

In-distribution samples have LOW energy; OOD samples have HIGH energy.
A threshold τ is set on the calibration set at a chosen FPR level.

For detection: we extract the backbone feature vector of the full image
and apply the energy formula to it. If E(x) > τ → OOD flag.

Practical note
--------------
YOLOv8 does not expose per-class logits for the whole image directly.
We use the highest-confidence box's class logits as a proxy, or extract
the final neck feature map averaged over spatial dims.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EnergyResult:
    score: float          # E(x); lower = more in-distribution
    is_ood: bool
    threshold: float


class EnergyScore:
    """Threshold-based OOD detection using energy score.

    Parameters
    ----------
    temperature : float
        Temperature for the energy formula (default 1.0).
    fpr_target : float
        Target false-positive rate for threshold selection (default 0.05).

    Examples
    --------
    >>> es = EnergyScore(temperature=1.0, fpr_target=0.05)
    >>> es.fit(in_logits)         # logits from in-distribution images
    >>> result = es.score(test_logits)
    >>> result.is_ood
    False
    """

    def __init__(self, temperature: float = 1.0, fpr_target: float = 0.05):
        self.temperature = temperature
        self.fpr_target = fpr_target
        self._threshold: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, in_logits: np.ndarray) -> float:
        """Fit threshold on in-distribution logits.

        Parameters
        ----------
        in_logits : ndarray of shape (N, C) or list of (C,) arrays
            Class logits from N in-distribution images.

        Returns
        -------
        threshold : float
        """
        energies = np.array([self._energy(lg) for lg in in_logits])
        # Threshold at (1 - fpr_target) quantile of in-dist energies
        self._threshold = float(np.quantile(energies, 1.0 - self.fpr_target))
        return self._threshold

    def score(self, logits: np.ndarray) -> EnergyResult:
        """Compute energy score for a single image.

        Parameters
        ----------
        logits : ndarray of shape (C,)

        Returns
        -------
        EnergyResult
        """
        if self._threshold is None:
            raise RuntimeError("Call .fit() with in-distribution logits first.")
        e = self._energy(logits)
        return EnergyResult(
            score=e,
            is_ood=e > self._threshold,
            threshold=self._threshold,
        )

    def batch_score(self, logits_list: list[np.ndarray]) -> list[EnergyResult]:
        """Score a list of images."""
        return [self.score(lg) for lg in logits_list]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _energy(self, logits: np.ndarray) -> float:
        """E(x) = -T * log Σ_c exp(f_c / T)"""
        T = self.temperature
        return float(-T * np.log(np.sum(np.exp(logits / T))))

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("Not fitted yet.")
        return self._threshold
