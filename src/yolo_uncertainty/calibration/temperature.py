"""Temperature Scaling — post-hoc calibration for detection confidence scores.

Reference:
    Guo, C. et al. "On Calibration of Modern Neural Networks."
    ICML 2017. https://arxiv.org/abs/1706.04599

Theory
------
A single scalar T (temperature) is learned on a held-out calibration set.
The calibrated probability for class c is:

    p̂_c = softmax(z / T)_c

For detection we apply the same idea to the per-box confidence score:

    conf_cal = σ(logit(conf_raw) / T)

where logit(p) = log(p / (1-p)) and σ is the sigmoid.
T > 1 → softens (reduces overconfidence).
T < 1 → sharpens.  T = 1 → no change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from dataclasses import dataclass, field


@dataclass
class TSResult:
    temperature: float
    ece_before: float
    ece_after: float
    n_samples: int
    history: list[float] = field(default_factory=list)


class TemperatureScaling:
    """Learn a single temperature parameter T on a calibration set.

    Parameters
    ----------
    n_bins : int
        Number of bins for ECE computation (default 15).
    max_iter : int
        Maximum L-BFGS iterations (default 50).
    lr : float
        Learning rate for L-BFGS (default 0.01).

    Examples
    --------
    >>> ts = TemperatureScaling()
    >>> ts.fit(logits, labels)      # logits: (N,), labels: (N,) binary
    >>> cal_probs = ts.calibrate(logits)
    """

    def __init__(self, n_bins: int = 15, max_iter: int = 50, lr: float = 0.01):
        self.n_bins = n_bins
        self.max_iter = max_iter
        self.lr = lr
        self._T = nn.Parameter(torch.ones(1))
        self._result: TSResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> TSResult:
        """Fit temperature T on calibration logits and binary labels.

        Parameters
        ----------
        logits : ndarray of shape (N,)
            Raw log-odds (logit) of detection confidence.
        labels : ndarray of shape (N,)
            Binary ground-truth labels (1 = TP, 0 = FP).

        Returns
        -------
        TSResult with fitted temperature and ECE before/after.
        """
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        probs_before = torch.sigmoid(logits_t).numpy()
        ece_before = self._ece(probs_before, labels)

        optimizer = LBFGS([self._T], lr=self.lr, max_iter=self.max_iter)
        loss_fn = nn.BCEWithLogitsLoss()
        history: list[float] = []

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(logits_t / self._T, labels_t)
            loss.backward()
            history.append(loss.item())
            return loss

        optimizer.step(closure)

        probs_after = self.calibrate(logits)
        ece_after = self._ece(probs_after, labels)

        self._result = TSResult(
            temperature=self._T.item(),
            ece_before=ece_before,
            ece_after=ece_after,
            n_samples=len(logits),
            history=history,
        )
        return self._result

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to new logits.

        Parameters
        ----------
        logits : ndarray of shape (N,)

        Returns
        -------
        ndarray of shape (N,) — calibrated probabilities.
        """
        if self._T is None:
            raise RuntimeError("Call .fit() before .calibrate()")
        t = self._T.item()
        return 1.0 / (1.0 + np.exp(-logits / t))

    @property
    def temperature(self) -> float:
        return self._T.item()

    @property
    def result(self) -> TSResult | None:
        return self._result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Expected Calibration Error (equal-width bins)."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(probs)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return float(ece)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"T": self._T.item()}

    def load_state_dict(self, d: dict) -> None:
        self._T = nn.Parameter(torch.tensor([d["T"]]))
