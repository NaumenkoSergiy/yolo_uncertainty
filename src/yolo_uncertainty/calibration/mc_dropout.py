"""MC Dropout — Bayesian approximation for detection uncertainty.

Reference:
    Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation."
    ICML 2016. https://arxiv.org/abs/1506.02142

Theory
------
During inference, Dropout layers are kept ACTIVE (train mode).
Running N stochastic forward passes gives a distribution over confidence:

    {conf_1, conf_2, ..., conf_N}

We report:
    mean_conf   = mean of N passes  (calibrated expected confidence)
    std_conf    = std  of N passes  (epistemic uncertainty estimate)
    entropy     = -Σ p_i log p_i    (predictive uncertainty)

Practical choice: N = 10–20 passes is sufficient for a reliable estimate.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class MCResult:
    """Output of a single MC Dropout inference call."""
    mean_conf: np.ndarray     # (num_boxes,)  mean confidence across N passes
    std_conf:  np.ndarray     # (num_boxes,)  std  confidence across N passes
    entropy:   np.ndarray     # (num_boxes,)  binary entropy of mean_conf
    raw_passes: np.ndarray    # (N, num_boxes) all individual confidences


class MCDropout:
    """Wrap any YOLOv8 inference function and run N stochastic passes.

    Parameters
    ----------
    infer_fn : callable
        A function that takes an image (np.ndarray HxWx3) and returns
        a list of confidence scores (np.ndarray of shape (K,)) for K boxes.
        It must run with dropout ACTIVE (model.train() or custom hook).
    n_passes : int
        Number of Monte-Carlo forward passes (default 15).

    Examples
    --------
    >>> from ultralytics import YOLO
    >>> model = YOLO("yolov8m.pt")
    >>> model.model.train()   # keep dropout active
    >>> def infer(img):
    ...     results = model(img, verbose=False)
    ...     return results[0].boxes.conf.cpu().numpy()
    >>> mc = MCDropout(infer_fn=infer, n_passes=15)
    >>> result = mc.predict(image)
    >>> print(result.mean_conf, result.std_conf)
    """

    def __init__(self, infer_fn: Callable, n_passes: int = 15):
        self.infer_fn = infer_fn
        self.n_passes = n_passes

    def predict(self, image: np.ndarray) -> MCResult:
        """Run N stochastic passes and aggregate uncertainty.

        Parameters
        ----------
        image : np.ndarray of shape (H, W, 3), uint8 BGR

        Returns
        -------
        MCResult with mean, std, entropy across passes.

        Notes
        -----
        The number of detected boxes can differ across passes.
        We use the first pass as the reference box set and collect
        confidence from subsequent passes for the same boxes by
        nearest-IOU matching. For simplicity in this prototype we
        assume consistent box count (single-pass NMS filtering).
        """
        passes: list[np.ndarray] = []

        for _ in range(self.n_passes):
            conf = self.infer_fn(image)   # (K,)
            passes.append(conf)

        # Align lengths to minimum detected boxes
        min_k = min(len(p) for p in passes)
        stacked = np.stack([p[:min_k] for p in passes], axis=0)  # (N, K)

        mean_conf = stacked.mean(axis=0)   # (K,)
        std_conf  = stacked.std(axis=0)    # (K,)
        # Binary entropy
        eps = 1e-9
        p = np.clip(mean_conf, eps, 1 - eps)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        return MCResult(
            mean_conf=mean_conf,
            std_conf=std_conf,
            entropy=entropy,
            raw_passes=stacked,
        )

    def uncertainty_summary(self, result: MCResult) -> dict:
        """Human-readable summary of uncertainty for a prediction."""
        return {
            "mean_confidence":   float(result.mean_conf.mean()),
            "mean_std":          float(result.std_conf.mean()),
            "mean_entropy":      float(result.entropy.mean()),
            "high_uncertainty":  int((result.std_conf > 0.10).sum()),
            "n_boxes":           len(result.mean_conf),
            "n_passes":          self.n_passes,
        }
