"""High-level Calibrator and OODDetector wrappers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from yolo_uncertainty.calibration.temperature import TemperatureScaling
from yolo_uncertainty.calibration.mc_dropout import MCDropout
from yolo_uncertainty.calibration.conformal import RAPS
from yolo_uncertainty.ood.energy import EnergyScore
from yolo_uncertainty.ood.mahalanobis import MahalanobisDetector


CalibrationMethod = Literal["temperature", "mc_dropout", "raps"]
OODMethod = Literal["energy", "mahalanobis"]


class Calibrator:
    """Unified interface for all calibration methods.

    Parameters
    ----------
    method : one of "temperature", "mc_dropout", "raps"
    **kwargs : forwarded to the underlying calibration class

    Examples
    --------
    >>> cal = Calibrator(method="temperature")
    >>> cal.fit(logits, labels)
    >>> cal_probs = cal.transform(logits)
    """

    def __init__(self, method: CalibrationMethod = "temperature", **kwargs):
        self.method = method
        if method == "temperature":
            self._model = TemperatureScaling(**kwargs)
        elif method == "raps":
            self._model = RAPS(**kwargs)
        elif method == "mc_dropout":
            self._model = MCDropout(**kwargs)
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, method: CalibrationMethod = "temperature", **kwargs) -> "Calibrator":
        """Create calibrator and load YOLOv8 checkpoint (placeholder)."""
        # In a full implementation this would load the YOLO model and
        # wire it to the calibration method.
        return cls(method=method, **kwargs)

    def fit(self, logits_or_confs: np.ndarray, labels: np.ndarray):
        """Fit calibration model on calibration set."""
        if self.method in ("temperature",):
            return self._model.fit(logits_or_confs, labels)
        elif self.method == "raps":
            return self._model.fit(logits_or_confs, labels)
        else:
            raise NotImplementedError(f"fit() not supported for method={self.method!r}")

    def transform(self, logits_or_confs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""
        if self.method == "temperature":
            return self._model.calibrate(logits_or_confs)
        elif self.method == "raps":
            return self._model.predict_set(logits_or_confs).astype(float)
        else:
            raise NotImplementedError(f"transform() not supported for method={self.method!r}")


class OODDetector:
    """Unified interface for OOD detection methods.

    Parameters
    ----------
    method : one of "energy", "mahalanobis"
    **kwargs : forwarded to the underlying detector class
    """

    def __init__(self, method: OODMethod = "energy", **kwargs):
        self.method = method
        if method == "energy":
            self._model = EnergyScore(**kwargs)
        elif method == "mahalanobis":
            self._model = MahalanobisDetector(**kwargs)
        else:
            raise ValueError(f"Unknown OOD method: {method!r}")

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None):
        """Fit OOD threshold on in-distribution features."""
        if self.method == "energy":
            return self._model.fit(features)
        elif self.method == "mahalanobis":
            if labels is None:
                raise ValueError("Mahalanobis requires class labels for fitting.")
            return self._model.fit(features, labels)

    def score(self, feature: np.ndarray):
        """Score a single feature vector."""
        return self._model.score(feature)
