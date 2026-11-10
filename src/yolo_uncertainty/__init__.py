"""
yolo-uncertainty
================
Post-hoc confidence calibration and OOD detection for YOLOv8
on aerial imagery (VisDrone-DET 2019).

Usage::

    from yolo_uncertainty import Calibrator, OODDetector

    cal = Calibrator.from_checkpoint("yolov8m.pt", method="temperature")
    cal.fit(val_loader)
    preds = cal.predict(image)
    # preds: list of dicts with keys: bbox, class, raw_conf, cal_conf

    ood = OODDetector(cal.model, method="energy")
    ood.fit(val_loader)
    flag = ood.score(image)   # "in" | "out"
"""

from yolo_uncertainty.calibration.temperature import TemperatureScaling
from yolo_uncertainty.calibration.mc_dropout import MCDropout
from yolo_uncertainty.calibration.conformal import RAPS
from yolo_uncertainty.ood.energy import EnergyScore
from yolo_uncertainty.ood.mahalanobis import MahalanobisDetector
from yolo_uncertainty.pipeline import Calibrator, OODDetector

__version__ = "0.1.0"
__all__ = [
    "Calibrator",
    "OODDetector",
    "TemperatureScaling",
    "MCDropout",
    "RAPS",
    "EnergyScore",
    "MahalanobisDetector",
]
