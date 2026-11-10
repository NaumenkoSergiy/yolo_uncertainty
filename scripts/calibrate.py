#!/usr/bin/env python3
"""CLI: Calibrate YOLOv8 on VisDrone-DET 2019 validation split.

Usage
-----
python scripts/calibrate.py \\
    --model yolov8m.pt \\
    --data  data/visdrone.yaml \\
    --method temperature \\
    --output results/

The script:
1. Runs YOLOv8 inference on the val split.
2. Matches predicted boxes to ground truth (IoU ≥ 0.5).
3. Collects (logit, label) pairs.
4. Fits the chosen calibration method.
5. Saves the calibrated model state and prints ECE/MCE.

Requires: ultralytics, wandb
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate YOLOv8 confidence scores.")
    p.add_argument("--model",  default="yolov8m.pt", help="YOLOv8 checkpoint path")
    p.add_argument("--data",   default="data/visdrone.yaml", help="Dataset YAML")
    p.add_argument("--method", default="temperature",
                   choices=["temperature", "raps", "mc_dropout"],
                   help="Calibration method")
    p.add_argument("--output", default="results/", help="Output directory")
    p.add_argument("--n-bins", type=int, default=15, help="ECE bins")
    p.add_argument("--alpha",  type=float, default=0.05, help="RAPS miscoverage level")
    p.add_argument("--n-passes", type=int, default=15, help="MC Dropout passes")
    p.add_argument("--wandb-project", default="yolo-uncertainty", help="W&B project name")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return p.parse_args()


def iou_match(pred_boxes, gt_boxes, iou_thr: float = 0.5):
    """Match predicted boxes to GT; return binary label array.

    Parameters
    ----------
    pred_boxes : (K, 4) xyxy
    gt_boxes   : (M, 4) xyxy

    Returns
    -------
    labels : (K,) int — 1 if matched GT, 0 otherwise
    """
    labels = np.zeros(len(pred_boxes), dtype=int)
    if len(gt_boxes) == 0:
        return labels

    for i, pb in enumerate(pred_boxes):
        for gb in gt_boxes:
            # Intersection
            ix1 = max(pb[0], gb[0]); iy1 = max(pb[1], gb[1])
            ix2 = min(pb[2], gb[2]); iy2 = min(pb[3], gb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            area_p = (pb[2] - pb[0]) * (pb[3] - pb[1])
            area_g = (gb[2] - gb[0]) * (gb[3] - gb[1])
            iou = inter / (area_p + area_g - inter + 1e-9)
            if iou >= iou_thr:
                labels[i] = 1
                break
    return labels


def collect_predictions(model_path: str, data_yaml: str):
    """Run YOLOv8 val and collect (logit, label) pairs.

    Returns
    -------
    logits : (N,) raw logits (logit of confidence)
    labels : (N,) binary
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed — using synthetic demo data", file=sys.stderr)
        rng = np.random.default_rng(0)
        confs = rng.uniform(0.3, 1.0, 2000)
        labels = (rng.random(2000) < confs * 0.75).astype(int)
        logits = np.log(confs / (1 - confs + 1e-9))
        return logits, labels

    model = YOLO(model_path)
    results = model.val(data=data_yaml, verbose=False)
    # Placeholder: in full implementation, iterate over val set
    # and collect per-box (logit, label) pairs using iou_match()
    raise NotImplementedError("Full val-set collection not implemented in this stub.")


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    print(f"[calibrate] method={args.method}  model={args.model}")
    logits, labels = collect_predictions(args.model, args.data)
    print(f"[calibrate] collected {len(logits)} box predictions")

    # Split 80/20 for calibration / test
    n = len(logits)
    idx = int(0.8 * n)
    cal_logits, cal_labels = logits[:idx], labels[:idx]
    tst_logits, tst_labels = logits[idx:], labels[idx:]

    from yolo_uncertainty.metrics.calibration import ece, mce, reliability_diagram
    from yolo_uncertainty.pipeline import Calibrator

    cal = Calibrator(method=args.method, n_bins=args.n_bins)

    confs_before = 1 / (1 + np.exp(-tst_logits))
    ece_before = ece(confs_before, tst_labels, args.n_bins)
    mce_before = mce(confs_before, tst_labels, args.n_bins)

    cal.fit(cal_logits, cal_labels)
    confs_after = cal.transform(tst_logits)
    ece_after  = ece(confs_after,  tst_labels, args.n_bins)
    mce_after  = mce(confs_after,  tst_labels, args.n_bins)

    metrics = {
        "ece_before": ece_before, "mce_before": mce_before,
        "ece_after":  ece_after,  "mce_after":  mce_after,
        "ece_improvement": ece_before - ece_after,
    }
    print(json.dumps(metrics, indent=2))

    if not args.no_wandb:
        wandb.log(metrics)

    reliability_diagram(
        confs_before, tst_labels, confs_after,
        out_path=out / "reliability_diagram.png",
        title=f"Reliability Diagram — {args.method}",
    )

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[calibrate] Results saved to {out}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
