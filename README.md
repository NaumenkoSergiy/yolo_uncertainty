# yolo-uncertainty

> **Post-hoc confidence calibration and OOD detection for YOLOv8 object detectors on aerial imagery**

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: VisDrone-DET 2019](https://img.shields.io/badge/dataset-VisDrone--DET%202019-orange)](https://github.com/VisDrone/VisDrone-Dataset)

**Research project** — NEW R&D Generation grant, 2027  
**Author:** Науменко Сергій Васильович, ЧНУ ім. Б. Хмельницького  

---

## Problem

YOLOv8's raw confidence score is a poor probability estimate — a detector
reporting 0.90 confidence may be correct only 60% of the time. For aerial
inspection with UAVs, an overconfident detector can cause a critical wrong
decision. This library **calibrates** that score post-hoc and **flags**
out-of-distribution images, without retraining the model.

## What this does

```
Raw YOLOv8 output
    └─► Calibration module (Temperature / MC-Dropout / RAPS)
            └─► calibrated_confidence  ∈ [0, 1]
    └─► OOD detector (Energy score / Mahalanobis)
            └─► ood_flag  ∈ {in / out}
```

**Metrics improved:**
| Metric | Baseline | After calibration |
|--------|----------|-------------------|
| ECE    | ~0.18    | < 0.05 (target)   |
| MCE    | ~0.25    | < 0.08 (target)   |
| Conformal coverage @ α=0.05 | — | ≥ 95 % |

## Methods implemented

| Method | Type | File |
|--------|------|------|
| Temperature Scaling | Parametric calibration | `calibration/temperature.py` |
| Platt Scaling | Parametric calibration | `calibration/platt.py` |
| MC Dropout | Bayesian calibration | `calibration/mc_dropout.py` |
| RAPS / SAPS | Conformal prediction | `calibration/conformal.py` |
| Energy Score | OOD detection | `ood/energy.py` |
| Mahalanobis Distance | OOD detection | `ood/mahalanobis.py` |

## Dataset

[VisDrone-DET 2019](https://github.com/VisDrone/VisDrone-Dataset) —
~10 000 aerial images, 10 object classes (pedestrian, car, bicycle, etc.),
collected by UAVs over urban and suburban scenes.

Download script: `scripts/download_visdrone.sh`

## Quick start

```bash
git clone https://github.com/snaumenko/yolo-uncertainty
cd yolo-uncertainty
pip install -e ".[dev]"

# Run calibration on VisDrone val split
python scripts/calibrate.py \
    --model yolov8m.pt \
    --data data/visdrone.yaml \
    --method temperature \
    --output results/

# Generate reliability diagram + ECE report
python scripts/report.py --results results/
```

## Experiment tracking

All runs are logged to [Weights & Biases](https://wandb.ai).  
Set `WANDB_API_KEY` in `.env` before running.

## Project structure

```
yolo-uncertainty/
├── src/yolo_uncertainty/
│   ├── calibration/       # Calibration methods
│   ├── ood/               # OOD detectors
│   ├── metrics/           # ECE, MCE, reliability diagram
│   └── utils/             # Data loading, bbox utils
├── notebooks/             # EDA and result visualization
├── scripts/               # CLI entry points
├── tests/                 # pytest suite
├── data/                  # Dataset configs (not data itself)
└── results/               # Outputs (gitignored)
```

## References

1. Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.
2. Romano et al. "Regularized Adaptive Prediction Sets." NeurIPS 2020. (RAPS)
3. Liu et al. "Energy-based Out-of-distribution Detection." NeurIPS 2020.
4. Lee et al. "A Simple Unified Framework for Detecting OOD Samples." NeurIPS 2018.

## License

MIT © 2027 Serhii Naumenko
