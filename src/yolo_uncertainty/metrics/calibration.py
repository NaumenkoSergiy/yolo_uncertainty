"""Calibration metrics: ECE, MCE, reliability diagram.

Definitions
-----------
Expected Calibration Error (ECE):
    ECE = Σ_m |B_m| / n · |acc(B_m) − conf(B_m)|

Maximum Calibration Error (MCE):
    MCE = max_m |acc(B_m) − conf(B_m)|

where B_m is the m-th confidence bin, acc and conf are
the mean accuracy and mean confidence within that bin.

Reference:
    Naeini, M. P. et al. "Obtaining Well Calibrated Probabilities
    Using Bayesian Binning into Quantiles." AAAI 2015.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Parameters
    ----------
    probs  : (N,) confidence scores ∈ [0, 1]
    labels : (N,) binary ground-truth
    n_bins : number of equal-width bins (default 15)

    Returns
    -------
    ECE ∈ [0, 1]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(probs)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        total += mask.sum() / n * abs(acc - conf)
    return float(total)


def mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error.

    Parameters
    ----------
    probs  : (N,) confidence scores
    labels : (N,) binary ground-truth
    n_bins : number of bins (default 15)

    Returns
    -------
    MCE ∈ [0, 1]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        max_err = max(max_err, abs(acc - conf))
    return float(max_err)


def reliability_diagram(
    probs_before: np.ndarray,
    labels: np.ndarray,
    probs_after: np.ndarray | None = None,
    n_bins: int = 15,
    out_path: str | Path | None = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """Plot reliability diagram (before and optionally after calibration).

    Parameters
    ----------
    probs_before : (N,) raw confidence scores
    labels       : (N,) binary ground-truth
    probs_after  : (N,) calibrated scores (optional)
    n_bins       : number of bins (default 15)
    out_path     : save figure to this path if provided
    title        : figure title

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    def _plot_line(probs: np.ndarray, label: str, color: str) -> None:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers, accs, confs, counts = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            accs.append(labels[mask].mean())
            confs.append(probs[mask].mean())
            counts.append(mask.sum())
        ax.plot(confs, accs, "o-", label=label, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    _plot_line(probs_before, "Before calibration", "#E74C3C")
    if probs_after is not None:
        _plot_line(probs_after, "After calibration", "#2ECC71")

    ece_b = ece(probs_before, labels, n_bins)
    info = f"ECE before = {ece_b:.4f}"
    if probs_after is not None:
        ece_a = ece(probs_after, labels, n_bins)
        info += f"\nECE after  = {ece_a:.4f}"
    ax.text(0.05, 0.75, info, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
