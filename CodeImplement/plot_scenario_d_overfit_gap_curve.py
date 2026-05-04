# -*- coding: utf-8 -*-
"""
Generalization-gap curve for Scenario D.

Run after fl_scenario_d.py in the same Python process. This is the cleanest
overfitting view when both train and test MSE decrease but one method develops
a larger test-train gap.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
    }
)


def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.endswith(".pdf"):
        fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")


def pad_to_same_length(list_of_curves):
    length = max(len(c) for c in list_of_curves)
    padded = []
    for curve in list_of_curves:
        curve = np.asarray(curve, dtype=float)
        if len(curve) < length:
            curve = np.pad(curve, (0, length - len(curve)), mode="edge")
        padded.append(curve)
    return np.stack(padded, axis=0)


fl_test = pad_to_same_length(curves["fltl_test"])
fl_train = pad_to_same_length(curves["fltl_train"])
lo_test = pad_to_same_length(curves["local_test"])
lo_train = pad_to_same_length(curves["local_train"])

fl_gap = fl_test - fl_train
lo_gap = lo_test - lo_train

fl_gap_mean = fl_gap.mean(axis=0)
lo_gap_mean = lo_gap.mean(axis=0)
fl_gap_std = fl_gap.std(axis=0, ddof=1) if fl_gap.shape[0] > 1 else np.zeros_like(fl_gap_mean)
lo_gap_std = lo_gap.std(axis=0, ddof=1) if lo_gap.shape[0] > 1 else np.zeros_like(lo_gap_mean)
epochs_axis = np.arange(len(fl_gap_mean))

fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.plot(epochs_axis, fl_gap_mean, label="FL-TL", color="tab:blue", linestyle="-")
ax.plot(epochs_axis, lo_gap_mean, label="Local TL", color="tab:red", linestyle="--")
ax.fill_between(epochs_axis, fl_gap_mean - fl_gap_std, fl_gap_mean + fl_gap_std, color="tab:blue", alpha=0.15, linewidth=0)
ax.fill_between(epochs_axis, lo_gap_mean - lo_gap_std, lo_gap_mean + lo_gap_std, color="tab:red", alpha=0.15, linewidth=0)
ax.axhline(0.0, color="0.2", linewidth=0.8)
ax.set_xlabel("Fine-tuning epoch on target")
ax.set_ylabel("Generalization gap\n(test - train MSE)")
ax.set_title(f"{TARGET_IBR.upper()} - Overfitting gap across fine-tuning")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="upper left", frameon=False)

save_figure(fig, f"fig_tl_{TARGET_IBR}_overfit_gap_curve_mean.pdf")
plt.show()
