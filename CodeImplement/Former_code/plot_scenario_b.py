"""
Plot results for Scenario B (Join vs NoJoin) from FL_SCENARIO_B.ipynb.

Usage:
    After running all cells in FL_SCENARIO_B.ipynb, run:
        %run plot_scenario_b.py

Required variables already in the notebook session:
    - rounds_axis, JOIN_ROUND
    - hist_nojoin, hist_join           (contain test curves)
    - global_state_nojoin, global_state_join
    - test_sets_gfli, HIDDEN_GFLI
    - evaluate_tests_full, set_model_state_dict, FullModel, Trunk, Head, device, input_dim, output_dim
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.lines import Line2D


# Matplotlib style (compact, IEEE-like)
mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
    }
)

NBINS_ADMITTANCE = 25
ZB_BASE_OHM = 95.2  # base impedance (Ohm) -> convert Y from pu to Siemens
COLOR_JOIN = "#1f77b4"     # tableau blue
COLOR_NOJOIN = "#ff7f0e"   # tableau orange
COLOR_POS = "#2ca02c"      # tableau green
COLOR_NEG = "#d62728"      # tableau red
COLOR_CUM = "#9467bd"      # tableau purple


def _require(names, scope):
    missing = [n for n in names if n not in scope]
    if missing:
        raise RuntimeError(
            f"Missing variables from notebook run: {', '.join(missing)}. "
            "Please run FL_SCENARIO_B.ipynb completely, then rerun this script."
        )


def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.endswith(".pdf"):
        fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")


REQUIRED_VARS = [
    "rounds_axis",
    "JOIN_ROUND",
    "hist_nojoin",
    "hist_join",
    "global_state_nojoin",
    "global_state_join",
    "test_sets_gfli",
    "HIDDEN_GFLI",
    "evaluate_tests_full",
    "set_model_state_dict",
    "FullModel",
    "Trunk",
    "Head",
    "device",
    "input_dim",
    "output_dim",
    "X_scaler_gfli",
    "Y_scaler_gfli",
]
_require(REQUIRED_VARS, globals())


# -----------------------------------------------------------------------------
# Figure 1: Learning curves (4 lines in one axis)
# -----------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(3.5, 2.0))
ax1.plot(
    rounds_axis,
    hist_nojoin["test_mse_mean_old"],
    label="Old clients (NoJoin)",
    linewidth=2.0,
)
ax1.plot(
    rounds_axis,
    hist_join["test_mse_mean_old"],
    label="Old clients (Join)",
    linestyle="--",
    linewidth=2.0,
)
ax1.plot(
    rounds_axis,
    hist_nojoin["test_mse_new"],
    label="New client (NoJoin / zero-shot)",
    linewidth=2.0,
)
ax1.plot(
    rounds_axis,
    hist_join["test_mse_new"],
    label="New client (Join)",
    linestyle="--",
    linewidth=2.0,
)
ax1.axvline(JOIN_ROUND, color="gray", linestyle=":", label="Join round")
ax1.set_xlabel("Global round")
ax1.set_ylabel("Test MSE")
ax1.set_title("Learning curves (Join vs NoJoin)")
ax1.grid(True, ls=":", alpha=0.5)
ax1.legend(loc="upper right", frameon=False, ncol=1, fontsize=7)
save_figure(fig1, "fig_scenario_b_learning_curves.pdf")
plt.show()


# -----------------------------------------------------------------------------
# Per-client evaluation and fairness
# -----------------------------------------------------------------------------
def _to_mse_dict(results):
    return {name: float(mse) for name, mse in results}


print("\nEvaluating per-client test MSE for Join vs NoJoin ...")
mse_tests_nojoin = evaluate_tests_full(global_state_nojoin, test_sets_gfli, HIDDEN_GFLI)
mse_tests_join = evaluate_tests_full(global_state_join, test_sets_gfli, HIDDEN_GFLI)

mse_nojoin_dict = _to_mse_dict(mse_tests_nojoin)
mse_join_dict = _to_mse_dict(mse_tests_join)
client_order = sorted(mse_nojoin_dict.keys())
mse_nojoin = [mse_nojoin_dict[c] for c in client_order]
mse_join = [mse_join_dict.get(c, float("nan")) for c in client_order]


def jain_index(mse_list, eps=1e-8):
    arr = np.array(mse_list, dtype=float)
    util = 1.0 / (arr + eps)
    return float((util.sum() ** 2) / (len(util) * (util ** 2).sum()))


jain_nojoin = jain_index(mse_nojoin)
jain_join = jain_index(mse_join)

print("Final per-client test MSE (NoJoin -> Join):")
for c, mn, mj in zip(client_order, mse_nojoin, mse_join):
    print(f"  {c}: {mn:.4e} -> {mj:.4e} (impr {(mn - mj) / mn * 100:+.2f}% )")
print(f"Jain fairness: NoJoin={jain_nojoin:.4f}, Join={jain_join:.4f}")


# -----------------------------------------------------------------------------
# Figure 2: Per-client bar chart (Join vs NoJoin)
# -----------------------------------------------------------------------------
x = np.arange(len(client_order))
width = 0.38
client_labels = []
for name in client_order:
    lbl = name.replace("_test_impedance_dataset", "").replace("_impedance_dataset", "")
    lbl = lbl.replace("gfli", "IBR")
    client_labels.append(lbl)
fig3, ax3 = plt.subplots(figsize=(7.0, 2.0))
ax3.bar(
    x - width / 2,
    mse_nojoin,
    width,
    label="NoJoin",
    color=COLOR_NOJOIN,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85,
)
ax3.bar(
    x + width / 2,
    mse_join,
    width,
    label="Join",
    color=COLOR_JOIN,
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85,
)
ax3.set_ylabel("Test MSE")
ax3.set_xlabel("Client")
ax3.set_title("Per-client test MSE")
ax3.set_xticks(x)
ax3.set_xticklabels(client_labels, rotation=0, ha="center")
ax3.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
save_figure(fig3, "fig_scenario_b_per_client.pdf")
plt.show()




# -----------------------------------------------------------------------------
# Figure 4: Fairness / distribution (boxplot)
# -----------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(3.5, 2.0))
bp = ax5.boxplot(
    [mse_nojoin, mse_join],
    labels=["NoJoin", "Join"],
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
)
bp["boxes"][0].set_facecolor(COLOR_NOJOIN)
bp["boxes"][0].set_alpha(0.45)
bp["boxes"][1].set_facecolor(COLOR_JOIN)
bp["boxes"][1].set_alpha(0.45)
ax5.set_ylabel("Test MSE")
ax5.set_title("Per-client MSE distribution")
ax5.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax5.text(
    0.98,
    0.98,
    f"Jain -> NoJoin: {jain_nojoin:.3f}\nJain -> Join:   {jain_join:.3f}",
    transform=ax5.transAxes,
    fontsize=7,
    va="top",
    ha="right",
    bbox=dict(boxstyle="round", facecolor="#F5E6C8", alpha=0.7),
)
save_figure(fig5, "fig_scenario_b_fairness.pdf")
plt.show()


print("\nSummary:")
print(f"  Jain fairness -> NoJoin: {jain_nojoin:.4f} | Join: {jain_join:.4f}")
print(f"  Mean MSE -> NoJoin: {np.mean(mse_nojoin):.4e} | Join: {np.mean(mse_join):.4e}")
best_clients = sorted(zip(client_order, improvements), key=lambda x: x[1], reverse=True)[:3]
print("  Top improvements (NoJoin -> Join):")
for name, imp in best_clients:
    print(f"    {name}: {imp:+.2f}%")
