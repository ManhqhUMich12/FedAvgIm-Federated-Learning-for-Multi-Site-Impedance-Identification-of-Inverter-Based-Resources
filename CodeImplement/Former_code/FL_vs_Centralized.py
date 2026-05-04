"""
FL vs Centralized FNN Comparison
Demonstrates FL performance compared to centralized training approach

YÊU CẦU: Notebook đã chạy trước và có sẵn các biến:
- history_fullfedavg, fullfedavg_model_state
- central_model, central_train_curve, central_test_curve
- test_sets_gfli, HIDDEN_GFLI
- FullModel, Trunk, Head, device, input_dim, output_dim
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib as mpl

# ================= IEEE-style Matplotlib config =================
mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
})

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")  # thêm dòng này
# ========================================================================
# 1. Convergence Speed Analysis: FL vs Centralized
# ========================================================================

print("\n===== FL vs Centralized: Convergence Speed Analysis =====\n")


def find_convergence_round(curve, target_ratio=0.95):
    """
    Find the round at which performance reaches target_ratio of final performance.
    For MSE (lower is better), we interpret "reaching" as:
        value <= final_value / target_ratio
    """
    final_val = curve[-1]
    target = final_val / target_ratio  # since MSE, lower is better

    for i, val in enumerate(curve):
        if val <= target:
            return i + 1
    return len(curve)


# Lấy curves từ notebook (đã được tính MSE thống nhất)
rounds_axis        = np.array(history_fullfedavg["round"])
fedavg_train_curve = np.array(history_fullfedavg["train_mse_mean"])
fedavg_test_curve  = np.array(history_fullfedavg["test_mse_mean"])

print(f"Using {len(rounds_axis)} rounds of FedAvg & Centralized curves.\n")

# Calculate convergence metrics
convergence_targets = [0.90, 0.95, 0.99]
fedavg_convergence = []
central_convergence = []

print("Convergence Analysis:")
print("-" * 50)
for target in convergence_targets:
    fedavg_round = find_convergence_round(fedavg_test_curve, target)
    central_round = find_convergence_round(central_test_curve, target)
    fedavg_convergence.append(fedavg_round)
    central_convergence.append(central_round)

    print(f"Rounds to {int(target*100)}% of final performance:")
    print(f"  FedAvg:      {fedavg_round:3d} rounds")
    print(f"  Centralized: {central_round:3d} rounds")
    speedup = (central_round - fedavg_round) / central_round * 100
    print(f"  Speedup:     {speedup:+.1f}%\n")

# ========================================================================
# 2. FIGURE 1: Learning Curves Comparison (IEEE-style)
# ========================================================================

# Combined training & test curves (single-column, 4 lines)
fig_train_test, ax_tt = plt.subplots(figsize=(3.5, 3.0))
ax_tt.plot(rounds_axis, fedavg_train_curve,
           linestyle='-', marker='o', markevery=10,
           markersize=3, label='FedAvg train')
ax_tt.plot(rounds_axis, fedavg_test_curve,
           linestyle='--', marker='^', markevery=10,
           markersize=3, label='FedAvg test')
ax_tt.plot(rounds_axis, central_train_curve,
           linestyle='-.', marker='s', markevery=10,
           markersize=3, label='Centralized train')
ax_tt.plot(rounds_axis, central_test_curve,
           linestyle=':', marker='D', markevery=10,
           markersize=3, label='Centralized test')
ax_tt.set_xlabel('Round / Epoch')
ax_tt.set_ylabel('MSE')
ax_tt.set_title('Training & test performance')
ax_tt.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax_tt.legend(loc='upper right', ncol=1, frameon=False)
save_figure(fig_train_test, "fig1_learning_train_test.pdf")
plt.show()


# 3. Convergence comparison with markers (on test curves)
fig_conv_line, ax3 = plt.subplots(figsize=(3.5, 2.8))
ax3.plot(rounds_axis, fedavg_test_curve, label='FedAvg',
         color='tab:blue', linewidth=2.5)
ax3.plot(rounds_axis, central_test_curve, label='Centralized',
         color='tab:orange', linewidth=2.5)

colors_markers = ['darkblue', 'blue', 'lightblue']
for i, target in enumerate(convergence_targets):
    if fedavg_convergence[i] < len(rounds_axis):
        ax3.axvline(x=fedavg_convergence[i], color=colors_markers[i],
                    linestyle=':', alpha=0.6, linewidth=1.5)
        ax3.text(fedavg_convergence[i], ax3.get_ylim()[1]*0.9,
                 f'{int(target*100)}%', rotation=90, va='top', ha='right',
                 fontsize=8, color=colors_markers[i])

ax3.set_xlabel('Round/Epoch', fontsize=11)
ax3.set_ylabel('Test MSE', fontsize=11)
ax3.set_title('Convergence speed (FedAvg markers)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=1, fontsize=10, frameon=False)
ax3.grid(True, ls=':', alpha=0.4)
save_figure(fig_conv_line, "fig1c_convergence_curve.pdf")
plt.show()

# 4. Rounds to convergence (bar chart)
x = np.arange(len(convergence_targets))
width = 0.35

fig_conv_bar, ax4 = plt.subplots(figsize=(3.5, 2.8))
bars1 = ax4.bar(x - width/2, fedavg_convergence, width,
                label='FedAvg', color='#1f77b4', alpha=0.85)
bars2 = ax4.bar(x + width/2, central_convergence, width,
                label='Centralized', color='#ff7f0e', alpha=0.85)

ax4.set_xlabel('Convergence Target', fontsize=11)
ax4.set_ylabel('Rounds Required', fontsize=11)
ax4.set_title('Rounds to convergence', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{int(t*100)}%' for t in convergence_targets])
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=1, fontsize=10, frameon=False)
ax4.grid(True, axis='y', ls=':', alpha=0.4)
save_figure(fig_conv_bar, "fig1d_convergence_rounds.pdf")
plt.show()

# ========================================================================
# 3. FIGURE 2: Final Performance Metrics (IEEE-style, colored)
# ========================================================================

# Chuẩn bị số liệu
final_fedavg_train   = fedavg_train_curve[-1]
final_fedavg_test    = fedavg_test_curve[-1]
final_central_train  = central_train_curve[-1]
final_central_test   = central_test_curve[-1]

methods   = ['FedAvg', 'Centralized']
train_mses = [final_fedavg_train,  final_central_train]
test_mses  = [final_fedavg_test,   final_central_test]

# AUC trên test curves
fedavg_auc   = np.trapz(fedavg_test_curve, rounds_axis)
central_auc  = np.trapz(central_test_curve, rounds_axis)
aucs         = [fedavg_auc, central_auc]

# Generalization gap
fedavg_gap   = final_fedavg_test  - final_fedavg_train
central_gap  = final_central_test - final_central_train
gaps         = [fedavg_gap, central_gap]

# Một palette màu mềm, hiện đại
# Updated color palette (colorblind-friendly / clear contrast)
color_fed   = '#1f77b4'  # blue (FedAvg)
color_cent  = '#ff7f0e'  # orange (Centralized)
color_train = '#66c2a5'  # teal (Train bars)
color_test  = '#fc8d62'  # coral (Test bars)

# ---------------- (a) Final MSE ----------------
fig_final_mse, ax1 = plt.subplots(figsize=(3.5, 2.8))
x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, train_mses, width,
        label='Train', color=color_train,
        alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.bar(x + width/2, test_mses, width,
        label='Test',  color=color_test,
        alpha=0.9, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Final MSE')
ax1.set_title('Final performance')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax1.legend(loc='upper right', ncol=1, frameon=False)
save_figure(fig_final_mse, "fig2a_final_mse.pdf")
plt.show()

# ---------------- (b) AUC comparison ----------------
fig_auc, ax2 = plt.subplots(figsize=(3.5, 2.8))
ax2.bar(methods, aucs,
        color=[color_fed, color_cent],
        alpha=0.9)
ax2.set_ylabel('AUC on test MSE\n(lower is better)')
ax2.set_title('Learning efficiency (AUC)')
ax2.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
improvement_auc = (central_auc - fedavg_auc) / central_auc * 100
save_figure(fig_auc, "fig2b_auc.pdf")
plt.show()

# ---------------- (c) Overfitting / Generalization gap ----------------
fig_gap, ax3 = plt.subplots(figsize=(3.5, 2.8))
colors_gap = [color_fed, color_cent]
ax3.bar(methods, gaps,
        color=colors_gap, alpha=0.9)
ax3.set_ylabel('Generalization gap\n(Test MSE - Train MSE)')
ax3.set_title('Overfitting analysis')
ax3.axhline(y=0.0, color='black', linestyle='-', linewidth=0.7)
ax3.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
save_figure(fig_gap, "fig2c_overfitting_gap.pdf")
plt.show()


# ========================================================================
# 4. Summary Statistics
# ========================================================================

print("\n" + "=" * 60)
print("SUMMARY: FL vs Centralized FNN")
print("=" * 60)
print("Final Test MSE:")
print(f"  FedAvg:      {final_fedavg_test:.6f}")
print(f"  Centralized: {final_central_test:.6f}")
improvement = (final_central_test - final_fedavg_test) / final_central_test * 100
print(f"  Improvement: {improvement:+.2f}%")

print("\nArea Under Curve:")
print(f"  FedAvg:      {fedavg_auc:.2f}")
print(f"  Centralized: {central_auc:.2f}")
print(f"  Improvement: {improvement_auc:+.2f}%")

print("\nGeneralization Gap (Test - Train):")
print(f"  FedAvg:      {fedavg_gap:.6f}")
print(f"  Centralized: {central_gap:.6f}")

print(f"\nBest approach: {'FedAvg' if final_fedavg_test < final_central_test else 'Centralized'}")
print("=" * 60)

# ========================================================================
# 5. Per-IBR Performance: Centralized vs FL
# ========================================================================

print("\n===== Per-IBR Performance Analysis: Centralized vs FedAvg =====\n")

# Check if central_model exists (from notebook)
if 'central_model' not in globals():
    print("❌ ERROR: 'central_model' not found!")
    print("   Please make sure you have run the training notebook first.")
    raise RuntimeError("Missing 'central_model'. Run training notebook first.")

# Calculate Per-Client Performance
ibr_names = []
mse_fedavg_per_client = []
mse_central_per_client = []

all_preds_fedavg = []
all_preds_central = []
all_targets = []

# Prepare models for evaluation
fedavg_model_eval = FullModel(
    Trunk(in_dim=input_dim, hidden_dim=HIDDEN_GFLI).to(device),
    Head(hidden_dim=HIDDEN_GFLI, out_dim=output_dim).to(device)
).to(device)
fedavg_model_eval.eval()
central_model = central_model.to(device).eval()

# Load global FedAvg state
set_model_state_dict(fedavg_model_eval, fullfedavg_model_state)

print("Evaluating on all IBR test sets...")

for test_set in test_sets_gfli:
    name = test_set["name"]
    # Ví dụ: 'gfli1_test_impedance_dataset' -> 'IBR1'
    name = name.replace('_test_impedance_dataset', '')
    name = name.replace('_impedance_dataset', '')
    name = name.replace('gfli', 'IBR')
    ibr_names.append(name)

    loader = DataLoader(
        TensorDataset(test_set["X"], test_set["Y"]),
        batch_size=512, shuffle=False
    )

    client_preds_fed = []
    client_preds_cent = []
    client_targets = []

    total_loss_fed = 0.0
    total_loss_cent = 0.0
    total_samples = 0

    crit = nn.MSELoss()  # mean per element

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # FedAvg prediction
            yp_fed = fedavg_model_eval(xb)
            loss_fed = crit(yp_fed, yb)  # mean
            total_loss_fed += loss_fed.item() * xb.size(0)

            # Centralized prediction
            yp_cent = central_model(xb)
            loss_cent = crit(yp_cent, yb)
            total_loss_cent += loss_cent.item() * xb.size(0)

            # Store preds/targets for later error analysis
            client_preds_fed.append(yp_fed.cpu().numpy())
            client_preds_cent.append(yp_cent.cpu().numpy())
            client_targets.append(yb.cpu().numpy())

            total_samples += xb.size(0)

    mse_fedavg_per_client.append(total_loss_fed / total_samples)
    mse_central_per_client.append(total_loss_cent / total_samples)

    all_preds_fedavg.append(np.concatenate(client_preds_fed))
    all_preds_central.append(np.concatenate(client_preds_cent))
    all_targets.append(np.concatenate(client_targets))

# FIGURE 3: Per-IBR Test Performance (colored)
fig, ax = plt.subplots(figsize=(7.0, 3.0))

x = np.arange(len(ibr_names))
width = 0.35

color_cent_bar = '#DD8452'   # orange
color_fed_bar  = '#4C72B0'   # blue

rects1 = ax.bar(x - width/2, mse_central_per_client, width,
                label='Centralized',
                color=color_cent_bar, alpha=0.85,
                edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, mse_fedavg_per_client, width,
                label='FedAvg',
                color=color_fed_bar, alpha=0.85,
                edgecolor='black', linewidth=0.5)

ax.set_ylabel('Test MSE')
ax.set_xlabel('Client (IBR)')
ax.set_title('Per-IBR Test Performance')
ax.set_xticks(x)
ax.set_xticklabels(ibr_names, rotation=0)
ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax.legend(loc='upper center', bbox_to_anchor=(0.60, 1.02), ncol=1, frameon=False)

save_figure(fig, "fig3_per_ibr_bar_colored.pdf")
plt.show()



# ========================================================================
# 6. Error Distribution (System-wide CDF)
# ========================================================================

print("\n===== Error Distribution Analysis (System-wide) =====\n")

# Flatten all predictions and targets
flat_targets = np.concatenate(all_targets)        # (N_total, D)
flat_preds_fed = np.concatenate(all_preds_fedavg) # (N_total, D)
flat_preds_cent = np.concatenate(all_preds_central)

# Calculate sample-wise MAPE (average over dimensions)
epsilon = 1e-6
ape_fed = np.mean(np.abs(flat_preds_fed - flat_targets) /
                  (np.abs(flat_targets) + epsilon), axis=1) * 100
ape_cent = np.mean(np.abs(flat_preds_cent - flat_targets) /
                   (np.abs(flat_targets) + epsilon), axis=1) * 100

# Plot Histogram and CDF
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # single-column

sorted_ape_fed  = np.sort(ape_fed)
y_fed           = np.arange(1, len(sorted_ape_fed) + 1) / len(sorted_ape_fed)
sorted_ape_cent = np.sort(ape_cent)
y_cent          = np.arange(1, len(sorted_ape_cent) + 1) / len(sorted_ape_cent)

ax.plot(sorted_ape_fed,  y_fed,
        linestyle='-',  linewidth=1.2, label='FedAvg')
ax.plot(sorted_ape_cent, y_cent,
        linestyle='--', linewidth=1.2, label='Centralized')

limit_98 = max(np.percentile(ape_fed, 98), np.percentile(ape_cent, 98))
ax.set_xlim(0, limit_98)

ax.set_xlabel('MAPE (%)')
ax.set_ylabel('CDF')
ax.set_title('System-wide Error CDF')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax.legend(loc='lower right', frameon=False)

save_figure(fig, "fig4_cdf_error.pdf")
plt.show()


print("\n" + "=" * 60)
print("FINAL SUMMARY: Domain-Specific Metrics")
print("=" * 60)
print("Mean MAPE (System-wide):")
print(f"  FedAvg:      {np.mean(ape_fed):.2f}%")
print(f"  Centralized: {np.mean(ape_cent):.2f}%")

print("\n90th Percentile Error (lower is better):")
print(f"  FedAvg:      {np.percentile(ape_fed, 90):.2f}%")
print(f"  Centralized: {np.percentile(ape_cent, 90):.2f}%")
print("=" * 60)
