"""
FL vs Local-Only Training Comparison
Demonstrates FL benefits for individual clients compared to local-only training
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ================== Matplotlib config: IEEE + màu hài hòa ==================
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

# Một palette màu mềm, đồng nhất giữa các hình
def save_figure(fig, filename):
    def _save(path):
        try:
            fig.savefig(path, bbox_inches="tight")
            return
        except PermissionError:
            stem, suffix = path.rsplit(".", 1)
            for idx in range(1, 100):
                fallback = f"{stem}_{idx}.{suffix}"
                try:
                    fig.savefig(fallback, bbox_inches="tight")
                    print(f"[WARN] Could not overwrite locked file: {path}")
                    print(f"[WARN] Saved figure instead as: {fallback}")
                    return
                except PermissionError:
                    continue
            raise

    fig.tight_layout()
    _save(filename)
    _save(filename.replace(".pdf", ".svg"))


COLOR_FED   = "#4C72B0"   # blue
COLOR_LOCAL = "#DD8452"   # orange
COLOR_POS   = "#55A868"   # green (improvement > 0)
COLOR_NEG   = "#C44E52"   # red   (improvement < 0)

# ========================================================================
# Train Local-Only Models for Each Client
# ========================================================================

print("\n===== FL vs Local-Only: Training Local Baselines =====\n")

def train_local_only_model(client_data, test_data, hidden_dim=HIDDEN_GFLI,
                           epochs=ROUNDS_FL, batch_size=BATCH_SIZE_FL, lr=LR_INIT_FL):
    """
    Train local-only model and return final test MSE plus per-epoch train/test curves.
    """
    trunk = Trunk(in_dim=input_dim, hidden_dim=hidden_dim).to(device)
    head  = Head(hidden_dim=hidden_dim, out_dim=output_dim).to(device)
    model = FullModel(trunk, head).to(device)

    loader = DataLoader(client_data["dataset"], batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    hist_tr, hist_te = [], []

    # Training
    for _ in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_pred = model(xb)
            loss = crit(y_pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)
        train_mse = total_loss / total_n if total_n > 0 else float("nan")
        hist_tr.append(train_mse)

        # evaluate on test set each epoch
        model.eval()
        test_loader = DataLoader(
            TensorDataset(test_data["X"], test_data["Y"]),
            batch_size=batch_size, shuffle=False
        )
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = crit(y_pred, yb)
                total_loss += loss.item() * xb.size(0)
                total_n    += xb.size(0)
        test_mse = total_loss / total_n if total_n > 0 else float("nan")
        hist_te.append(test_mse)

    final_mse = hist_te[-1] if hist_te else float("nan")
    return final_mse, np.array(hist_tr), np.array(hist_te)


# Train local models and collect results
print("Training local-only models for each client...")
print("-" * 50)

local_results      = {}
local_histories    = {}
fedavg_results     = {}
client_sample_sizes = {}

for i, client in enumerate(clients_gfli):
    print(f"Processing {client['name']}...")

    # Local-only model (now returns histories)
    local_mse, hist_tr, hist_te = train_local_only_model(client, test_sets_gfli[i])
    local_results[client["name"]]   = local_mse
    local_histories[client["name"]] = {"train": hist_tr, "test": hist_te}
    client_sample_sizes[client["name"]] = client["n"]

    # FedAvg global model performance on this client's test set
    trunk = Trunk(in_dim=input_dim, hidden_dim=HIDDEN_GFLI).to(device)
    head  = Head(hidden_dim=HIDDEN_GFLI, out_dim=output_dim).to(device)
    fedavg_model = FullModel(trunk, head).to(device)
    set_model_state_dict(fedavg_model, fullfedavg_model_state)

    fedavg_model.eval()
    test_loader = DataLoader(
        TensorDataset(test_sets_gfli[i]["X"], test_sets_gfli[i]["Y"]),
        batch_size=512, shuffle=False
    )
    crit = nn.MSELoss()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = fedavg_model(xb)
            loss   = crit(y_pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_n    += xb.size(0)

    fedavg_mse = total_loss / total_n
    fedavg_results[client["name"]] = fedavg_mse

    improvement = (local_mse - fedavg_mse) / local_mse * 100
    print(f"  Local: {local_mse:.4e} | FedAvg: {fedavg_mse:.4e} | Improvement: {improvement:+.1f}%")

print("-" * 50)

# ========================================================================
# FIGURE 1: Per-Client Performance Comparison (high-quality)
# ========================================================================

client_names = list(local_results.keys())
client_short_names = [
    name.replace('_impedance_dataset', '').replace('gfli', 'IBR')
    for name in client_names
]

local_mses = [local_results[name] for name in client_names]
fedavg_mses = [fedavg_results[name] for name in client_names]
improvements = [
    (local_results[name] - fedavg_results[name]) / local_results[name] * 100
    for name in client_names
]
sample_sizes = [client_sample_sizes[name] for name in client_names]

# Create each panel as a separate IEEE-style figure (single-column small figures)

# --- (a) Per-client bar chart ---
x = np.arange(len(client_short_names))
width = 0.35
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(1, 1, 1)
bars_local = ax.bar(x - width/2, local_mses, width, label='Local-only', color=COLOR_LOCAL, alpha=0.85, edgecolor='black', linewidth=0.5)
bars_fed = ax.bar(x + width/2, fedavg_mses, width, label='FedAvg', color=COLOR_FED, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Client', fontsize=9)
ax.set_ylabel('Test MSE', fontsize=9)
ax.set_title('(a) Per-client test performance', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(client_short_names, rotation=30, ha='right', fontsize=8)
ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax.legend(loc='upper left', fontsize=8, frameon=False)
save_figure(fig, "fig_local_vs_fl_per_client_a.pdf")
plt.show()

# --- (b) Per-client relative improvement (horizontal) ---
fig = plt.figure(figsize=(3.5, 2.5))
ax = fig.add_subplot(1, 1, 1)
colors_imp = [COLOR_POS if imp > 0 else COLOR_NEG for imp in improvements]
ax.barh(np.arange(len(client_short_names)), improvements, color=colors_imp, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_yticks(np.arange(len(client_short_names)))
ax.set_yticklabels(client_short_names, fontsize=8)
ax.set_xlabel('Improvement of FedAvg over Local-only (%)', fontsize=9)
ax.set_title('(b) Per-client relative improvement', fontsize=9, fontweight='bold')
ax.axvline(x=0.0, color='black', linestyle='-', linewidth=0.7)
ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
save_figure(fig, "fig_local_vs_fl_per_client_b.pdf")
plt.show()

# --- (c) Sample size vs improvement scatter ---
fig = plt.figure(figsize=(3.5, 2.5))
ax = fig.add_subplot(1, 1, 1)
sc = ax.scatter(sample_sizes, improvements, s=40, c=improvements, cmap='RdYlGn', alpha=0.85, edgecolors='black', linewidths=0.5)
for i, label in enumerate(client_short_names):
    ax.annotate(label, (sample_sizes[i], improvements[i]), fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Training sample size', fontsize=9)
ax.set_ylabel('Improvement (%)', fontsize=9)
ax.set_title('(c) Sample size vs. FL improvement', fontsize=9, fontweight='bold')
ax.axhline(y=0.0, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
fig.colorbar(sc, ax=ax, label='Improvement (%)')
save_figure(fig, "fig_local_vs_fl_per_client_c.pdf")
plt.show()

# --- (d) Boxplot: distribution of per-client MSE ---
fig = plt.figure(figsize=(3.5, 2.5))
ax = fig.add_subplot(1, 1, 1)
data_box = [fedavg_mses, local_mses]
bp = ax.boxplot(data_box, labels=['FedAvg', 'Local-only'], patch_artist=True, showmeans=True, meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=5))
bp['boxes'][0].set_facecolor(COLOR_FED)
bp['boxes'][0].set_alpha(0.45)
bp['boxes'][1].set_facecolor(COLOR_LOCAL)
bp['boxes'][1].set_alpha(0.45)
ax.set_ylabel('Test MSE', fontsize=9)
ax.set_title('(d) Distribution of per-client test MSE', fontsize=9, fontweight='bold')
ax.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
fedavg_mean, fedavg_std = np.mean(fedavg_mses), np.std(fedavg_mses)
local_mean,  local_std  = np.mean(local_mses),  np.std(local_mses)
ax.text(0.02, 0.98, f'FedAvg: μ={fedavg_mean:.3e}, σ={fedavg_std:.3e}\nLocal:  μ={local_mean:.3e}, σ={local_std:.3e}', transform=ax.transAxes, fontsize=7, va='top', bbox=dict(boxstyle='round', facecolor='#F5E6C8', alpha=0.7))
save_figure(fig, "fig_local_vs_fl_per_client_d.pdf")
plt.show()

# ========================================================================
# FIGURE 2: Fairness and Variance Analysis (high-quality)
# ========================================================================

# ---------------- (a) Normalized performance scatter ----------------
local_norm  = np.array(local_mses) / max(local_mses)
fedavg_norm = np.array(fedavg_mses) / max(local_mses)
x_pos       = np.arange(len(client_short_names))

fig_scatter, ax1 = plt.subplots(figsize=(3.5, 2.8))
ax1.scatter(
    x_pos, local_norm,
    s=35, c=COLOR_LOCAL, alpha=0.9,
    label='Local-only', marker='o', edgecolors='black', linewidths=0.5
)
ax1.scatter(
    x_pos, fedavg_norm,
    s=35, c=COLOR_FED, alpha=0.9,
    label='FedAvg', marker='^', edgecolors='black', linewidths=0.5
)

for i in range(len(x_pos)):
    ax1.plot(
        [x_pos[i], x_pos[i]],
        [local_norm[i], fedavg_norm[i]],
        color='gray', alpha=0.4, linewidth=0.8
    )

ax1.set_xticks(x_pos)
ax1.set_xticklabels(client_short_names, rotation=30, ha='right')
ax1.set_ylabel('Normalized test MSE')
ax1.set_xlabel('Client')
ax1.set_title('Normalized per-client performance')
ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax1.legend(loc='upper right', frameon=False)

save_figure(fig_scatter, "fig_local_vs_fl_fairness_a.pdf")
plt.show()

# ---------------- (b) Individual & cumulative improvements --------------
sorted_indices         = np.argsort(improvements)[::-1]
sorted_improvements    = [improvements[i] for i in sorted_indices]
sorted_names           = [client_short_names[i] for i in sorted_indices]
cumulative_improvement = np.cumsum(sorted_improvements)

bar_colors = [COLOR_POS if imp > 0 else COLOR_NEG for imp in sorted_improvements]

fig_improve, ax2 = plt.subplots(figsize=(3.5, 2.8))
ax2.bar(
    range(len(sorted_names)), sorted_improvements,
    color=bar_colors, alpha=0.85,
    edgecolor='black', linewidth=0.5,
    label='Individual'
)

ax2.plot(
    range(len(sorted_names)), cumulative_improvement,
    color='#1F4E79', marker='o', linewidth=1.5,
    markersize=4, linestyle='--', label='Cumulative'
)

ax2.set_xticks(range(len(sorted_names)))
ax2.set_xticklabels(sorted_names, rotation=30, ha='right')
ax2.set_ylabel('Improvement (%)')
ax2.set_xlabel('Client (sorted by improvement)')
ax2.set_title('Individual and cumulative improvements')
ax2.axhline(y=0.0, color='black', linestyle='-', linewidth=0.7)
ax2.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
ax2.legend(loc='upper right', frameon=False)

save_figure(fig_improve, "fig_local_vs_fl_fairness_b.pdf")
plt.show()

# ========================================================================
# Summary Statistics
# ========================================================================

print("\n" + "=" * 60)
print("SUMMARY: FL vs Local-only Training")
print("=" * 60)
print("Average Test MSE:")
print(f"  FedAvg:     {fedavg_mean:.6f} (σ={fedavg_std:.6f})")
print(f"  Local-only: {local_mean:.6f} (σ={local_std:.6f})")
overall_improvement = (local_mean - fedavg_mean) / local_mean * 100
print(f"  Overall Improvement: {overall_improvement:+.2f}%")

print("\nVariance Reduction:")
variance_reduction = (local_std - fedavg_std) / local_std * 100
print(f"  Std reduction: {variance_reduction:+.2f}%")
print(f"  {'✓ FL reduces variance across clients' if variance_reduction > 0 else '✗ FL increases variance'}")

print("\nClient-Level Analysis:")
improved_clients = sum(1 for imp in improvements if imp > 0)
print(f"  Clients improved: {improved_clients}/{len(improvements)}")
print(f"  Best improvement: {max(improvements):+.2f}% "
      f"({client_short_names[improvements.index(max(improvements))]})")
print(f"  Worst result: {min(improvements):+.2f}% "
      f"({client_short_names[improvements.index(min(improvements))]})")
print(f"  Average improvement: {np.mean(improvements):+.2f}%")

print("\nFairness (performance inequality):")
print(f"  FedAvg range: {max(fedavg_mses) - min(fedavg_mses):.6f}")
print(f"  Local range:  {max(local_mses)  - min(local_mses):.6f}")
fairness_improvement = (
    (max(local_mses) - min(local_mses))
    - (max(fedavg_mses) - min(fedavg_mses))
) / (max(local_mses) - min(local_mses)) * 100
print(f"  {'✓ FL improves fairness' if fairness_improvement > 0 else '✗ FL reduces fairness'} "
      f"({fairness_improvement:+.1f}%)")
print("=" * 60)

# ========================================================================
# Extra: Train/Test curves for IBR1 and IBR9 (high-quality)
# ========================================================================

def find_client_key(prefix):
    for name in local_histories.keys():
        if name.lower().startswith(prefix.lower()):
            return name
    return None

ibr1_key = find_client_key("gfli1")
ibr9_key = find_client_key("gfli9")

rounds = np.arange(1, ROUNDS_FL + 1)

fed_rounds = np.array(history_fullfedavg["round"])                \
             if "history_fullfedavg" in globals() else rounds
fed_train  = np.array(history_fullfedavg["train_mse_mean"])       \
             if "history_fullfedavg" in globals() else None
fed_test   = np.array(history_fullfedavg["test_mse_mean"])        \
             if "history_fullfedavg" in globals() else None

def plot_ibr_curve(ibr_key, ibr_label, filename):
    h = local_histories[ibr_key]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    ax.plot(
        rounds[:len(h["train"])], h["train"],
        label=f'Local train ({ibr_label})',
        color=COLOR_LOCAL, marker='o', markersize=3, markevery=max(len(h["train"]) // 10, 1)
    )
    ax.plot(
        rounds[:len(h["test"])], h["test"],
        label=f'Local test ({ibr_label})',
        color=COLOR_LOCAL, linestyle='--', marker='s', markersize=3,
        markevery=max(len(h["test"]) // 10, 1)
    )

    if fed_train is not None and fed_test is not None:
        ax.plot(
            fed_rounds, fed_train,
            label='FedAvg train (global mean)',
            color=COLOR_FED, linestyle='-'
        )
        ax.plot(
            fed_rounds, fed_test,
            label='FedAvg test (global mean)',
            color=COLOR_FED, linestyle='--'
        )

    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("MSE")
    ax.set_title(f"{ibr_label}: train & test curves")
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.5)

    save_figure(fig, filename)
    plt.show()

if ibr1_key:
    plot_ibr_curve(ibr1_key, "IBR1", "fig_local_vs_fl_ibr1_curves.pdf")

if ibr9_key:
    plot_ibr_curve(ibr9_key, "IBR9", "fig_local_vs_fl_ibr9_curves.pdf")
