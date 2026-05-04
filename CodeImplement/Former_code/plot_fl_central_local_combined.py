"""
Combined FL vs centralized vs local-only figures.

Assumes the following objects already exist in the running session
(from the training notebooks/scripts):
- history_fullfedavg, fullfedavg_model_state
- central_model, central_train_curve, central_test_curve
- clients_gfli, test_sets_gfli
- HIDDEN_GFLI, ROUNDS_FL, BATCH_SIZE_FL, LR_INIT_FL
- Trunk, Head, FullModel, set_model_state_dict, device, input_dim, output_dim

This script recomputes local-only models (per client), evaluates all
three approaches, and produces consolidated figures:
- Per-IBR bar chart with 3 columns (Centralized, FedAvg, Local-only)
- Per-client MSE distribution (boxplot, 3 groups)
- Final metrics panel (final MSE, AUC, generalization gap)
- Fairness scatter (normalized per-client MSE for 3 models)
- Error CDF with 3 curves (FedAvg, Centralized, Local-only)
- TRUE vs predicted admittance Bode (Re/Im) for all 3 models
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.lines import Line2D

# Matplotlib style (IEEE-ish)
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

COLOR_FED = "#1f77b4"     # tableau blue
COLOR_CENT = "#ff7f0e"    # tableau orange
COLOR_LOCAL = "#2ca02c"   # tableau green


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")


def _check_required_globals():
    required = [
        "history_fullfedavg",
        "fullfedavg_model_state",
        "central_model",
        "central_train_curve",
        "central_test_curve",
        "clients_gfli",
        "test_sets_gfli",
        "HIDDEN_GFLI",
        "ROUNDS_FL",
        "BATCH_SIZE_FL",
        "LR_INIT_FL",
        "Trunk",
        "Head",
        "FullModel",
        "set_model_state_dict",
        "device",
        "input_dim",
        "output_dim",
        "X_scaler_gfli",
        "Y_scaler_gfli",
    ]
    missing = [k for k in required if k not in globals()]
    if missing:
        raise RuntimeError(f"Missing globals: {missing}. Run training notebook first.")


def build_freq_bins_from_tests(test_sets, x_scaler, nbins):
    """Use the test sets to build log-spaced frequency bins."""
    f_list = []
    for ds in test_sets:
        X_phys = x_scaler.inverse_transform(ds["X"].cpu().numpy())
        f_list.append(X_phys[:, 3])
    f_all = np.concatenate(f_list)
    f_all = f_all[f_all > 0]  # keep positive frequencies for log-scale
    if f_all.size == 0:
        raise ValueError("No positive frequency values found to build bins.")
    f_min, f_max = f_all.min(), f_all.max()
    edges = np.logspace(np.log10(f_min), np.log10(f_max), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def reduce_curve(f, y, edges):
    """Median-reduce a curve into frequency bins."""
    f = np.asarray(f)
    y = np.asarray(y)
    out = np.zeros(len(edges) - 1, dtype=float)
    for i in range(len(edges) - 1):
        mask = (f >= edges[i]) & (f < edges[i + 1])
        if not np.any(mask):
            out[i] = np.nan
        else:
            out[i] = np.median(y[mask])
    return out


components = [
    ("Ydd", 0, 1),
    ("Ydq", 2, 3),
    ("Yqd", 4, 5),
    ("Yqq", 6, 7),
]


def _select_owners(X_phys, n_owners_per_test=1, min_points_per_owner=20):
    """
    Pick up to n owners (unique V,P,Q) that have enough samples.
    Preserves first-seen order to keep plots interpretable.
    """
    V, P, Q = X_phys[:, 0], X_phys[:, 1], X_phys[:, 2]
    seen = {}
    for idx, triple in enumerate(zip(V, P, Q)):
        if triple not in seen:
            seen[triple] = []
        seen[triple].append(idx)

    owners = []
    for (v, p, q), idxs in seen.items():
        idxs_arr = np.array(idxs)
        if idxs_arr.size < min_points_per_owner:
            continue
        owners.append((v, p, q, idxs_arr))
        if len(owners) >= n_owners_per_test:
            break
    return owners


def _slugify(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def plot_true_vs_pred_three_models(
    test_sets,
    x_scaler,
    y_scaler,
    fedavg_model,
    central_model,
    local_models,
    nbins=None,
    z_base=95.2,
    n_owners_per_test=1,
    min_points_per_owner=20,
    fig_size=(6.2, 5.0),
):
    """
    Plot TRUE vs predicted (Centralized, FedAvg, Local-only) Bode Re/Im curves.
    """
    if len(local_models) != len(test_sets):
        raise ValueError("local_models must align with test_sets order.")

    nbins = nbins or globals().get("NBINS", 25)
    edges, centers = build_freq_bins_from_tests(test_sets, x_scaler, nbins)

    legend_proxies = [
        Line2D([0], [0], color="black", lw=2.4, ls="-", label="TRUE"),
        Line2D([0], [0], color=COLOR_CENT, lw=2.0, ls="-.", label="Centralized"),
        Line2D([0], [0], color=COLOR_FED, lw=2.0, ls="--", label="FedAvg"),
        Line2D([0], [0], color=COLOR_LOCAL, lw=2.0, ls=":", label="Local-only"),
    ]

    for idx, ds in enumerate(test_sets):
        test_name = ds["name"]
        X_scaled = ds["X"].cpu().numpy()
        Y_scaled = ds["Y"].cpu().numpy()

        X_phys = x_scaler.inverse_transform(X_scaled)
        Y_true = y_scaler.inverse_transform(Y_scaled) / z_base

        inputs = ds["X"].to(device)
        with torch.no_grad():
            Y_pred_fed = fedavg_model(inputs).cpu().numpy()
            Y_pred_cent = central_model(inputs).cpu().numpy()
            local_model = local_models[idx].to(device).eval()
            Y_pred_local = local_model(inputs).cpu().numpy()

        Y_fed = y_scaler.inverse_transform(Y_pred_fed) / z_base
        Y_cent = y_scaler.inverse_transform(Y_pred_cent) / z_base
        Y_loc = y_scaler.inverse_transform(Y_pred_local) / z_base

        owners_selected = _select_owners(
            X_phys, n_owners_per_test=n_owners_per_test, min_points_per_owner=min_points_per_owner
        )
        if not owners_selected:
            print(f"[BODE] {test_name}: no owner with enough points to plot.")
            continue

        fig_re, axes_re = plt.subplots(
            2, 2, figsize=fig_size, gridspec_kw={"hspace": 0.55, "wspace": 0.35}
        )
        fig_im, axes_im = plt.subplots(
            2, 2, figsize=fig_size, gridspec_kw={"hspace": 0.55, "wspace": 0.35}
        )
        axes_re = axes_re.ravel()
        axes_im = axes_im.ravel()

        owner_labels = []
        for (v, p, q, idxs_owner) in owners_selected:
            f_owner = X_phys[idxs_owner, 3]
            Yt = Y_true[idxs_owner, :]
            Yf = Y_fed[idxs_owner, :]
            Yc = Y_cent[idxs_owner, :]
            Yl = Y_loc[idxs_owner, :]
            owner_labels.append(f"V={v:.0f}, P={p:.0f}, Q={q:.0f}")

            for comp_i, (name, re_i, im_i) in enumerate(components):
                # Real part
                curve_re_true = reduce_curve(f_owner, Yt[:, re_i], edges)
                curve_re_fed = reduce_curve(f_owner, Yf[:, re_i], edges)
                curve_re_cent = reduce_curve(f_owner, Yc[:, re_i], edges)
                curve_re_loc = reduce_curve(f_owner, Yl[:, re_i], edges)

                axr = axes_re[comp_i]
                axr.semilogx(centers, curve_re_true, color="black", linewidth=2.2, alpha=0.9)
                axr.semilogx(centers, curve_re_cent, color=COLOR_CENT, linewidth=1.8, alpha=0.9, linestyle="-.")
                axr.semilogx(centers, curve_re_fed, color=COLOR_FED, linewidth=1.8, alpha=0.9, linestyle="--")
                axr.semilogx(centers, curve_re_loc, color=COLOR_LOCAL, linewidth=1.8, alpha=0.9, linestyle=":")
                axr.set_title(f"Re({name})")
                axr.grid(True, which="both", ls=":", alpha=0.45)

                # Imag part
                curve_im_true = reduce_curve(f_owner, Yt[:, im_i], edges)
                curve_im_fed = reduce_curve(f_owner, Yf[:, im_i], edges)
                curve_im_cent = reduce_curve(f_owner, Yc[:, im_i], edges)
                curve_im_loc = reduce_curve(f_owner, Yl[:, im_i], edges)

                axi = axes_im[comp_i]
                axi.semilogx(centers, curve_im_true, color="black", linewidth=2.2, alpha=0.9)
                axi.semilogx(centers, curve_im_cent, color=COLOR_CENT, linewidth=1.8, alpha=0.9, linestyle="-.")
                axi.semilogx(centers, curve_im_fed, color=COLOR_FED, linewidth=1.8, alpha=0.9, linestyle="--")
                axi.semilogx(centers, curve_im_loc, color=COLOR_LOCAL, linewidth=1.8, alpha=0.9, linestyle=":")
                axi.set_title(f"Im({name})")
                axi.grid(True, which="both", ls=":", alpha=0.45)

        for ax in axes_re[2:]:
            ax.set_xlabel("Frequency")
        for ax in axes_im[2:]:
            ax.set_xlabel("Frequency")

        # Common Y label in SI units (Siemens)
        fig_re.text(0.01, 0.5, "Admittance (S)", va="center", rotation="vertical")
        fig_im.text(0.01, 0.5, "Admittance (S)", va="center", rotation="vertical")

        legend_loc = {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1.02),
            "ncol": 4,
            "frameon": False,
            "fontsize": 6,
            "columnspacing": 0.8,
            "handletextpad": 0.4,
            "borderaxespad": 0.2,
            "labelspacing": 0.3,
        }
        fig_re.legend(handles=legend_proxies, **legend_loc)
        fig_im.legend(handles=legend_proxies, **legend_loc)

        tag = _slugify(test_name)
        fname_re = f"fig_combined_bode_re_{tag}.pdf"
        fname_im = f"fig_combined_bode_im_{tag}.pdf"
        fig_re.savefig(fname_re, bbox_inches="tight")
        fig_re.savefig(fname_re.replace(".pdf", ".svg"), bbox_inches="tight")
        fig_im.savefig(fname_im, bbox_inches="tight")
        fig_im.savefig(fname_im.replace(".pdf", ".svg"), bbox_inches="tight")
        print(f"[BODE] Saved TRUE vs predicted curves for {test_name}.")


def train_local_only_model(client_data, test_data,
                           hidden_dim, epochs, batch_size, lr):
    """Train a local-only model and return model plus train/test curves."""
    trunk = Trunk(in_dim=input_dim, hidden_dim=hidden_dim).to(device)
    head = Head(hidden_dim=hidden_dim, out_dim=output_dim).to(device)
    model = FullModel(trunk, head).to(device)

    loader = DataLoader(client_data["dataset"], batch_size=batch_size, shuffle=True)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist_tr, hist_te = [], []

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
            total_n += xb.size(0)
        train_mse = total_loss / total_n if total_n > 0 else float("nan")
        hist_tr.append(train_mse)

        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(
            TensorDataset(test_data["X"], test_data["Y"]),
            batch_size=batch_size, shuffle=False,
        )
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = crit(y_pred, yb)
                total_loss += loss.item() * xb.size(0)
                total_n += xb.size(0)
        test_mse = total_loss / total_n if total_n > 0 else float("nan")
        hist_te.append(test_mse)

    return model, np.array(hist_tr), np.array(hist_te)


def eval_model_on_test(model, test_data, batch_size=512):
    """Return MSE, predictions, and targets for a given model on a test set."""
    model.eval()
    loader = DataLoader(
        TensorDataset(test_data["X"], test_data["Y"]),
        batch_size=batch_size, shuffle=False,
    )
    crit = nn.MSELoss()
    total_loss, total_n = 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = crit(y_pred, yb)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
            preds.append(y_pred.cpu().numpy())
            targets.append(yb.cpu().numpy())
    mse = total_loss / total_n if total_n > 0 else float("nan")
    return mse, np.concatenate(preds), np.concatenate(targets)


def main():
    _check_required_globals()

    # FedAvg model for evaluation
    fedavg_model = FullModel(
        Trunk(in_dim=input_dim, hidden_dim=HIDDEN_GFLI).to(device),
        Head(hidden_dim=HIDDEN_GFLI, out_dim=output_dim).to(device),
    ).to(device)
    set_model_state_dict(fedavg_model, fullfedavg_model_state)
    fedavg_model.eval()

    central = central_model.to(device).eval()

    # Storage
    client_names = []
    client_short = []
    mse_fedavg_per_client = []
    mse_central_per_client = []
    mse_local_per_client = []

    preds_fed_all = []
    preds_cent_all = []
    preds_local_all = []
    targets_all = []

    local_train_curves = []
    local_test_curves = []
    local_models = []

    print("Training/evaluating local-only models and collecting metrics...")
    for idx, client in enumerate(clients_gfli):
        test_set = test_sets_gfli[idx]
        name = client["name"]
        short = name.replace("_impedance_dataset", "").replace("_test", "")
        short = short.replace("gfli", "IBR")
        client_names.append(name)
        client_short.append(short)

        # Local training + eval
        local_model, hist_tr, hist_te = train_local_only_model(
            client, test_set,
            hidden_dim=HIDDEN_GFLI,
            epochs=ROUNDS_FL,
            batch_size=BATCH_SIZE_FL,
            lr=LR_INIT_FL,
        )
        local_train_curves.append(hist_tr)
        local_test_curves.append(hist_te)
        mse_local, preds_local, targets = eval_model_on_test(local_model, test_set)
        mse_local_per_client.append(mse_local)
        preds_local_all.append(preds_local)
        targets_all.append(targets)
        local_models.append(local_model.eval())

        # FedAvg eval
        mse_fed, preds_fed, _ = eval_model_on_test(fedavg_model, test_set)
        mse_fedavg_per_client.append(mse_fed)
        preds_fed_all.append(preds_fed)

        # Centralized eval
        mse_cent, preds_cent, _ = eval_model_on_test(central, test_set)
        mse_central_per_client.append(mse_cent)
        preds_cent_all.append(preds_cent)

        print(f"  {short}: Local={mse_local:.4e} | FedAvg={mse_fed:.4e} | Central={mse_cent:.4e}")

    # ---------------- Per-IBR bar chart (3 columns) ----------------
    x = np.arange(len(client_short))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.bar(x - width, mse_central_per_client, width, label="Centralized",
           color=COLOR_CENT, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x, mse_fedavg_per_client, width, label="FedAvg",
           color=COLOR_FED, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, mse_local_per_client, width, label="Local-only",
           color=COLOR_LOCAL, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test MSE")
    ax.set_xlabel("Client (IBR)")
    ax.set_title("Per-IBR Test Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(client_short, rotation=0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), ncol=3, frameon=False)
    save_figure(fig, "fig_combined_per_ibr_bar.pdf")

    # ---------------- Boxplot: per-client MSE distribution (3 groups) ----------------
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    data_box = [mse_fedavg_per_client, mse_central_per_client, mse_local_per_client]
    labels = ["FedAvg", "Centralized", "Local-only"]
    box = ax.boxplot(
        data_box,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
    )
    colors = [COLOR_FED, COLOR_CENT, COLOR_LOCAL]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_ylabel("Test MSE")
    ax.set_title("Per-client MSE distribution")

    # Log scale to separate the tight FedAvg/Centralized boxes from Local-only
    all_vals = np.concatenate(data_box)
    ymin = max(1e-8, np.min(all_vals) * 0.5)
    ymax = np.max(mse_local_per_client) * 1.5
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    save_figure(fig, "fig_combined_boxplot_mse.pdf")

    # ---------------- Final metrics panel ----------------
    rounds_axis = np.array(history_fullfedavg["round"])
    fed_train_curve = np.array(history_fullfedavg["train_mse_mean"])
    fed_test_curve = np.array(history_fullfedavg["test_mse_mean"])
    central_train_curve_arr = np.array(central_train_curve)
    central_test_curve_arr = np.array(central_test_curve)

    local_test_mat = np.vstack(local_test_curves)
    local_train_mat = np.vstack(local_train_curves)
    mean_local_test = np.mean(local_test_mat, axis=0)
    mean_local_train = np.mean(local_train_mat, axis=0)
    local_rounds = np.arange(1, mean_local_test.shape[0] + 1)

    # ---------------- Learning curves (train/test, 3 models) ----------------
    fig_tt, ax_tt = plt.subplots(figsize=(3.5, 2.0))
    ax_tt.plot(rounds_axis, fed_train_curve, linestyle='-', marker='o', markevery=max(len(rounds_axis)//12, 1),
               markersize=3, label='FedAvg train', color=COLOR_FED)
    ax_tt.plot(rounds_axis, fed_test_curve, linestyle='--', marker='^', markevery=max(len(rounds_axis)//12, 1),
               markersize=3, label='FedAvg test', color=COLOR_FED)
    ax_tt.plot(rounds_axis, central_train_curve_arr, linestyle='-.', marker='s', markevery=max(len(rounds_axis)//12, 1),
               markersize=3, label='Centralized train', color=COLOR_CENT)
    ax_tt.plot(rounds_axis, central_test_curve_arr, linestyle=':', marker='D', markevery=max(len(rounds_axis)//12, 1),
               markersize=3, label='Centralized test', color=COLOR_CENT)
    ax_tt.plot(local_rounds, mean_local_train, linestyle='-', marker='v', markevery=max(len(local_rounds)//12, 1),
               markersize=3, label='Local-only train', color=COLOR_LOCAL)
    ax_tt.plot(local_rounds, mean_local_test, linestyle='--', marker='x', markevery=max(len(local_rounds)//12, 1),
               markersize=3, label='Local-only test', color=COLOR_LOCAL)
    ax_tt.set_xlabel("Round / Epoch")
    ax_tt.set_ylabel("MSE")
    ax_tt.set_title("Training & test performance (3 models)")
    ax_tt.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax_tt.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.5)
    save_figure(fig_tt, "fig_combined_learning_curves.pdf")

    final_mse = [
        fed_test_curve[-1],
        central_test_curve_arr[-1],
        mean_local_test[-1],
    ]
    auc_vals = [
        np.trapz(fed_test_curve, rounds_axis),
        np.trapz(central_test_curve_arr, rounds_axis),
        np.trapz(mean_local_test, local_rounds),
    ]
    gaps = [
        fed_test_curve[-1] - fed_train_curve[-1],
        central_test_curve_arr[-1] - central_train_curve_arr[-1],
        mean_local_test[-1] - mean_local_train[-1],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2))
    # (a) Final MSE
    axes[0].bar(labels, final_mse, color=[COLOR_FED, COLOR_CENT, COLOR_LOCAL], alpha=0.9, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Final Test MSE")
    axes[0].set_title("(a) Final performance")
    axes[0].grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    # (b) AUC
    axes[1].bar(labels, auc_vals, color=[COLOR_FED, COLOR_CENT, COLOR_LOCAL], alpha=0.9, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("AUC on test MSE\n(lower is better)")
    axes[1].set_title("(b) Learning efficiency")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    # (c) Generalization gap
    axes[2].bar(labels, gaps, color=[COLOR_FED, COLOR_CENT, COLOR_LOCAL], alpha=0.9, edgecolor="black", linewidth=0.5)
    axes[2].axhline(y=0.0, color="black", linestyle="-", linewidth=0.7)
    axes[2].set_ylabel("Gap (Test - Train)")
    axes[2].set_title("(c) Overfitting gap")
    axes[2].grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    save_figure(fig, "fig_combined_final_metrics.pdf")

    # ---------------- Fairness scatter (normalized) ----------------
    max_ref = max(max(mse_local_per_client), max(mse_fedavg_per_client), max(mse_central_per_client))
    norm_local = np.array(mse_local_per_client) / max_ref
    norm_fed = np.array(mse_fedavg_per_client) / max_ref
    norm_cent = np.array(mse_central_per_client) / max_ref
    xpos = np.arange(len(client_short))

    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.scatter(xpos - 0.1, norm_cent, s=35, c=COLOR_CENT, alpha=0.9, label="Centralized",
               marker="s", edgecolors="black", linewidths=0.5)
    ax.scatter(xpos, norm_fed, s=35, c=COLOR_FED, alpha=0.9, label="FedAvg",
               marker="^", edgecolors="black", linewidths=0.5)
    ax.scatter(xpos + 0.1, norm_local, s=35, c=COLOR_LOCAL, alpha=0.9, label="Local-only",
               marker="o", edgecolors="black", linewidths=0.5)
    for i in range(len(xpos)):
        ax.plot([xpos[i] - 0.1, xpos[i], xpos[i] + 0.1],
                [norm_cent[i], norm_fed[i], norm_local[i]],
                color="gray", alpha=0.4, linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(client_short, rotation=30, ha="right")
    ax.set_ylabel("Normalized test MSE")
    ax.set_xlabel("Client")
    ax.set_title("Normalized per-client performance (3 models)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "fig_combined_fairness_scatter.pdf")

    # ---------------- Error CDF (MAPE) ----------------
    epsilon = 1e-6
    flat_targets = np.concatenate(targets_all)
    flat_fed = np.concatenate(preds_fed_all)
    flat_cent = np.concatenate(preds_cent_all)
    flat_local = np.concatenate(preds_local_all)

    ape_fed = np.mean(np.abs(flat_fed - flat_targets) / (np.abs(flat_targets) + epsilon), axis=1) * 100
    ape_cent = np.mean(np.abs(flat_cent - flat_targets) / (np.abs(flat_targets) + epsilon), axis=1) * 100
    ape_local = np.mean(np.abs(flat_local - flat_targets) / (np.abs(flat_targets) + epsilon), axis=1) * 100

    def _cdf(data):
        sorted_d = np.sort(data)
        y = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        return sorted_d, y

    sorted_fed, y_fed = _cdf(ape_fed)
    sorted_cent, y_cent = _cdf(ape_cent)
    sorted_local, y_local = _cdf(ape_local)

    limit_98 = max(np.percentile(ape_fed, 98), np.percentile(ape_cent, 98), np.percentile(ape_local, 98))

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(sorted_cent, y_cent, linestyle="-", linewidth=1.2, label="Centralized", color=COLOR_CENT)
    ax.plot(sorted_fed, y_fed, linestyle="--", linewidth=1.2, label="FedAvg", color=COLOR_FED)
    ax.plot(sorted_local, y_local, linestyle="-.", linewidth=1.2, label="Local-only", color=COLOR_LOCAL)
    ax.set_xlim(0, limit_98)
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("CDF")
    ax.set_title("System-wide Error CDF (3 models)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "fig_combined_cdf_error.pdf")

    # ---------------- TRUE vs predicted Bode (Re/Im, 3 models) ----------------
    if test_sets_gfli:
        plot_true_vs_pred_three_models(
            test_sets=[test_sets_gfli[0]],
            x_scaler=X_scaler_gfli,
            y_scaler=Y_scaler_gfli,
            fedavg_model=fedavg_model,
            central_model=central,
            local_models=[local_models[0]],
            fig_size=(3.5, 3.0),
        )

    print("Done. Figures saved:\n"
          "- fig_combined_per_ibr_bar.pdf/svg\n"
          "- fig_combined_boxplot_mse.pdf/svg\n"
          "- fig_combined_final_metrics.pdf/svg\n"
          "- fig_combined_fairness_scatter.pdf/svg\n"
          "- fig_combined_cdf_error.pdf/svg\n"
          "- fig_combined_bode_re_*.pdf/svg and fig_combined_bode_im_*.pdf/svg")


if __name__ == "__main__":
    main()
