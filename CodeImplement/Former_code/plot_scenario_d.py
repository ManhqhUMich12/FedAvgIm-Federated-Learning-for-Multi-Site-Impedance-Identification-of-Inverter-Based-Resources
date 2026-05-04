# ============================================================
# Cell 10 — Plot (mean curve across seeds) + show overfitting (train vs test)
# ============================================================

mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
    }
)

# Legend placement configuration (edit loc / bbox_to_anchor to move legends manually)
# Example: {"loc": "upper left", "bbox": (0.0, 1.0)} anchors legend at top-left.
LEGEND_COORDS = {
    "overfit_mean": {"loc": "upper right", "bbox": (0.95, 1.05)},
    "test_mean_std": {"loc": "upper right", "bbox": (0.15, 0.85)},
    "final_mse_per_seed": {"loc": "upper right", "bbox": (0.15, 0.85)},
    "auc_per_seed": {"loc": "upper right", "bbox": (0.15, 0.85)},
    "overfit_gap_per_seed": {"loc": "upper right", "bbox": (0.25, 1.0)},
    "data_efficiency": {"loc": "upper right", "bbox": (0.95, 0.9)},
    "admittance_re": {"loc": "best", "bbox": None},
    "admittance_im": {"loc": "best", "bbox": None},
}

def place_legend(ax, key, default_loc="upper right", default_bbox=None, frameon=False):
    cfg = LEGEND_COORDS.get(key, {})
    loc = cfg.get("loc", default_loc)
    bbox = cfg.get("bbox", default_bbox)
    ax.legend(loc=loc, bbox_to_anchor=bbox, frameon=cfg.get("frameon", frameon))

def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.endswith(".pdf"):
        fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")

def pad_to_same_length(list_of_curves):
    L = max(len(c) for c in list_of_curves)
    out = []
    for c in list_of_curves:
        if len(c) < L:
            c2 = np.pad(np.asarray(c, dtype=float), (0, L - len(c)), mode="edge")
        else:
            c2 = np.asarray(c, dtype=float)
        out.append(c2)
    return np.stack(out, axis=0)

fl_test_mat   = pad_to_same_length(curves["fltl_test"])
lo_test_mat   = pad_to_same_length(curves["local_test"])
fl_train_mat  = pad_to_same_length(curves["fltl_train"])
lo_train_mat  = pad_to_same_length(curves["local_train"])

fl_test_mean  = fl_test_mat.mean(axis=0)
lo_test_mean  = lo_test_mat.mean(axis=0)
fl_train_mean = fl_train_mat.mean(axis=0)
lo_train_mean = lo_train_mat.mean(axis=0)

epochs_axis = np.arange(len(fl_test_mean))

'''# Figure A: Test curves (mean across seeds)
figA, axA = plt.subplots(figsize=(4.4, 3.2))
axA.plot(epochs_axis, fl_test_mean, label="FL-TL (FedAvg backbone) — test", linestyle="--")
axA.plot(epochs_axis, lo_test_mean, label=f"Local TL ({LOCAL_TL_SOURCE_IBR.upper()} backbone) — test", linestyle="-.")
axA.set_xlabel("Fine-tuning epoch on target")
axA.set_ylabel("Test MSE on target (scaled-Y)")
axA.set_title(f"{TARGET_IBR.upper()} — FL-TL vs Local TL (mean over seeds)")
axA.grid(True, ls=":", alpha=0.5)
axA.legend(loc="upper right", frameon=False)
save_figure(figA, f"fig_tl_{TARGET_IBR}_fltl_vs_localtl_test_mean.pdf")
plt.show()'''

# Figure B: Overfitting view (train vs test)
figB, axB = plt.subplots(figsize=(3.5, 2.0))
axB.plot(epochs_axis, fl_train_mean, label="FL-TL — train", linestyle="-")
axB.plot(epochs_axis, fl_test_mean,  label="FL-TL — test",  linestyle="--")
axB.plot(epochs_axis, lo_train_mean, label="Local TL — train", linestyle="-")
axB.plot(epochs_axis, lo_test_mean,  label="Local TL — test",  linestyle="-.")
axB.set_xlabel("Fine-tuning epoch on target")
axB.set_ylabel("MSE (scaled-Y)")
axB.set_title(f"IBR1 — Overfitting check (train vs test, mean)")
axB.grid(True, ls=":", alpha=0.5)
place_legend(axB, "overfit_mean", default_loc="upper right", frameon=False)
save_figure(figB, f"fig_tl_{TARGET_IBR}_fltl_vs_localtl_overfit_mean.pdf")
plt.show()
'''
# ---- NEW: mean ± std band across seeds for TEST MSE ----
fl_test_std = fl_test_mat.std(axis=0, ddof=1) if fl_test_mat.shape[0] > 1 else np.zeros_like(fl_test_mean)
lo_test_std = lo_test_mat.std(axis=0, ddof=1) if lo_test_mat.shape[0] > 1 else np.zeros_like(lo_test_mean)

figV, axV = plt.subplots(figsize=(3.5, 2.0))

# mean lines
axV.plot(epochs_axis, fl_test_mean, label="FL-TL test (mean)", linestyle="--")
axV.plot(epochs_axis, lo_test_mean, label=f"Local TL test (mean)", linestyle="-.")

# ±1 std shaded area
axV.fill_between(epochs_axis, fl_test_mean - fl_test_std, fl_test_mean + fl_test_std, alpha=0.2, label="FL-TL ±1 std")
axV.fill_between(epochs_axis, lo_test_mean - lo_test_std, lo_test_mean + lo_test_std, alpha=0.2, label="Local TL ±1 std")

axV.set_xlabel("Fine-tuning epoch on target")
axV.set_ylabel("Test MSE on target (scaled-Y)")
axV.set_title(f"{TARGET_IBR.upper()} — Test learning curves (mean ± std over seeds)")
axV.grid(True, ls=":", alpha=0.5)
place_legend(axV, "test_mean_std", default_loc="upper right", frameon=False)
save_figure(figV, f"fig_tl_{TARGET_IBR}_fltl_vs_localtl_test_mean_std.pdf")
plt.show()'''

# ============================================================
# Cell 10C — Per-seed summary plots: Final MSE, AUC, Overfit gap
# ============================================================

seeds = np.asarray(results["seed"])

fl_final = np.asarray(results["final_mse_fltl"], dtype=float)
lo_final = np.asarray(results["final_mse_localtl"], dtype=float)

fl_auc = np.asarray(results["auc_fltl"], dtype=float)
lo_auc = np.asarray(results["auc_localtl"], dtype=float)

fl_gap = np.asarray(results["gap_fltl"], dtype=float)
lo_gap = np.asarray(results["gap_localtl"], dtype=float)

# sort by seed for nicer plots
order = np.argsort(seeds)
seeds = seeds[order]
fl_final, lo_final = fl_final[order], lo_final[order]
fl_auc, lo_auc     = fl_auc[order], lo_auc[order]
fl_gap, lo_gap     = fl_gap[order], lo_gap[order]

x = np.arange(len(seeds))
w = 0.36

def mean_std(a):
    a = np.asarray(a, dtype=float)
    if len(a) <= 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1))
'''
# -------- Figure 1: Final test MSE per seed (bar) + mean±std lines
m_fl, sd_fl = mean_std(fl_final)
m_lo, sd_lo = mean_std(lo_final)

fig1, ax1 = plt.subplots(figsize=(3.5, 2.0))
ax1.bar(x - w/2, lo_final, w, label="Local TL (final)")
ax1.bar(x + w/2, fl_final, w, label="FL-TL (final)")
ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in seeds], rotation=0)
ax1.set_xlabel("Seed")
ax1.set_ylabel("Final test MSE (scaled-Y)")
ax1.set_title(f"{TARGET_IBR.upper()} — Final test MSE per seed")

# mean±std as horizontal spans (visual cue)
ax1.axhline(m_lo, linestyle="--")
ax1.axhline(m_fl, linestyle="--")
ax1.grid(True, axis="y", ls=":", alpha=0.5)
place_legend(ax1, "final_mse_per_seed", default_loc="upper right", frameon=False)
save_figure(fig1, f"fig_tl_{TARGET_IBR}_final_mse_per_seed.pdf")
plt.show()

# -------- Figure 2: AUC per seed (bar)
m_afl, sd_afl = mean_std(fl_auc)
m_alo, sd_alo = mean_std(lo_auc)

fig2, ax2 = plt.subplots(figsize=(3.5, 2.0))
ax2.bar(x - w/2, lo_auc, w, label="Local TL (AUC)")
ax2.bar(x + w/2, fl_auc, w, label="FL-TL (AUC)")
ax2.set_xticks(x)
ax2.set_xticklabels([str(s) for s in seeds], rotation=0)
ax2.set_xlabel("Seed")
ax2.set_ylabel("AUC of test MSE curve (lower better)")
ax2.set_title(f"{TARGET_IBR.upper()} — AUC per seed")
ax2.grid(True, axis="y", ls=":", alpha=0.5)
place_legend(ax2, "auc_per_seed", default_loc="upper right", frameon=False)
save_figure(fig2, f"fig_tl_{TARGET_IBR}_auc_per_seed.pdf")
plt.show()
'''
# -------- Figure 3: Overfit gap per seed (bar)
# gap = test_final - train_final ; bigger => more overfit
m_gfl, sd_gfl = mean_std(fl_gap)
m_glo, sd_glo = mean_std(lo_gap)

fig3, ax3 = plt.subplots(figsize=(3.5, 2.0))
ax3.bar(x - w/2, lo_gap, w, label="Local TL (gap)")
ax3.bar(x + w/2, fl_gap, w, label="FL-TL (gap)")
ax3.axhline(0.0, linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels([str(s) for s in seeds], rotation=0)
ax3.set_xlabel("Seed")
ax3.set_ylabel("Overfit gap (test - train MSE)")
ax3.set_title(f"IBR1 — Overfit gap per seed")
ax3.grid(True, axis="y", ls=":", alpha=0.5)
place_legend(ax3, "overfit_gap_per_seed", default_loc="upper right", frameon=False)
save_figure(fig3, f"fig_tl_{TARGET_IBR}_overfit_gap_per_seed.pdf")
plt.show()
'''
# -------- Figure 4: Compact comparison (boxplot) for stability view
fig4, ax4 = plt.subplots(figsize=(3.5, 2.0))
ax4.boxplot(
    [lo_final, fl_final, lo_auc, fl_auc, lo_gap, fl_gap],
    labels=[
        "Local-final", "FL-final",
        "Local-AUC", "FL-AUC",
        "Local-gap", "FL-gap"
    ],
    showmeans=True
)
ax4.set_title(f"{TARGET_IBR.upper()} — Distribution over seeds (boxplots)")
ax4.grid(True, axis="y", ls=":", alpha=0.5)
save_figure(fig4, f"fig_tl_{TARGET_IBR}_boxplots_over_seeds.pdf")
plt.show()
'''
# ============================================================
# Cell 11 — Optional: Data efficiency (FIXED to use train_target_full)
# ============================================================

ENABLE_DATA_EFFICIENCY = True
K_SHOT = 12
DATA_EFF_FRACS = [0.10, 0.25, 0.50, 1.0]
DATA_EFF_SEED = 42
DATA_EFF_EPOCHS = min(EPOCHS_FT_TARGET, 200)

def subset_dataset(ds: TensorDataset, frac=None, k=None, seed=42):
    assert (frac is not None) ^ (k is not None)
    n_total = len(ds)
    if frac is not None:
        n = max(1, min(n_total, int(np.ceil(n_total * frac))))
    else:
        n = max(1, min(n_total, int(k)))

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=g)[:n]
    tensors = [t[idx] for t in ds.tensors]
    return TensorDataset(*tensors), n, n / n_total

if ENABLE_DATA_EFFICIENCY:
    specs = [("k-shot", {"k": K_SHOT})] + [(f"{int(fr*100)}%", {"frac": fr}) for fr in DATA_EFF_FRACS]
    rows = []

    # IMPORTANT FIX: use train_target_full (NOT train_target_sub / NOT pre-cut 6%)
    base_for_eff = train_target_full

    # build backbones once for this demo run (use first seed for determinism)
    seed_eff = DATA_EFF_SEED
    set_seed(seed_eff)
    fl_state_eff = fedavg_pretrain(
        train_sets_dict=train_sets,
        ibr_list=SOURCE_IBRS,
        rounds=ROUNDS_PRE_FL,
        local_epochs=LOCAL_EPOCHS_FL,
        batch_size=BATCH_SIZE_FL,
        lr_init=LR_INIT_FL,
        seed=seed_eff,
        verbose=False,
    )
    local_state_eff = pretrain_single_source(
        train_ds=train_sets[LOCAL_TL_SOURCE_IBR],
        epochs=EPOCHS_PRE_LOCAL,
        batch_size=BATCH_SIZE_PRE_LOCAL,
        lr=LR_PRE_LOCAL,
        seed=seed_eff + 777,
        verbose=False,
        tag=f"LocalPretrain-{LOCAL_TL_SOURCE_IBR.upper()}",
    )

    print("\n===== Data efficiency (FIXED): limited target labels from FULL target set =====")
    for i, (label, cfg) in enumerate(specs):
        subset, n_used, frac_used = subset_dataset(
            base_for_eff, frac=cfg.get("frac"), k=cfg.get("k"), seed=DATA_EFF_SEED + i
        )
        bs = min(BATCH_SIZE_FT, len(subset))

        # FL-TL
        _, z_fl, tr_fl, te_fl = fine_tune(
            base_state_dict_cpu=fl_state_eff,
            train_ds=subset,
            test_ds=test_target,
            epochs=DATA_EFF_EPOCHS,
            batch_size=bs,
            lr=LR_FT,
            seed=DATA_EFF_SEED + 100 + i,
            tag=f"DataEff-FLTL-{label}",
            verbose=False,
        )
        final_fl = te_fl[-1]
        gap_fl = te_fl[-1] - tr_fl[-1]

        # Local TL
        _, z_lo, tr_lo, te_lo = fine_tune(
            base_state_dict_cpu=local_state_eff,
            train_ds=subset,
            test_ds=test_target,
            epochs=DATA_EFF_EPOCHS,
            batch_size=bs,
            lr=LR_FT,
            seed=DATA_EFF_SEED + 200 + i,
            tag=f"DataEff-LocalTL-{label}",
            verbose=False,
        )
        final_lo = te_lo[-1]
        gap_lo = te_lo[-1] - tr_lo[-1]

        rows.append({
            "label": label, "n_used": n_used, "frac_used": frac_used,
            "final_fltl": float(final_fl), "final_localtl": float(final_lo),
            "gap_fltl": float(gap_fl), "gap_localtl": float(gap_lo)
        })

        print(f"[{label:>6}] n={n_used:>5} ({frac_used:>6.2%}) | "
              f"Final test MSE FL-TL={final_fl:.4e}, LocalTL={final_lo:.4e} | "
              f"Gap(test-train) FL-TL={gap_fl:.4e}, LocalTL={gap_lo:.4e}")

    labels_de = [r["label"] for r in rows]
    fltl_vals = [r["final_fltl"] for r in rows]
    local_vals = [r["final_localtl"] for r in rows]

    fig_de, ax_de = plt.subplots(figsize=(3.5, 2.0))
    ax_de.plot(labels_de, local_vals, marker="s", linestyle="--", label=f"Local TL ({LOCAL_TL_SOURCE_IBR.upper()})")
    ax_de.plot(labels_de, fltl_vals, marker="^", linestyle="-", label="FL-TL (FedAvg backbone)")
    ax_de.set_xlabel("Target labels used (from FULL target set)")
    ax_de.set_ylabel("Final test MSE on target (scaled-Y)")
    ax_de.set_title(f"Data efficiency on IBR1 (FL-TL vs Local TL)")
    ax_de.grid(True, ls=":", alpha=0.5)
    place_legend(ax_de, "data_efficiency", default_loc="upper right", frameon=False)
    save_figure(fig_de, f"fig_tl_{TARGET_IBR}_data_efficiency_fltl_vs_localtl.pdf")
    plt.show()


# ============================================================
# Cell 12 — Optional: Admittance plot (TRUE vs FL-TL vs Local TL)
# (kept from your version; only swap Scratch -> Local TL)
# ============================================================

ZB_BASE_OHM = 95.2  # pu -> Siemens conversion (Y_S = Y_pu / Zbase)

components = [
    ("Ydd", 0, 1),
    ("Ydq", 2, 3),
    ("Yqd", 4, 5),
    ("Yqq", 6, 7),
]

def build_freq_bins(freq_pool, nbins=25):
    f_all = np.asarray(freq_pool)
    f_all = f_all[np.isfinite(f_all)]
    f_all = f_all[f_all > 0]
    f_min, f_max = f_all.min(), f_all.max()
    edges = np.logspace(np.log10(f_min), np.log10(f_max), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers

def reduce_by_bin(f, y, edges):
    f = np.asarray(f); y = np.asarray(y)
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i+1])
        if np.any(m):
            out[i] = np.median(y[m])
    return out

def plot_admittance_true_vs_models(model_fltl, model_localtl, test_ds_scaled, target_ibr, nbins=25):
    Xs = test_ds_scaled.tensors[0].cpu().numpy()
    Ys = test_ds_scaled.tensors[1].cpu().numpy()

    X_phys = X_scaler.inverse_transform(Xs)
    Y_true = Y_scaler.inverse_transform(Ys)

    with torch.no_grad():
        Yp_fl = model_fltl(test_ds_scaled.tensors[0].to(device)).cpu().numpy()
        Yp_lo = model_localtl(test_ds_scaled.tensors[0].to(device)).cpu().numpy()

    Yp_fl = Y_scaler.inverse_transform(Yp_fl)
    Yp_lo = Y_scaler.inverse_transform(Yp_lo)

    # pu -> Siemens
    Y_true = Y_true / ZB_BASE_OHM
    Yp_fl  = Yp_fl  / ZB_BASE_OHM
    Yp_lo  = Yp_lo  / ZB_BASE_OHM

    f = X_phys[:, 3]
    edges, centers = build_freq_bins(f, nbins=nbins)

    # pick one owner (V,P,Q) with enough points
    V, P, Q = X_phys[:, 0], X_phys[:, 1], X_phys[:, 2]
    df = pd.DataFrame({"V": np.round(V, 3), "P": np.round(P, 3), "Q": np.round(Q, 3), "idx": np.arange(len(f))})

    chosen = None
    for (v,p,q), grp in df.groupby(["V","P","Q"]):
        if len(grp) >= 20:
            chosen = (v,p,q, grp["idx"].values)
            break
    if chosen is None:
        chosen = (None, None, None, np.arange(len(f)))

    v,p,q, idxs = chosen
    owner_tag = f"{target_ibr.upper()} | all points" if v is None else f"{target_ibr.upper()} | V={v}, P={p}, Q={q}"

    fig_re, axes_re = plt.subplots(2, 2, figsize=(3.5, 2.0), gridspec_kw={"hspace": 0.7, "wspace": 0.35})
    fig_im, axes_im = plt.subplots(2, 2, figsize=(3.5, 2.0), gridspec_kw={"hspace": 0.7, "wspace": 0.35})
    axes_re = axes_re.ravel()
    axes_im = axes_im.ravel()

    for comp_i, (name, re_i, im_i) in enumerate(components):
        axr = axes_re[comp_i]
        axi = axes_im[comp_i]

        curve_re_true = reduce_by_bin(f[idxs], Y_true[idxs, re_i], edges)
        curve_re_fl   = reduce_by_bin(f[idxs], Yp_fl[idxs,  re_i], edges)
        curve_re_lo   = reduce_by_bin(f[idxs], Yp_lo[idxs,  re_i], edges)

        curve_im_true = reduce_by_bin(f[idxs], Y_true[idxs, im_i], edges)
        curve_im_fl   = reduce_by_bin(f[idxs], Yp_fl[idxs,  im_i], edges)
        curve_im_lo   = reduce_by_bin(f[idxs], Yp_lo[idxs,  im_i], edges)

        axr.semilogx(centers, curve_re_true, label="TRUE")
        axr.semilogx(centers, curve_re_fl,   label="FL-TL", linestyle="--")
        axr.semilogx(centers, curve_re_lo,   label="Local TL", linestyle="-.")
        axr.grid(True, which="both", ls=":", alpha=0.4)
        axr.set_title(f"Re({name})")
        axr.text(0.02, 0.98, owner_tag, transform=axr.transAxes, va="top", ha="left",
                 fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8))

        axi.semilogx(centers, curve_im_true, label="TRUE")
        axi.semilogx(centers, curve_im_fl,   label="FL-TL", linestyle="--")
        axi.semilogx(centers, curve_im_lo,   label="Local TL", linestyle="-.")
        axi.grid(True, which="both", ls=":", alpha=0.4)
        axi.set_title(f"Im({name})")
        axi.text(0.02, 0.98, owner_tag, transform=axi.transAxes, va="top", ha="left",
                 fontsize=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8))

    for ax in axes_re: ax.set_xlabel("Frequency (Hz)")
    for ax in axes_im: ax.set_xlabel("Frequency (Hz)")
    place_legend(axes_re[0], "admittance_re", default_loc="best", frameon=False)
    place_legend(axes_im[0], "admittance_im", default_loc="best", frameon=False)

    fig_re.suptitle(f"{target_ibr.upper()} admittance (Y in S): TRUE vs FL-TL vs Local TL", fontsize=10)
    fig_im.suptitle(f"{target_ibr.upper()} admittance (Y in S): TRUE vs FL-TL vs Local TL", fontsize=10)

    save_figure(fig_re, f"fig_tl_{target_ibr}_admittance_re_fltl_vs_localtl.pdf")
    save_figure(fig_im, f"fig_tl_{target_ibr}_admittance_im_fltl_vs_localtl.pdf")
    plt.show()

# If you want one admittance plot from the LAST seed run, you need the last trained models.
# In this script we didn't keep the final models from multi-seed loop to save memory.
# Quick option: re-run ONE seed with verbose=False and keep models, then call plot_admittance_true_vs_models().
# Example:
#
#   s = SEEDS[0]
#   set_seed(s)
#   train_target_sub, *_ = subset_by_fraction(train_target_full, TARGET_FRACTION, seed=s)
#   fl_state = fedavg_pretrain(...seed=s)
#   local_state = pretrain_single_source(...seed=s+777)
#   model_fltl, _, _, _ = fine_tune(fl_state, train_target_sub, test_target, EPOCHS_FT_TARGET, min(BATCH_SIZE_FT,len(train_target_sub)), LR_FT, seed=s+100)
#   model_local, _, _, _ = fine_tune(local_state, train_target_sub, test_target, EPOCHS_FT_TARGET, min(BATCH_SIZE_FT,len(train_target_sub)), LR_FT, seed=s+200)
#   plot_admittance_true_vs_models(model_fltl, model_local, test_target, TARGET_IBR, nbins=25)
