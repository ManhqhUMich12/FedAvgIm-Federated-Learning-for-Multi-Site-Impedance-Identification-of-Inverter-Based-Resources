# -*- coding: utf-8 -*-
"""
Scenario 4.2 (revised):
Compare FL-TL vs Local TL (single-source pretrain on IBR4) on TARGET IBR1,
run multiple seeds, report mean±std of FINAL test MSE + AUC,
use MSELoss(mean) for optimization (log MSE separately),
and fix Cell 11 to use train_target_full.
"""

# ============================================================
# Cell 1 — Imports & global setup
# ============================================================

import os
from pathlib import Path
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import h5py
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Make runs more reproducible (still not bit-exact on all GPUs, but better)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Cell 2 — User configuration
# ============================================================

LABEL_KEY = "Y_Y"

TRAIN_FILE_PATHS = [
    "gfli1_impedance_dataset.mat",
    "gfli2_impedance_dataset.mat",
    "gfli3_impedance_dataset.mat",
    "gfli4_impedance_dataset.mat",
    "gfli5_impedance_dataset.mat",
    "gfli6_impedance_dataset.mat",
    "gfli7_impedance_dataset.mat",
    "gfli8_impedance_dataset.mat",
    "gfli9_impedance_dataset.mat",
]

TEST_FILE_PATHS = [
    "gfli1_test_impedance_dataset.mat",
    "gfli2_test_impedance_dataset.mat",
    "gfli3_test_impedance_dataset.mat",
    "gfli4_test_impedance_dataset.mat",
    "gfli5_test_impedance_dataset.mat",
    "gfli6_test_impedance_dataset.mat",
    "gfli7_test_impedance_dataset.mat",
    "gfli8_test_impedance_dataset.mat",
    "gfli9_test_impedance_dataset.mat",
]

# Target + Local TL source
TARGET_IBR = "gfli1"
LOCAL_TL_SOURCE_IBR = "gfli4"  # single-source TL baseline (pretrain on IBR4 -> fine-tune on IBR1)

# Fraction of TARGET train set to use for fine-tune (e.g., 0.06 -> 6%).
TARGET_FRACTION = 0.06

# Fine-tune hyperparams
BATCH_SIZE_FT = 512
EPOCHS_FT_TARGET = 500
LR_FT = 1e-3

# FL (FedAvg) pretrain hyperparams
ROUNDS_PRE_FL = 100
LOCAL_EPOCHS_FL = 5
LR_INIT_FL = 1e-3
BATCH_SIZE_FL = 512

# Local TL pretrain hyperparams (single source)
EPOCHS_PRE_LOCAL = 100   # you can tune this; goal is "good init" from IBR4
BATCH_SIZE_PRE_LOCAL = 512
LR_PRE_LOCAL = 1e-3

# Seeds (5–10)
SEEDS = [42, 43, 44, 45, 46]  # change to 10 seeds if you want: e.g., list(range(42, 52))

print(f"#TRAIN files: {len(TRAIN_FILE_PATHS)}, #TEST files: {len(TEST_FILE_PATHS)}")
print("TARGET_IBR:", TARGET_IBR, "| LOCAL_TL_SOURCE_IBR:", LOCAL_TL_SOURCE_IBR)
print("TARGET_FRACTION:", TARGET_FRACTION)
print("SEEDS:", SEEDS)

# ============================================================
# Cell 3 — IO helpers (.mat / .h5)
# ============================================================

def _fix_shape(arr, expected_cols=None):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if expected_cols is not None:
        if arr.shape[1] != expected_cols and arr.shape[0] == expected_cols:
            arr = arr.T
    return arr

def _extract_from_h5(f, label_key="Y_Y"):
    def get_node(g, key):
        return g[key] if key in g else None

    g = get_node(f, "Dataset") or get_node(f, "dataset")

    if g is not None:
        X = _fix_shape(g["X"][()], expected_cols=4)
        Ysrc = g[label_key] if label_key in g else g.get("Y_Y")
        if Ysrc is None:
            raise KeyError("Neither label_key nor 'Y_Y' found under /Dataset")
        Y = _fix_shape(Ysrc[()], expected_cols=8)
        return X, Y

    Xnode = get_node(f, "X")
    Ynode = get_node(f, label_key) or get_node(f, "Y_Y")
    if Xnode is None or Ynode is None:
        raise KeyError("Could not find X and Y in HDF5 file.")

    X = _fix_shape(Xnode[()], expected_cols=4)
    Y = _fix_shape(Ynode[()], expected_cols=8)
    return X, Y

def _extract_from_mat(d, label_key="Y_Y"):
    d2 = {k: v for k, v in d.items() if not k.startswith("__")}

    if "Dataset" in d2:
        G = d2["Dataset"]
        if hasattr(G, "dtype") and G.dtype.names:
            fields = G.dtype.names

            def get_field(name):
                if name in fields:
                    val = np.array(G[name]).squeeze()
                    return val
                return None

            X_raw = get_field("X")
            if X_raw is None:
                raise KeyError("Field 'X' not found in Dataset struct.")
            X = _fix_shape(X_raw, expected_cols=4)

            Yraw = get_field(label_key) or get_field("Y_Y")
            if Yraw is None:
                raise KeyError("Neither label_key nor 'Y_Y' found in Dataset struct.")
            Y = _fix_shape(Yraw, expected_cols=8)
            return X, Y

    X = d2.get("X", None)
    Y = d2.get(label_key, None) or d2.get("Y_Y", None)
    if X is None or Y is None:
        raise KeyError("Could not find X and Y in MAT file (top-level).")

    X = _fix_shape(X, expected_cols=4)
    Y = _fix_shape(Y, expected_cols=8)
    return X, Y

def load_dataset_mat(path, label_key="Y_Y"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lower = path.name.lower()
    family = "GFMI" if "gfmi" in lower else "GFLI"
    is_test = "_test_" in lower
    ibr_full = path.stem.lower()
    ibr_prefix = ibr_full.split("_")[0]  # "gfli1"

    try:
        with h5py.File(path, "r") as f:
            X, Y = _extract_from_h5(f, label_key=label_key)
    except Exception as e_h5:
        try:
            d = loadmat(str(path))
            X, Y = _extract_from_mat(d, label_key=label_key)
        except Exception as e_mat:
            raise Exception(f"Error loading {path}: h5py-> {e_h5}; loadmat-> {e_mat}")

    return {"X": X, "Y": Y, "family": family, "ibr": ibr_prefix, "is_test": is_test, "path": str(path)}

# ============================================================
# Cell 4 — Load TRAIN / TEST (raw) and build per-IBR dict
# ============================================================

loaded_tr, loaded_te = [], []

for fp in TRAIN_FILE_PATHS:
    d = load_dataset_mat(fp, LABEL_KEY)
    if d["family"] == "GFLI":
        loaded_tr.append(d)
        print(f"[TRAIN] {d['ibr']}  X:{d['X'].shape} Y:{d['Y'].shape}")

for fp in TEST_FILE_PATHS:
    d = load_dataset_mat(fp, LABEL_KEY)
    if d["family"] == "GFLI":
        loaded_te.append(d)
        print(f"[TEST ] {d['ibr']}  X:{d['X'].shape} Y:{d['Y'].shape}")

train_raw = {d["ibr"]: d for d in loaded_tr}
test_raw  = {d["ibr"]: d for d in loaded_te}

ALL_IBRS = sorted(train_raw.keys())
if TARGET_IBR not in train_raw or TARGET_IBR not in test_raw:
    raise RuntimeError(f"Missing target train/test for {TARGET_IBR}.")
if LOCAL_TL_SOURCE_IBR not in train_raw:
    raise RuntimeError(f"Missing LOCAL_TL_SOURCE_IBR train for {LOCAL_TL_SOURCE_IBR}.")
if LOCAL_TL_SOURCE_IBR == TARGET_IBR:
    raise ValueError("LOCAL_TL_SOURCE_IBR must be different from TARGET_IBR.")

SOURCE_IBRS = [ibr for ibr in ALL_IBRS if ibr != TARGET_IBR]

print("ALL_IBRS   :", ALL_IBRS)
print("SOURCE_IBRS:", SOURCE_IBRS)
print("TARGET_IBR :", TARGET_IBR)
print("LOCAL TL source:", LOCAL_TL_SOURCE_IBR)

# ============================================================
# Cell 5 — Scaling (fit on SOURCE only) + TensorDatasets (CPU)
# ============================================================

# Fit scalers on SOURCE only (avoid leakage into pretraining)
X_src_all = np.vstack([train_raw[ibr]["X"] for ibr in SOURCE_IBRS])
Y_src_all = np.vstack([train_raw[ibr]["Y"] for ibr in SOURCE_IBRS])

X_scaler = StandardScaler().fit(X_src_all)
Y_scaler = StandardScaler().fit(Y_src_all)

input_dim  = X_src_all.shape[1]  # expect 4
output_dim = Y_src_all.shape[1]  # expect 8
print("Input dim:", input_dim, "| Output dim:", output_dim)

def to_tensor_dataset(X_np, Y_np):
    Xs = X_scaler.transform(X_np).astype(np.float32)
    Ys = Y_scaler.transform(Y_np).astype(np.float32)
    return TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Ys))  # keep CPU tensors

def subset_by_fraction(ds: TensorDataset, frac, seed: int):
    if frac is None or frac >= 1.0:
        return ds, len(ds), len(ds)
    if frac <= 0:
        raise ValueError("frac must be > 0")
    n_total = len(ds)
    n_keep = max(1, int(np.ceil(n_total * frac)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n_total, generator=g)[:n_keep]
    tensors = [t[idx] for t in ds.tensors]
    return TensorDataset(*tensors), n_keep, n_total

train_sets = {ibr: to_tensor_dataset(train_raw[ibr]["X"], train_raw[ibr]["Y"]) for ibr in ALL_IBRS}
test_sets  = {ibr: to_tensor_dataset(test_raw[ibr]["X"],  test_raw[ibr]["Y"])  for ibr in sorted(test_raw.keys())}

train_target_full = train_sets[TARGET_IBR]
test_target = test_sets[TARGET_IBR]

print(f"#train target FULL ({TARGET_IBR}): {len(train_target_full)}")
print(f"#test  target      ({TARGET_IBR}): {len(test_target)}")

# ============================================================
# Cell 6 — Model definitions
# ============================================================

HIDDEN_GFLI = 32

class Trunk(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class Head(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))
    def forward(self, h): return self.net(h)

class FullModel(nn.Module):
    def __init__(self, trunk: Trunk, head: Head):
        super().__init__()
        self.trunk = trunk
        self.head  = head
    def forward(self, x): return self.head(self.trunk(x))

def make_model(in_dim=input_dim, hidden_dim=HIDDEN_GFLI, out_dim=output_dim):
    trunk = Trunk(in_dim=in_dim, hidden_dim=hidden_dim)
    head  = Head(hidden_dim=hidden_dim, out_dim=out_dim)
    return FullModel(trunk, head).to(device)

# ============================================================
# Cell 7 — Metrics + helpers (optimize with mean loss; log MSE separately)
# ============================================================

def compute_mse_on_dataset(model, dataset, batch_size=512):
    """MSE over all elements = sum(e^2) / (N*D)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    crit_sum = nn.MSELoss(reduction="sum")

    total_sum, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total_sum += crit_sum(pred, yb).item()
            total_n += xb.size(0)
    return total_sum / (total_n * output_dim)

def get_model_state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_state_dict_cpu_to_model(model, state_dict_cpu):
    model.load_state_dict(state_dict_cpu, strict=True)

def fine_tune(
    base_state_dict_cpu,  # None => scratch init
    train_ds,
    test_ds,
    epochs,
    batch_size,
    lr,
    seed,
    tag="FineTune",
    verbose=False,
):
    """
    Optimization uses MSELoss(mean).
    Logging uses dataset MSE = sum(e^2)/(N*D) for both train and test.
    """
    model = make_model()
    if base_state_dict_cpu is not None:
        load_state_dict_cpu_to_model(model, base_state_dict_cpu)

    # zero-shot test mse
    mse0_test = compute_mse_on_dataset(model, test_ds, batch_size=batch_size)

    # deterministic-ish shuffling
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_mean = nn.MSELoss(reduction="mean")  # for backprop

    hist_train_mse, hist_test_mse = [], []

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_mean(pred, yb)
            loss.backward()
            opt.step()

        # log MSE separately (exact dataset-level)
        train_mse = compute_mse_on_dataset(model, train_ds, batch_size=batch_size)
        test_mse  = compute_mse_on_dataset(model, test_ds,  batch_size=batch_size)
        hist_train_mse.append(train_mse)
        hist_test_mse.append(test_mse)

        if verbose:
            print(f"[{tag}] Epoch {ep:04d} | train MSE={train_mse:.4e} | test MSE={test_mse:.4e}")

    return model, mse0_test, hist_train_mse, hist_test_mse

# ============================================================
# Cell 8 — FedAvg pretrain (SOURCE only)
# ============================================================

def get_lr_for_round(r, rounds, lr_init=1e-3, factor=0.1):
    d1 = int(np.ceil(0.70 * rounds))
    d2 = int(np.ceil(0.85 * rounds))
    if r <= d1:
        return lr_init
    elif r <= d2:
        return lr_init * factor
    else:
        return lr_init * factor * 0.5

def fedavg_states(model_states_cpu, sizes):
    total = float(sum(sizes))
    keys = model_states_cpu[0].keys()
    avg = {}
    for k in keys:
        acc = torch.zeros_like(model_states_cpu[0][k])
        for sd, n in zip(model_states_cpu, sizes):
            acc += (n / total) * sd[k]
        avg[k] = acc
    return avg

def train_one_client_local(client_ds, global_state_cpu, epochs, batch_size, lr, seed):
    model = make_model()
    load_state_dict_cpu_to_model(model, global_state_cpu)
    model.train()

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(client_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_mean = nn.MSELoss(reduction="mean")

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_mean(pred, yb)
            loss.backward()
            opt.step()

    return get_model_state_dict_cpu(model)

def fedavg_pretrain(train_sets_dict, ibr_list, rounds, local_epochs, batch_size, lr_init, seed, verbose=False):
    """
    FedAvg pretrain on SOURCE IBRs (excluding target).
    """
    clients = []
    for ibr in ibr_list:
        ds = train_sets_dict[ibr]
        n = ds.tensors[0].shape[0]
        clients.append((ibr, ds, n))

    global_model = make_model()
    global_state = get_model_state_dict_cpu(global_model)

    for r in range(1, rounds + 1):
        lr_round = get_lr_for_round(r, rounds, lr_init=lr_init, factor=0.1)
        states, sizes = [], []
        for i, (ibr, ds, n) in enumerate(clients):
            # per-client seed (stable, but changes across outer seed)
            client_seed = seed * 1000 + r * 10 + i
            new_state = train_one_client_local(
                client_ds=ds,
                global_state_cpu=global_state,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr_round,
                seed=client_seed,
            )
            states.append(new_state)
            sizes.append(n)

        global_state = fedavg_states(states, sizes)

        if verbose and (r == 1 or r % 10 == 0 or r == rounds):
            print(f"[FedAvg] Round {r:03d}/{rounds} | lr={lr_round:.2e}")

    return global_state

# ============================================================
# Cell 8b — Local TL pretrain (single-source IBR4)
# ============================================================

def pretrain_single_source(train_ds, epochs, batch_size, lr, seed, verbose=False, tag="LocalPretrain"):
    """
    Pretrain on a single source domain (e.g., IBR4) -> returns state_dict_cpu.
    Uses MSELoss(mean) for optimization.
    """
    model = make_model()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_mean = nn.MSELoss(reduction="mean")

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_mean(pred, yb)
            loss.backward()
            opt.step()
        if verbose and (ep == 1 or ep % 50 == 0 or ep == epochs):
            tr_mse = compute_mse_on_dataset(model, train_ds, batch_size=batch_size)
            print(f"[{tag}] Ep {ep:04d}/{epochs} | train MSE={tr_mse:.4e}")

    return get_model_state_dict_cpu(model)

# ============================================================
# Cell 9 — Multi-seed experiment: FL-TL vs Local TL (IBR4->IBR1)
# ============================================================

def build_curve(zero_val, curve_list):
    curve = np.asarray(curve_list, dtype=float)
    return np.insert(curve, 0, float(zero_val))  # include zero-shot as epoch 0

def auc_trapz(curve):
    x = np.arange(len(curve))
    return float(np.trapz(curve, x))

results = {
    "seed": [],
    "final_mse_fltl": [],
    "auc_fltl": [],
    "final_mse_localtl": [],
    "auc_localtl": [],
    # optional: overfit gap at end (train-test)
    "gap_fltl": [],
    "gap_localtl": [],
}

# Store curves for optional plotting across seeds
curves = {
    "fltl_train": [],
    "fltl_test": [],
    "local_train": [],
    "local_test": [],
}

for s in SEEDS:
    print("\n" + "=" * 80)
    print(f"RUN seed = {s}")
    print("=" * 80)
    set_seed(s)

    # 1) Build target subset ONCE per seed (same subset used by both methods)
    train_target_sub, n_used, n_total = subset_by_fraction(train_target_full, TARGET_FRACTION, seed=s)
    bs_ft = min(BATCH_SIZE_FT, len(train_target_sub))
    print(f"Target subset: {n_used}/{n_total} samples (frac={TARGET_FRACTION}), batch={bs_ft}")

    # 2) FL backbone (FedAvg on SOURCE_IBRS)
    fl_state = fedavg_pretrain(
        train_sets_dict=train_sets,
        ibr_list=SOURCE_IBRS,
        rounds=ROUNDS_PRE_FL,
        local_epochs=LOCAL_EPOCHS_FL,
        batch_size=BATCH_SIZE_FL,
        lr_init=LR_INIT_FL,
        seed=s,
        verbose=False,
    )

    # 3) Local TL backbone: pretrain only on IBR4
    local_state = pretrain_single_source(
        train_ds=train_sets[LOCAL_TL_SOURCE_IBR],
        epochs=EPOCHS_PRE_LOCAL,
        batch_size=BATCH_SIZE_PRE_LOCAL,
        lr=LR_PRE_LOCAL,
        seed=s + 777,  # different stream from fine-tune
        verbose=False,
        tag=f"LocalPretrain-{LOCAL_TL_SOURCE_IBR.upper()}",
    )

    # 4) Fine-tune on TARGET from each backbone
    _, z_fl, tr_fl, te_fl = fine_tune(
        base_state_dict_cpu=fl_state,
        train_ds=train_target_sub,
        test_ds=test_target,
        epochs=EPOCHS_FT_TARGET,
        batch_size=bs_ft,
        lr=LR_FT,
        seed=s + 100,
        tag=f"FL-TL->Target ({TARGET_IBR})",
        verbose=False,
    )

    _, z_lo, tr_lo, te_lo = fine_tune(
        base_state_dict_cpu=local_state,
        train_ds=train_target_sub,
        test_ds=test_target,
        epochs=EPOCHS_FT_TARGET,
        batch_size=bs_ft,
        lr=LR_FT,
        seed=s + 200,
        tag=f"LocalTL({LOCAL_TL_SOURCE_IBR})->Target ({TARGET_IBR})",
        verbose=False,
    )

    fl_test_curve = build_curve(z_fl, te_fl)
    lo_test_curve = build_curve(z_lo, te_lo)

    fl_train_curve = build_curve(tr_fl[0], tr_fl)  # train curve starts after epoch1; pad with first train MSE
    lo_train_curve = build_curve(tr_lo[0], tr_lo)

    # Save per-seed summary
    results["seed"].append(s)
    results["final_mse_fltl"].append(float(fl_test_curve[-1]))
    results["auc_fltl"].append(auc_trapz(fl_test_curve))
    results["final_mse_localtl"].append(float(lo_test_curve[-1]))
    results["auc_localtl"].append(auc_trapz(lo_test_curve))

    # Overfit gap = (test - train) at end (positive => test worse than train)
    results["gap_fltl"].append(float(fl_test_curve[-1] - fl_train_curve[-1]))
    results["gap_localtl"].append(float(lo_test_curve[-1] - lo_train_curve[-1]))

    curves["fltl_train"].append(fl_train_curve)
    curves["fltl_test"].append(fl_test_curve)
    curves["local_train"].append(lo_train_curve)
    curves["local_test"].append(lo_test_curve)

    print(f"[Seed {s}] Final test MSE | FL-TL: {fl_test_curve[-1]:.4e} | Local TL: {lo_test_curve[-1]:.4e}")
    print(f"[Seed {s}] AUC (lower better) | FL-TL: {auc_trapz(fl_test_curve):.4e} | Local TL: {auc_trapz(lo_test_curve):.4e}")
    print(f"[Seed {s}] Overfit gap (test-train) | FL-TL: {results['gap_fltl'][-1]:.4e} | Local TL: {results['gap_localtl'][-1]:.4e}")

# Mean ± std
def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else (float(xs.mean()), 0.0)

m_final_fl, sd_final_fl = mean_std(results["final_mse_fltl"])
m_final_lo, sd_final_lo = mean_std(results["final_mse_localtl"])

m_auc_fl, sd_auc_fl = mean_std(results["auc_fltl"])
m_auc_lo, sd_auc_lo = mean_std(results["auc_localtl"])

m_gap_fl, sd_gap_fl = mean_std(results["gap_fltl"])
m_gap_lo, sd_gap_lo = mean_std(results["gap_localtl"])

print("\n" + "=" * 80)
print("MULTI-SEED SUMMARY (mean ± std)")
print("=" * 80)
print(f"TARGET_IBR = {TARGET_IBR.upper()} | Local TL source = {LOCAL_TL_SOURCE_IBR.upper()} | frac={TARGET_FRACTION}")
print("-" * 80)
print(f"FINAL test MSE | FL-TL   : {m_final_fl:.4e} ± {sd_final_fl:.4e}")
print(f"FINAL test MSE | Local TL: {m_final_lo:.4e} ± {sd_final_lo:.4e}")
print("-" * 80)
print(f"AUC(test MSE curve) | FL-TL   : {m_auc_fl:.4e} ± {sd_auc_fl:.4e}")
print(f"AUC(test MSE curve) | Local TL: {m_auc_lo:.4e} ± {sd_auc_lo:.4e}")
print("-" * 80)
print(f"Overfit gap (test-train) | FL-TL   : {m_gap_fl:.4e} ± {sd_gap_fl:.4e}")
print(f"Overfit gap (test-train) | Local TL: {m_gap_lo:.4e} ± {sd_gap_lo:.4e}")
print("=" * 80)


