# -*- coding: utf-8 -*-
"""
FL_SCENARIO_D_NO_OVERFIT.ipynb

Goal: reduce overfitting when fine-tuning FL-pretrained model on low-data target (e.g., IBR9).
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
from torch.utils.data import TensorDataset, DataLoader, random_split

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Scenario: New client TL with FL backbone
TARGET_IBR = "gfli9"

# Optional: single-source TL baseline (pretrain on one IBR, then fine-tune target)
ENABLE_SINGLE_SOURCE_TL = True
SINGLE_SOURCE_PRETRAIN_IBR = "gfli4"
EPOCHS_PRETRAIN_SINGLE_SOURCE = 80  # reduce a bit to avoid overfit on single source

# ---------- Fine-tune hyperparams (anti-overfit) ----------
BATCH_SIZE_FT = 8                    # small batch helps generalization on small dataset
EPOCHS_FT_MAX = 120
VAL_FRAC = 0.15
EARLY_STOP_PATIENCE = 15

FREEZE_TRUNK_EPOCHS = 8              # head-only warmup
LR_HEAD = 3e-4                       # head learns faster
LR_TRUNK = 1e-4                      # trunk adapts slowly
WEIGHT_DECAY = 2e-4
DROPOUT_P = 0.10
GRAD_CLIP_NORM = 1.0

# scheduler on VAL
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Train loss (robust) but report MSE metric
TRAIN_LOSS_TYPE = "smoothl1"         # "mse" or "smoothl1"
SMOOTHL1_BETA = 0.5                  # Huber beta

# ---------- FedAvg pretrain ----------
ROUNDS_PRE_FL = 120
LOCAL_EPOCHS_FL = 4
LR_INIT_FL = 1e-3
BATCH_SIZE_FL = 8
WEIGHT_DECAY_FL = 1e-5               # small, optional
DROPOUT_PRETRAIN = 0.0               # keep 0 for stable FedAvg

print(f"#TRAIN files: {len(TRAIN_FILE_PATHS)}, #TEST files: {len(TEST_FILE_PATHS)}")
print("TARGET_IBR:", TARGET_IBR)

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
    raise RuntimeError(f"Missing target train/test for {TARGET_IBR}. Available train: {ALL_IBRS}, test: {sorted(test_raw.keys())}")

SOURCE_IBRS = [ibr for ibr in ALL_IBRS if ibr != TARGET_IBR]
print("ALL_IBRS   :", ALL_IBRS)
print("SOURCE_IBRS:", SOURCE_IBRS)
print("TARGET_IBR :", TARGET_IBR)

if ENABLE_SINGLE_SOURCE_TL:
    if SINGLE_SOURCE_PRETRAIN_IBR == TARGET_IBR:
        raise RuntimeError("SINGLE_SOURCE_PRETRAIN_IBR must differ from TARGET_IBR.")
    if SINGLE_SOURCE_PRETRAIN_IBR not in train_raw or SINGLE_SOURCE_PRETRAIN_IBR not in test_raw:
        raise RuntimeError(f"SINGLE_SOURCE_PRETRAIN_IBR '{SINGLE_SOURCE_PRETRAIN_IBR}' not found in datasets.")

# ============================================================
# Cell 5 — Scaling (fit on SOURCE only) + TensorDatasets (CPU)
# ============================================================

# Fit scalers on SOURCE only (avoid leakage)
X_src_all = np.vstack([train_raw[ibr]["X"] for ibr in SOURCE_IBRS])
Y_src_all = np.vstack([train_raw[ibr]["Y"] for ibr in SOURCE_IBRS])

X_scaler = StandardScaler().fit(X_src_all)
Y_scaler = StandardScaler().fit(Y_src_all)

input_dim  = X_src_all.shape[1]   # expect 4
output_dim = Y_src_all.shape[1]   # expect 8
print("Input dim:", input_dim, "| Output dim:", output_dim)

def to_tensor_dataset(X_np, Y_np):
    Xs = X_scaler.transform(X_np).astype(np.float32)
    Ys = Y_scaler.transform(Y_np).astype(np.float32)
    return TensorDataset(torch.from_numpy(Xs), torch.from_numpy(Ys))

train_sets = {ibr: to_tensor_dataset(train_raw[ibr]["X"], train_raw[ibr]["Y"]) for ibr in ALL_IBRS}
test_sets  = {ibr: to_tensor_dataset(test_raw[ibr]["X"],  test_raw[ibr]["Y"])  for ibr in sorted(test_raw.keys())}

train_target = train_sets[TARGET_IBR]
test_target  = test_sets[TARGET_IBR]
print(f"#train target ({TARGET_IBR}): {len(train_target)}")
print(f"#test  target ({TARGET_IBR}): {len(test_target)}")

# ============================================================
# Cell 6 — Model definitions
# ============================================================

HIDDEN_GFLI = 32

class Trunk(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, dropout_p=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        if dropout_p and dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=8, dropout_p=0.0):
        super().__init__()
        layers = [nn.Linear(hidden_dim, 16), nn.ReLU()]
        if dropout_p and dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(16, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)

class FullModel(nn.Module):
    def __init__(self, trunk: Trunk, head: Head):
        super().__init__()
        self.trunk = trunk
        self.head  = head
    def forward(self, x):
        return self.head(self.trunk(x))

def make_model(in_dim=input_dim, hidden_dim=HIDDEN_GFLI, out_dim=output_dim, dropout_p=0.0):
    trunk = Trunk(in_dim=in_dim, hidden_dim=hidden_dim, dropout_p=dropout_p)
    head  = Head(hidden_dim=hidden_dim, out_dim=out_dim, dropout_p=dropout_p)
    return FullModel(trunk, head).to(device)

# ============================================================
# Cell 7 — Metrics + fine-tune helpers (NO test leakage)
# ============================================================

def compute_mse_on_dataset(model, dataset, batch_size=64):
    """Mean MSE over all elements (N*D)."""
    if dataset is None:
        return None
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    crit = nn.MSELoss(reduction="mean")

    total_weighted, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            total_weighted += loss.item() * xb.size(0)
            total_n += xb.size(0)

    return total_weighted / total_n if total_n > 0 else float("nan")

def get_model_state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_state_dict_cpu_to_model(model, state_dict_cpu):
    sd = dict(state_dict_cpu)

    # Compatibility: if checkpoint has head.net.2.weight/bias (no dropout),
    # but current model expects head.net.3 (dropout inserted at idx 2).
    needs_shift = ("head.net.3.weight" in model.state_dict()) and ("head.net.3.weight" not in sd)
    if needs_shift and "head.net.2.weight" in sd:
        sd["head.net.3.weight"] = sd.pop("head.net.2.weight")
        sd["head.net.3.bias"] = sd.pop("head.net.2.bias")

    model.load_state_dict(sd, strict=False)

def build_train_criterion(loss_type="smoothl1", beta=0.5):
    if loss_type.lower() == "mse":
        return nn.MSELoss(reduction="mean")
    # SmoothL1 = Huber-like
    return nn.SmoothL1Loss(beta=beta, reduction="mean")

def fine_tune_fltl(
    base_state_dict_cpu,                 # FL backbone
    train_ds,
    test_ds,
    epochs_max=120,
    batch_size=8,
    val_frac=0.15,
    early_stop_patience=15,
    freeze_trunk_epochs=8,
    lr_head=3e-4,
    lr_trunk=1e-4,
    weight_decay=2e-4,
    dropout_p=0.1,
    grad_clip_norm=1.0,
    loss_type="smoothl1",
    smoothl1_beta=0.5,
    seed=SEED,
    tag="FLTL-FineTune",
):
    model = make_model(dropout_p=dropout_p)
    load_state_dict_cpu_to_model(model, base_state_dict_cpu)

    # init metrics (report only)
    mse0_test = compute_mse_on_dataset(model, test_ds, batch_size=max(8, batch_size))
    print(f"[{tag}] Init/Zero-shot TEST MSE = {mse0_test:.4e}")

    # val split (from TRAIN only)
    if val_frac > 0.0 and len(train_ds) >= 5:
        n_val = max(1, int(round(len(train_ds) * val_frac)))
        n_val = min(n_val, len(train_ds) - 2)  # keep >=2 for train
        n_train = len(train_ds) - n_val
        train_split, val_split = random_split(
            train_ds, lengths=[n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_split, val_split = train_ds, None
        print(f"[{tag}] Warning: not enough data for VAL split -> best checkpoint uses TRAIN metric (no TEST leakage).")

    train_loader = DataLoader(train_split, batch_size=min(batch_size, len(train_split)), shuffle=True)
    val_loader_bs = max(8, batch_size)

    # Freeze trunk initially
    def set_trunk_grad(flag: bool):
        for p in model.trunk.parameters():
            p.requires_grad = flag

    set_trunk_grad(False if freeze_trunk_epochs > 0 else True)
    trunk_frozen = (freeze_trunk_epochs > 0)

    # Optimizer with param groups (trunk slower)
    # Note: even when trunk frozen, optimizer is fine; frozen params just have no grad.
    opt = torch.optim.Adam(
        [
            {"params": model.head.parameters(),  "lr": lr_head,  "weight_decay": weight_decay},
            {"params": model.trunk.parameters(), "lr": lr_trunk, "weight_decay": weight_decay},
        ]
    )

    crit = build_train_criterion(loss_type=loss_type, beta=smoothl1_beta)

    # Scheduler watches VAL (or TRAIN if no VAL)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR, verbose=True
    )

    hist = {"train_mse": [], "val_mse": [], "test_mse": []}

    best_state = get_model_state_dict_cpu(model)
    best_metric = float("inf")
    patience_ctr = 0

    for ep in range(1, epochs_max + 1):
        # unfreeze trunk after warmup
        if trunk_frozen and ep > freeze_trunk_epochs:
            set_trunk_grad(True)
            trunk_frozen = False
            print(f"[{tag}] Unfreeze trunk at epoch {ep}")

        model.train()
        total_weighted, n_total = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            opt.step()

            # report MSE on train for tracking (not train loss)
            with torch.no_grad():
                mse_batch = nn.MSELoss(reduction="mean")(pred, yb)
            total_weighted += mse_batch.item() * xb.size(0)
            n_total += xb.size(0)

        train_mse = total_weighted / max(1, n_total)
        val_mse = compute_mse_on_dataset(model, val_split, batch_size=val_loader_bs) if val_split else None
        test_mse = compute_mse_on_dataset(model, test_ds, batch_size=val_loader_bs)

        hist["train_mse"].append(train_mse)
        hist["val_mse"].append(val_mse if val_mse is not None else np.nan)
        hist["test_mse"].append(test_mse)

        # choose metric for checkpointing: VAL if exists else TRAIN (NEVER TEST)
        ref_mse = val_mse if val_mse is not None else train_mse

        # scheduler step (same ref metric)
        scheduler.step(ref_mse)

        improved = (ref_mse < best_metric - 1e-8)
        if improved:
            best_metric = ref_mse
            best_state = get_model_state_dict_cpu(model)
            patience_ctr = 0
        else:
            patience_ctr += 1

        # print current lrs
        lr_h = opt.param_groups[0]["lr"]
        lr_t = opt.param_groups[1]["lr"]

        print(
            f"[{tag}] Ep {ep:03d} | "
            f"train MSE={train_mse:.4e} | "
            f"{'val MSE=' + f'{val_mse:.4e}' + ' | ' if val_mse is not None else ''}"
            f"test MSE={test_mse:.4e} | "
            f"lr_head={lr_h:.2e}, lr_trunk={lr_t:.2e}"
        )

        if early_stop_patience and patience_ctr >= early_stop_patience:
            print(f"[{tag}] Early stopping at epoch {ep} (no improvement for {early_stop_patience} checks).")
            break

    # load best checkpoint (by VAL or TRAIN)
    load_state_dict_cpu_to_model(model, best_state)

    # final metrics after loading best
    final_test = compute_mse_on_dataset(model, test_ds, batch_size=val_loader_bs)
    print(f"[{tag}] Best checkpoint loaded | FINAL TEST MSE = {final_test:.4e}")

    return model, mse0_test, hist

# ============================================================
# Cell 8 — FedAvg pretrain (SOURCE only, stable)
# ============================================================

def get_lr_for_round(r, rounds, lr_init=1e-3):
    return lr_init  # fixed

def fedavg_states(model_states_cpu, sizes):
    total = float(sum(sizes))
    avg = {}
    keys = model_states_cpu[0].keys()
    for k in keys:
        acc = torch.zeros_like(model_states_cpu[0][k])
        for sd, n in zip(model_states_cpu, sizes):
            acc += (n / total) * sd[k]
        avg[k] = acc
    return avg

def train_one_client_local(client_ds, global_state_cpu, epochs=1, batch_size=8, lr=1e-3, weight_decay=0.0):
    model = make_model(dropout_p=DROPOUT_PRETRAIN)
    load_state_dict_cpu_to_model(model, global_state_cpu)
    model.train()

    loader = DataLoader(client_ds, batch_size=min(batch_size, len(client_ds)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss(reduction="mean")

    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

    return get_model_state_dict_cpu(model)

def fedavg_pretrain(train_sets_dict, ibr_list, rounds=120, local_epochs=4, batch_size=8, tag="FedAvg-Pretrain"):
    print(f"\n===== {tag} on SOURCE IBRs: {ibr_list} =====")

    clients = []
    for ibr in ibr_list:
        ds = train_sets_dict[ibr]
        n = ds.tensors[0].shape[0]
        clients.append((ibr, ds, n))

    global_model = make_model(dropout_p=DROPOUT_PRETRAIN)
    global_state = get_model_state_dict_cpu(global_model)

    for r in range(1, rounds + 1):
        lr_round = get_lr_for_round(r, rounds, lr_init=LR_INIT_FL)

        states, sizes = [], []
        for ibr, ds, n in clients:
            new_state = train_one_client_local(
                client_ds=ds,
                global_state_cpu=global_state,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr_round,
                weight_decay=WEIGHT_DECAY_FL,
            )
            states.append(new_state)
            sizes.append(n)

        global_state = fedavg_states(states, sizes)

        if r == 1 or r % 10 == 0 or r == rounds:
            print(f"[{tag}] Round {r:03d}/{rounds} done | lr={lr_round:.2e}")

    return global_state

# ============================================================
# Cell 9 — Run: FL pretrain -> fine-tune target (anti-overfit)
# ============================================================

# 1) FL backbone (FedAvg on SOURCE)
fl_global_state = fedavg_pretrain(
    train_sets_dict=train_sets,
    ibr_list=SOURCE_IBRS,
    rounds=ROUNDS_PRE_FL,
    local_epochs=LOCAL_EPOCHS_FL,
    batch_size=BATCH_SIZE_FL,
    tag="FedAvg-Pretrain",
)

# 2) Fine-tune on TARGET from FL backbone (anti-overfit)
model_ft_fl, zero_fl_test, hist_fl = fine_tune_fltl(
    base_state_dict_cpu=fl_global_state,
    train_ds=train_target,
    test_ds=test_target,
    epochs_max=EPOCHS_FT_MAX,
    batch_size=BATCH_SIZE_FT,
    val_frac=VAL_FRAC,
    early_stop_patience=EARLY_STOP_PATIENCE,
    freeze_trunk_epochs=FREEZE_TRUNK_EPOCHS,
    lr_head=LR_HEAD,
    lr_trunk=LR_TRUNK,
    weight_decay=WEIGHT_DECAY,
    dropout_p=DROPOUT_P,
    grad_clip_norm=GRAD_CLIP_NORM,
    loss_type=TRAIN_LOSS_TYPE,
    smoothl1_beta=SMOOTHL1_BETA,
    tag=f"FL-TL->Target ({TARGET_IBR})",
)

# Optional: single-source TL baseline (still with anti-overfit fine-tune)
model_ft_single, zero_ss_test, hist_ss = None, None, None
if ENABLE_SINGLE_SOURCE_TL:
    print(f"\n===== Single-source baseline: pretrain {SINGLE_SOURCE_PRETRAIN_IBR} then fine-tune {TARGET_IBR} =====")

    # single-source pretrain (no val split; best checkpoint uses TRAIN, never TEST)
    model_pre_single, _, hist_pre = fine_tune_fltl(
        base_state_dict_cpu=get_model_state_dict_cpu(make_model(dropout_p=0.0)),  # start from random init state
        train_ds=train_sets[SINGLE_SOURCE_PRETRAIN_IBR],
        test_ds=test_sets[SINGLE_SOURCE_PRETRAIN_IBR],
        epochs_max=EPOCHS_PRETRAIN_SINGLE_SOURCE,
        batch_size=BATCH_SIZE_FT,
        val_frac=0.0,                   # no VAL
        early_stop_patience=0,           # no ES
        freeze_trunk_epochs=0,           # pretrain full model
        lr_head=1e-3,
        lr_trunk=1e-3,
        weight_decay=WEIGHT_DECAY,
        dropout_p=0.0,                  # keep stable in pretrain
        grad_clip_norm=GRAD_CLIP_NORM,
        loss_type="mse",
        smoothl1_beta=SMOOTHL1_BETA,
        tag=f"Pretrain ({SINGLE_SOURCE_PRETRAIN_IBR})",
    )
    single_src_state = get_model_state_dict_cpu(model_pre_single)

    # fine-tune target with anti-overfit settings
    model_ft_single, zero_ss_test, hist_ss = fine_tune_fltl(
        base_state_dict_cpu=single_src_state,
        train_ds=train_target,
        test_ds=test_target,
        epochs_max=EPOCHS_FT_MAX,
        batch_size=BATCH_SIZE_FT,
        val_frac=VAL_FRAC,
        early_stop_patience=EARLY_STOP_PATIENCE,
        freeze_trunk_epochs=FREEZE_TRUNK_EPOCHS,
        lr_head=LR_HEAD,
        lr_trunk=LR_TRUNK,
        weight_decay=WEIGHT_DECAY,
        dropout_p=DROPOUT_P,
        grad_clip_norm=GRAD_CLIP_NORM,
        loss_type=TRAIN_LOSS_TYPE,
        smoothl1_beta=SMOOTHL1_BETA,
        tag=f"TL-{SINGLE_SOURCE_PRETRAIN_IBR}->{TARGET_IBR}",
    )

# ============================================================
# Cell 10 — Plot (FL-TL + optional single-source)
# ============================================================

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

def save_figure(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.endswith(".pdf"):
        fig.savefig(filename.replace(".pdf", ".svg"), bbox_inches="tight")

def curve_from_hist(zero_test_mse, hist_dict):
    # show test curve; insert zero-shot test MSE at epoch 0
    test_curve = np.asarray(hist_dict["test_mse"], dtype=float)
    return np.insert(test_curve, 0, float(zero_test_mse))

curves = {"FL-TL": curve_from_hist(zero_fl_test, hist_fl)}
plot_order = ["FL-TL"]

if ENABLE_SINGLE_SOURCE_TL and (hist_ss is not None):
    single_label = f"TL-{SINGLE_SOURCE_PRETRAIN_IBR.upper()}"
    curves[single_label] = curve_from_hist(zero_ss_test, hist_ss)
    plot_order.append(single_label)

final_mse = {k: v[-1] for k, v in curves.items()}
zero_shot = {k: v[0] for k, v in curves.items()}
auc = {k: float(np.trapz(v, np.arange(len(v)))) for k, v in curves.items()}

fig1, ax1 = plt.subplots(figsize=(4.2, 3.2))
styles = {"FL-TL": {"linestyle": "--", "marker": "^"}}
if ENABLE_SINGLE_SOURCE_TL and (hist_ss is not None):
    styles[single_label] = {"linestyle": ":", "marker": "v"}

for lbl in plot_order:
    curve = curves[lbl]
    st = styles.get(lbl, {"linestyle": "-", "marker": "o"})
    x_axis = np.arange(len(curve))
    ax1.plot(
        x_axis,
        curve,
        label=lbl,
        linestyle=st["linestyle"],
        marker=st["marker"],
        markevery=max(len(x_axis)//12, 1),
        markersize=3,
    )
ax1.set_xlabel("Fine-tuning epoch on target")
ax1.set_ylabel("Test MSE on target")
ax1.set_title(f"Scenario 4.2: TL on {TARGET_IBR.upper()} (anti-overfit)")
ax1.grid(True, ls=":", alpha=0.5)
ax1.legend(loc="upper right", frameon=False)
save_figure(fig1, f"fig_tl_{TARGET_IBR}_curves_anti_overfit.pdf")
plt.show()

# bar: zero-shot vs final
labels = plot_order
zero_vals  = [zero_shot[lbl] for lbl in labels]
final_vals = [final_mse[lbl] for lbl in labels]

x = np.arange(len(labels))
w = 0.36
fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
ax2.bar(x - w/2, zero_vals,  w, label="Init/Zero-shot")
ax2.bar(x + w/2, final_vals, w, label="After fine-tune")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=10)
ax2.set_ylabel("Test MSE on target")
ax2.set_title("Init vs fine-tuned performance (anti-overfit)")
ax2.grid(True, axis="y", ls=":", alpha=0.5)
ax2.legend(loc="upper right", frameon=False)
save_figure(fig2, f"fig_tl_{TARGET_IBR}_zero_vs_final_anti_overfit.pdf")
plt.show()

# bar: AUC
fig3, ax3 = plt.subplots(figsize=(3.8, 2.8))
ax3.bar(labels, [auc[lbl] for lbl in labels])
ax3.set_ylabel("AUC of test MSE (lower is better)")
ax3.set_title("Learning efficiency across fine-tuning (anti-overfit)")
ax3.grid(True, axis="y", ls=":", alpha=0.5)
save_figure(fig3, f"fig_tl_{TARGET_IBR}_auc_anti_overfit.pdf")
plt.show()

print("\n" + "="*70)
print("Scenario 4.2 summary (anti-overfit, NO test leakage for checkpointing)")
print("="*70)
print("TARGET_IBR:", TARGET_IBR)
print("SOURCE_IBRS:", SOURCE_IBRS)
print("-"*70)
print("Init/Zero-shot TEST MSE:")
for lbl in labels:
    print(f"  {lbl:>15}: {zero_shot[lbl]:.4e}")
print("Final TEST MSE (best checkpoint loaded):")
for lbl in labels:
    print(f"  {lbl:>15}: {final_mse[lbl]:.4e}")
print("AUC (lower better):")
for lbl in labels:
    print(f"  {lbl:>15}: {auc[lbl]:.4e}")
best = min(final_mse, key=final_mse.get)
print(f"Best final TEST MSE: {best} ({final_mse[best]:.4e})")
print("="*70)
