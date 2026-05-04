# -*- coding: utf-8 -*-
"""
Scenario A/E: standard FedAvg for GFLI impedance estimation.

This script is a standalone implementation of the standard FedAvg algorithm
from McMahan et al. (AISTATS 2017, arXiv:1602.05629) for the same data setup
used in FL_SCENARIO_AE.ipynb.

Key differences from personalized FL:
  - One global FullModel is shared by all clients.
  - Every selected client trains all model parameters locally.
  - The server aggregates full client model weights by local sample count.
  - No client-specific heads, fine-tuning, or personalization state is kept.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


@dataclass
class Client:
    name: str
    dataset: TensorDataset
    n: int


@dataclass
class TestSet:
    name: str
    dataset: TensorDataset
    n: int


class Trunk(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    def __init__(self, hidden_dim: int = 32, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class FullModel(nn.Module):
    """Same Trunk + Head model interface used by FL_SCENARIO_AE.ipynb."""

    def __init__(self, trunk: Trunk, head: Head):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class FNN(FullModel):
    """Convenience constructor for the same full model."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, out_dim: int = 8):
        super().__init__(Trunk(in_dim, hidden_dim), Head(hidden_dim, out_dim))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    g = get_node(f, "Dataset")
    if g is None:
        g = get_node(f, "dataset")

    if g is not None:
        x = _fix_shape(g["X"][()], expected_cols=4)
        y_src = g[label_key] if label_key in g else g.get("Y_Y")
        if y_src is None:
            raise KeyError("Neither label_key nor 'Y_Y' found under /Dataset.")
        y = _fix_shape(y_src[()], expected_cols=8)
        return x, y

    x_node = get_node(f, "X")
    y_node = get_node(f, label_key) or get_node(f, "Y_Y")
    if x_node is None or y_node is None:
        raise KeyError("Could not find X and Y in HDF5 file.")
    x = _fix_shape(x_node[()], expected_cols=4)
    y = _fix_shape(y_node[()], expected_cols=8)
    return x, y


def _extract_from_mat(d, label_key="Y_Y"):
    d2 = {k: v for k, v in d.items() if not k.startswith("__")}

    if "Dataset" in d2:
        group = d2["Dataset"]
        if hasattr(group, "dtype") and group.dtype.names:
            fields = group.dtype.names

            def get_field(name):
                if name in fields:
                    return np.array(group[name]).squeeze()
                return None

            x_raw = get_field("X")
            if x_raw is None:
                raise KeyError("Field 'X' not found in Dataset struct.")
            y_raw = get_field(label_key)
            if y_raw is None:
                y_raw = get_field("Y_Y")
            if y_raw is None:
                raise KeyError("Neither label_key nor 'Y_Y' found in Dataset struct.")
            return _fix_shape(x_raw, expected_cols=4), _fix_shape(y_raw, expected_cols=8)

    x = d2.get("X", None)
    y = d2.get(label_key, None)
    if y is None:
        y = d2.get("Y_Y", None)
    if x is None or y is None:
        raise KeyError("Could not find X and Y in MAT file.")
    return _fix_shape(x, expected_cols=4), _fix_shape(y, expected_cols=8)


def load_dataset_mat(path, label_key="Y_Y"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lower = path.name.lower()
    family = "GFMI" if "gfmi" in lower else "GFLI"
    is_test = "_test_" in lower
    ibr = path.stem.lower().split("_")[0]

    try:
        with h5py.File(path, "r") as f:
            x, y = _extract_from_h5(f, label_key=label_key)
    except Exception as e_h5:
        try:
            d = loadmat(str(path))
            x, y = _extract_from_mat(d, label_key=label_key)
        except Exception as e_mat:
            raise RuntimeError(f"Error loading {path}: h5py -> {e_h5}; loadmat -> {e_mat}") from e_mat

    return {"X": x, "Y": y, "family": family, "ibr": ibr, "is_test": is_test, "path": str(path)}


def to_tensor_dataset(x_np, y_np, x_scaler, y_scaler) -> TensorDataset:
    x_scaled = x_scaler.transform(x_np).astype(np.float32)
    y_scaled = y_scaler.transform(y_np).astype(np.float32)
    return TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_scaled))


def load_scenario_clients(train_paths, test_paths, label_key):
    loaded_train = [load_dataset_mat(path, label_key) for path in train_paths]
    loaded_test = [load_dataset_mat(path, label_key) for path in test_paths]

    train_gfli = [d for d in loaded_train if d["family"] == "GFLI"]
    test_gfli = [d for d in loaded_test if d["family"] == "GFLI"]
    if not train_gfli:
        raise RuntimeError("No GFLI train datasets were loaded.")
    if not test_gfli:
        raise RuntimeError("No GFLI test datasets were loaded.")

    x_train_all = np.vstack([d["X"] for d in train_gfli])
    y_train_all = np.vstack([d["Y"] for d in train_gfli])
    x_scaler = StandardScaler().fit(x_train_all)
    y_scaler = StandardScaler().fit(y_train_all)

    clients = []
    for d in train_gfli:
        dataset = to_tensor_dataset(d["X"], d["Y"], x_scaler, y_scaler)
        clients.append(Client(name=d["ibr"], dataset=dataset, n=len(dataset)))

    test_sets = []
    for d in test_gfli:
        dataset = to_tensor_dataset(d["X"], d["Y"], x_scaler, y_scaler)
        test_sets.append(TestSet(name=d["ibr"], dataset=dataset, n=len(dataset)))

    input_dim = x_train_all.shape[1]
    output_dim = y_train_all.shape[1]
    return clients, test_sets, input_dim, output_dim, x_scaler, y_scaler


def get_lr(round_idx: int, initial_lr: float, decay_round_1: int, decay_round_2: int, decay_factor: float) -> float:
    if decay_round_1 <= 0:
        return initial_lr
    if round_idx <= decay_round_1:
        return initial_lr
    if decay_round_2 <= 0 or round_idx <= decay_round_2:
        return initial_lr * decay_factor
    return initial_lr * decay_factor * 0.5


def clone_state_dict(model: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def savefig_no_crash(fig, path: Path) -> Path:
    """Save a figure, falling back to a numbered filename if the target is locked."""
    path = Path(path)
    try:
        fig.savefig(path)
        return path
    except PermissionError:
        for idx in range(1, 100):
            fallback = path.with_name(f"{path.stem}_{idx}{path.suffix}")
            try:
                fig.savefig(fallback)
                print(f"[WARN] Could not overwrite locked file: {path}")
                print(f"[WARN] Saved figure instead as: {fallback}")
                return fallback
            except PermissionError:
                continue
        raise


def aggregate_fedavg(client_states, client_sizes):
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Cannot aggregate empty client updates.")

    avg_state = {}
    for key in client_states[0].keys():
        avg_state[key] = sum((n / total) * state[key] for state, n in zip(client_states, client_sizes))
    return avg_state


def train_client_sgd(
    client: Client,
    global_state,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str,
    device: torch.device,
):
    model = FNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()

    loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    total_loss, total_n = 0.0, 0
    for _ in range(local_epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

    return clone_state_dict(model), total_loss / max(total_n, 1)


def evaluate_state(global_state, datasets, input_dim, hidden_dim, output_dim, batch_size, device):
    model = FNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(global_state, strict=True)
    model.eval()

    criterion = nn.MSELoss()
    rows = []
    with torch.no_grad():
        for item in datasets:
            loader = DataLoader(item.dataset, batch_size=batch_size, shuffle=False)
            total_loss, total_n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = criterion(model(xb), yb)
                total_loss += loss.item() * xb.size(0)
                total_n += xb.size(0)
            rows.append({"name": item.name, "mse": total_loss / total_n, "n": total_n})
    return rows


def evaluate_model(model, dataset: TensorDataset, batch_size: int, device: torch.device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = criterion(model(xb), yb)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
    return total_loss / total_n


def build_global_dataset(clients) -> TensorDataset:
    x_all = torch.cat([client.dataset.tensors[0] for client in clients], dim=0)
    y_all = torch.cat([client.dataset.tensors[1] for client in clients], dim=0)
    return TensorDataset(x_all, y_all)


def train_centralized_model(
    train_dataset: TensorDataset,
    test_sets,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    optimizer_name: str,
    device: torch.device,
):
    model = FNN(input_dim, hidden_dim, output_dim).to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_curve, test_curve = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        train_mse = total_loss / total_n
        test_mse = float(
            np.mean([evaluate_model(model, item.dataset, eval_batch_size, device) for item in test_sets])
        )
        train_curve.append(train_mse)
        test_curve.append(test_mse)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(f"[Centralized] epoch={epoch:03d} train={train_mse:.4e} test={test_mse:.4e}")

    return model, np.array(train_curve), np.array(test_curve)


def set_model_state_dict(model: nn.Module, state_dict) -> None:
    model.load_state_dict(state_dict, strict=True)


def as_plot_client(client: Client):
    return {"name": client.name, "dataset": client.dataset, "n": client.n}


def as_plot_test_set(test_set: TestSet):
    x, y = test_set.dataset.tensors
    return {"name": test_set.name, "X": x, "Y": y, "n": test_set.n}


def run_existing_plot_scripts(plot_globals, output_dir: Path, run_local_only_plots: bool) -> None:
    original_cwd = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(output_dir)
        script_dir = Path(__file__).resolve().parent
        combined_path = script_dir / "plot_fl_central_local_combined.py"
        print(f"\nRunning combined plot script -> {output_dir}")
        exec(compile(combined_path.read_text(encoding="utf-8"), str(combined_path), "exec"), plot_globals)

        if run_local_only_plots:
            local_path = script_dir / "FL_vs_LocalOnly.py"
            print(f"\nRunning FL-vs-local plot script -> {output_dir}")
            exec(compile(local_path.read_text(encoding="utf-8"), str(local_path), "exec"), plot_globals)
    finally:
        os.chdir(original_cwd)


def select_clients(clients, fraction: float, rng: random.Random):
    if not 0 < fraction <= 1:
        raise ValueError("--client-fraction must be in (0, 1].")
    m = max(1, int(np.ceil(fraction * len(clients))))
    if m >= len(clients):
        return list(clients)
    return rng.sample(clients, m)


def run_fedavg(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    clients, test_sets, input_dim, output_dim, x_scaler, y_scaler = load_scenario_clients(
        TRAIN_FILE_PATHS,
        TEST_FILE_PATHS,
        LABEL_KEY,
    )
    print(f"Loaded {len(clients)} train clients and {len(test_sets)} test sets.")
    for client in clients:
        print(f"  client={client.name}, n={client.n}")

    global_model = FNN(input_dim, args.hidden_dim, output_dim).to(device)
    global_state = clone_state_dict(global_model)
    rng = random.Random(args.seed)
    history_rows = []

    for round_idx in range(1, args.rounds + 1):
        selected = select_clients(clients, args.client_fraction, rng)
        lr = get_lr(round_idx, args.lr, args.decay_round_1, args.decay_round_2, args.decay_factor)

        client_states = []
        client_sizes = []
        local_losses = []
        for client in selected:
            client_state, local_loss = train_client_sgd(
                client=client,
                global_state=global_state,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=lr,
                optimizer_name=args.optimizer,
                device=device,
            )
            client_states.append(client_state)
            client_sizes.append(client.n)
            local_losses.append(local_loss)

        global_state = aggregate_fedavg(client_states, client_sizes)

        train_eval = evaluate_state(global_state, clients, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device)
        test_eval = evaluate_state(global_state, test_sets, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device)
        train_macro_mse = float(np.mean([row["mse"] for row in train_eval]))
        test_macro_mse = float(np.mean([row["mse"] for row in test_eval]))
        test_accuracy_percent = float(np.clip(100.0 * (1.0 - test_macro_mse), 0.0, 100.0))

        history_rows.append(
            {
                "round": round_idx,
                "lr": lr,
                "selected_clients": len(selected),
                "local_train_mse_mean": float(np.mean(local_losses)),
                "train_macro_mse": train_macro_mse,
                "test_macro_mse": test_macro_mse,
                "train_mse_mean": train_macro_mse,
                "test_mse_mean": test_macro_mse,
                "test_accuracy_percent": test_accuracy_percent,
            }
        )

        if round_idx == 1 or round_idx == args.rounds or round_idx % args.log_every == 0:
            print(
                f"[FedAvg] round={round_idx:03d} "
                f"clients={len(selected)}/{len(clients)} "
                f"lr={lr:.2e} train={train_macro_mse:.4e} test={test_macro_mse:.4e}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(history_rows)
    history_path = output_dir / "scenario_ae_fedavg_history.csv"
    history.to_csv(history_path, index=False)

    final_train = pd.DataFrame(evaluate_state(global_state, clients, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device))
    final_test = pd.DataFrame(evaluate_state(global_state, test_sets, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device))
    final_train.to_csv(output_dir / "scenario_ae_fedavg_final_train_per_client.csv", index=False)
    final_test.to_csv(output_dir / "scenario_ae_fedavg_final_test_per_client.csv", index=False)

    model_path = output_dir / "scenario_ae_fedavg_global_model.pt"
    torch.save(
        {
            "model_state_dict": deepcopy(global_state),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": output_dim,
            "args": vars(args),
        },
        model_path,
    )

    if args.plot:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(history["round"], history["train_macro_mse"], label="Train macro MSE")
        ax.plot(history["round"], history["test_macro_mse"], label="Test macro MSE")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Scaled MSE")
        ax.set_title("Scenario A/E Standard FedAvg")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        savefig_no_crash(fig, output_dir / "scenario_ae_fedavg_learning_curves.svg")
        savefig_no_crash(fig, output_dir / "scenario_ae_fedavg_learning_curves.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(
            history["round"],
            history["test_accuracy_percent"],
            color="#2ca02c",
            linewidth=1.8,
            label="FedAvg test accuracy",
        )
        final_round = int(history["round"].iloc[-1])
        final_acc = float(history["test_accuracy_percent"].iloc[-1])
        ax.scatter([final_round], [final_acc], color="#2ca02c", s=28, zorder=3)
        ax.annotate(
            f"Final: {final_acc:.2f}%",
            xy=(final_round, final_acc),
            xytext=(-72, -18 if final_acc > 88 else 14),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2ca02c", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0),
        )
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Testing accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Scenario A/E FedAvg Testing Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        savefig_no_crash(fig, output_dir / "scenario_ae_fedavg_test_accuracy.svg")
        savefig_no_crash(fig, output_dir / "scenario_ae_fedavg_test_accuracy.pdf")
        plt.close(fig)

    if args.compare:
        print("\n===== Centralized global training for comparison =====")
        global_dataset = build_global_dataset(clients)
        central_model, central_train_curve, central_test_curve = train_centralized_model(
            train_dataset=global_dataset,
            test_sets=test_sets,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            epochs=args.rounds,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            optimizer_name=args.optimizer,
            device=device,
        )

        central_df = pd.DataFrame(
            {
                "epoch": np.arange(1, args.rounds + 1),
                "train_mse": central_train_curve,
                "test_mse": central_test_curve,
            }
        )
        central_df.to_csv(output_dir / "scenario_ae_centralized_history.csv", index=False)

        plot_globals = {
            "__name__": "__main__",
            "np": np,
            "pd": pd,
            "plt": plt,
            "torch": torch,
            "nn": nn,
            "DataLoader": DataLoader,
            "TensorDataset": TensorDataset,
            "history_fullfedavg": history.to_dict(orient="list"),
            "fullfedavg_model_state": deepcopy(global_state),
            "central_model": central_model,
            "central_train_curve": central_train_curve,
            "central_test_curve": central_test_curve,
            "clients_gfli": [as_plot_client(client) for client in clients],
            "test_sets_gfli": [as_plot_test_set(test_set) for test_set in test_sets],
            "HIDDEN_GFLI": args.hidden_dim,
            "ROUNDS_FL": args.rounds,
            "BATCH_SIZE_FL": args.batch_size,
            "LR_INIT_FL": args.lr,
            "Trunk": Trunk,
            "Head": Head,
            "FullModel": FullModel,
            "set_model_state_dict": set_model_state_dict,
            "device": device,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "X_scaler_gfli": x_scaler,
            "Y_scaler_gfli": y_scaler,
        }
        run_existing_plot_scripts(
            plot_globals=plot_globals,
            output_dir=output_dir,
            run_local_only_plots=not args.skip_local_only_script,
        )

    print("\nFinal test MSE by IBR:")
    print(final_test.to_string(index=False))
    print(f"\nSaved history: {history_path}")
    print(f"Saved model:   {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standard FedAvg for FL_SCENARIO_AE GFLI impedance data.")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--decay-round-1", type=int, default=700)
    parser.add_argument("--decay-round-2", type=int, default=800)
    parser.add_argument("--decay-factor", type=float, default=0.1)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="fedavg_scenario_ae_results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train centralized baseline, train local-only baselines inside plot scripts, and create AE comparison figures.",
    )
    parser.add_argument(
        "--skip-local-only-script",
        action="store_true",
        help="With --compare, skip FL_vs_LocalOnly.py because plot_fl_central_local_combined.py already trains local-only models.",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_fedavg(parse_args())
