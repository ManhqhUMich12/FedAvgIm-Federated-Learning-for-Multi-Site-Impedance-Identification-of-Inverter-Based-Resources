# -*- coding: utf-8 -*-
"""
Scenario H: export FedAvg-estimated IBR1 admittance for Matlab GNC/Nyquist.

This script uses the Scenario F data workflow for two noise conditions
(default: 0% and 2%), trains or reuses cached FedAvg global models, then
queries the model at the five operating points requested for the GNC study.

Matlab-facing output:
  - One MAT file per (noise level, operating point) with:
        Yibr_resp : 2 x 2 x nGNC complex double
        fGNC      : 1 x nGNC frequency in Hz
        wGNC      : 1 x nGNC angular frequency in rad/s
  - One combined MAT file with:
        Yibr_resp_all : 2 x 2 x nGNC x nOP x nNoise complex double
  - One long CSV with real/imag columns for audit/import.

Default assumptions:
  - X columns are [V, P, Q, f_Hz].
  - The OP image gives [P, Q], so V defaults to 1.0 p.u.
  - Frequencies default to the 20 unique f_Hz points in gfli1 test data,
    matching the usual fGNC = logspace(log10(1), log10(200), 20).

Example:
    .\\.venv\\Scripts\\python.exe fl_scenario_h_gnc_export.py

Then in Matlab:
    load('scenario_h_gnc_exports/scenario_h_noise0pct_OP1_Yibr_resp.mat');
    % Yibr_resp is ready for:
    %   Lk = Yibr_resp(:,:,k) / Yg_resp(:,:,k);
"""

from __future__ import annotations

import argparse
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat, savemat

from fl_scenario_ae_fedavg import (
    LABEL_KEY,
    FNN,
    aggregate_fedavg,
    clone_state_dict,
    evaluate_state,
    load_dataset_mat,
    load_scenario_clients,
    select_clients,
    set_seed,
    train_client_sgd,
)
from fl_scenario_f_noise_fedavg import discover_conditions, format_noise_label


DEFAULT_OPERATING_POINTS_PQ = np.array(
    [
        [1.0, 0.0],
        [0.5, 0.0],
        [0.8, 0.0],
        [0.8, 0.6],
        [0.8, -0.6],
    ],
    dtype=float,
)

COMPONENT_COLUMNS = (
    ("Ydd", 0, 0, 0, 1),
    ("Ydq", 0, 1, 2, 3),
    ("Yqd", 1, 0, 4, 5),
    ("Yqq", 1, 1, 6, 7),
)


@dataclass(frozen=True)
class OperatingPoint:
    label: str
    v: float
    p: float
    q: float


@dataclass
class ModelBundle:
    noise_label: str
    noise_percent: float
    model_state_dict: dict
    input_dim: int
    hidden_dim: int
    output_dim: int
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    checkpoint_path: Path


def percent_tag(percent: float) -> str:
    if float(percent).is_integer():
        return f"{int(percent)}pct"
    text = f"{percent:g}".replace(".", "p").replace("-", "neg")
    return f"{text}pct"


def safe_tag(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def scaler_arrays(scaler) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(scaler.scale_, dtype=np.float64).copy()
    scale[scale == 0.0] = 1.0
    return np.asarray(scaler.mean_, dtype=np.float64).copy(), scale


def transform_with_arrays(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - mean) / scale).astype(np.float32)


def inverse_transform_with_arrays(y: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64) * scale + mean


def state_to_cpu(state_dict: dict) -> dict:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def load_checkpoint(path: Path, expected_noise_percent: float | None = None) -> ModelBundle | None:
    if not path.exists():
        return None

    try:
        ckpt = torch.load(path, map_location="cpu")
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    required = {
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "x_scaler_mean",
        "x_scaler_scale",
        "y_scaler_mean",
        "y_scaler_scale",
        "noise_label",
        "noise_percent",
    }
    missing = sorted(required.difference(ckpt.keys()))
    if missing:
        print(f"[WARN] Checkpoint {path} is missing {missing}; retraining.")
        return None

    noise_percent = float(ckpt["noise_percent"])
    if expected_noise_percent is not None and not np.isclose(noise_percent, expected_noise_percent):
        print(
            f"[WARN] Checkpoint {path} has noise={noise_percent:g}%, "
            f"expected {expected_noise_percent:g}%; retraining."
        )
        return None

    return ModelBundle(
        noise_label=str(ckpt["noise_label"]),
        noise_percent=noise_percent,
        model_state_dict=ckpt["model_state_dict"],
        input_dim=int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        output_dim=int(ckpt["output_dim"]),
        x_mean=np.asarray(ckpt["x_scaler_mean"], dtype=np.float64),
        x_scale=np.asarray(ckpt["x_scaler_scale"], dtype=np.float64),
        y_mean=np.asarray(ckpt["y_scaler_mean"], dtype=np.float64),
        y_scale=np.asarray(ckpt["y_scaler_scale"], dtype=np.float64),
        checkpoint_path=path,
    )


def save_checkpoint(
    path: Path,
    condition,
    global_state: dict,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    x_scaler,
    y_scaler,
    args: argparse.Namespace,
) -> ModelBundle:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_mean, x_scale = scaler_arrays(x_scaler)
    y_mean, y_scale = scaler_arrays(y_scaler)

    payload = {
        "model_state_dict": state_to_cpu(global_state),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "output_dim": int(output_dim),
        "x_scaler_mean": x_mean,
        "x_scaler_scale": x_scale,
        "y_scaler_mean": y_mean,
        "y_scaler_scale": y_scale,
        "noise_label": condition.label,
        "noise_percent": float(condition.percent),
        "label_key": LABEL_KEY,
        "train_paths": list(map(str, condition.train_paths)),
        "test_paths": list(map(str, condition.test_paths)),
        "scenario_h_args": vars(args),
    }
    torch.save(payload, path)

    return ModelBundle(
        noise_label=condition.label,
        noise_percent=float(condition.percent),
        model_state_dict=payload["model_state_dict"],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        checkpoint_path=path,
    )


def train_fedavg_condition(condition, args: argparse.Namespace, device: torch.device, output_dir: Path) -> ModelBundle:
    set_seed(args.seed)
    clients, test_sets, input_dim, output_dim, x_scaler, y_scaler = load_scenario_clients(
        condition.train_paths,
        condition.test_paths,
        LABEL_KEY,
    )

    global_model = FNN(input_dim, args.hidden_dim, output_dim).to(device)
    global_state = clone_state_dict(global_model)
    rng = random.Random(args.seed)
    history_rows = []

    print(f"\n===== Scenario H train/cache | noise={condition.label} =====")
    print(f"Loaded {len(clients)} train clients and {len(test_sets)} test sets.")

    for round_idx in range(1, args.rounds + 1):
        selected = select_clients(clients, args.client_fraction, rng)
        client_states, client_sizes, local_losses = [], [], []

        for client in selected:
            client_state, local_loss = train_client_sgd(
                client=client,
                global_state=global_state,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                optimizer_name=args.optimizer,
                device=device,
            )
            client_states.append(client_state)
            client_sizes.append(client.n)
            local_losses.append(local_loss)

        global_state = aggregate_fedavg(client_states, client_sizes)

        train_eval = evaluate_state(
            global_state,
            clients,
            input_dim,
            args.hidden_dim,
            output_dim,
            args.eval_batch_size,
            device,
        )
        test_eval = evaluate_state(
            global_state,
            test_sets,
            input_dim,
            args.hidden_dim,
            output_dim,
            args.eval_batch_size,
            device,
        )
        train_mse = float(np.mean([row["mse"] for row in train_eval]))
        test_mse = float(np.mean([row["mse"] for row in test_eval]))
        history_rows.append(
            {
                "noise_label": condition.label,
                "noise_percent": float(condition.percent),
                "round": round_idx,
                "local_train_mse_mean": float(np.mean(local_losses)),
                "train_mse_mean": train_mse,
                "test_mse_mean": test_mse,
            }
        )

        if round_idx == 1 or round_idx == args.rounds or round_idx % args.log_every == 0:
            print(
                f"[Scenario H noise={condition.label}] round={round_idx:03d} "
                f"train={train_mse:.4e} test={test_mse:.4e}"
            )

    tag = percent_tag(condition.percent)
    history_path = output_dir / f"scenario_h_fedavg_noise{tag}_history.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    final_test = pd.DataFrame(
        evaluate_state(
            global_state,
            test_sets,
            input_dim,
            args.hidden_dim,
            output_dim,
            args.eval_batch_size,
            device,
        )
    )
    final_test["noise_label"] = condition.label
    final_test["noise_percent"] = float(condition.percent)
    final_test.to_csv(output_dir / f"scenario_h_fedavg_noise{tag}_final_test_per_ibr.csv", index=False)

    checkpoint_path = output_dir / "models" / f"scenario_h_fedavg_noise{tag}.pt"
    bundle = save_checkpoint(
        checkpoint_path,
        condition,
        global_state,
        input_dim,
        args.hidden_dim,
        output_dim,
        x_scaler,
        y_scaler,
        args,
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    return bundle


def load_or_train_condition(condition, args: argparse.Namespace, device: torch.device, output_dir: Path) -> ModelBundle:
    checkpoint_path = output_dir / "models" / f"scenario_h_fedavg_noise{percent_tag(condition.percent)}.pt"
    if not args.force_retrain:
        bundle = load_checkpoint(checkpoint_path, expected_noise_percent=float(condition.percent))
        if bundle is not None:
            print(f"Using cached checkpoint for noise={condition.label}: {checkpoint_path}")
            return bundle

    return train_fedavg_condition(condition, args, device, output_dir)


def filter_conditions(root: Path, noise_levels: Iterable[float], include_clean: bool):
    conditions = discover_conditions(root, include_clean=include_clean)
    wanted = [float(item) for item in noise_levels]
    selected = []
    for wanted_level in wanted:
        matches = [condition for condition in conditions if np.isclose(condition.percent, wanted_level)]
        if not matches:
            available = ", ".join(format_noise_label(condition.percent) for condition in conditions)
            raise RuntimeError(
                f"Could not find complete Scenario F condition for noise={wanted_level:g}%. "
                f"Available conditions: {available or 'none'}."
            )
        selected.append(matches[0])
    return selected


def read_operating_points(args: argparse.Namespace) -> list[OperatingPoint]:
    if args.op_csv is None:
        arr = DEFAULT_OPERATING_POINTS_PQ
        return [
            OperatingPoint(label=f"OP{idx + 1}", v=float(args.v_ref), p=float(row[0]), q=float(row[1]))
            for idx, row in enumerate(arr)
        ]

    path = Path(args.op_csv)
    if not path.exists():
        raise FileNotFoundError(f"Operating-point CSV not found: {path}")

    try:
        df = pd.read_csv(path)
        lower_cols = {col.lower(): col for col in df.columns}
        if {"p", "q"}.issubset(lower_cols):
            label_col = lower_cols.get("label") or lower_cols.get("op") or lower_cols.get("name")
            v_col = lower_cols.get("v")
            points = []
            for idx, row in df.iterrows():
                label = str(row[label_col]) if label_col else f"OP{idx + 1}"
                v = float(row[v_col]) if v_col else float(args.v_ref)
                points.append(
                    OperatingPoint(
                        label=label,
                        v=v,
                        p=float(row[lower_cols["p"]]),
                        q=float(row[lower_cols["q"]]),
                    )
                )
            return points
    except pd.errors.ParserError:
        pass

    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] == 2:
        return [
            OperatingPoint(label=f"OP{idx + 1}", v=float(args.v_ref), p=float(row[0]), q=float(row[1]))
            for idx, row in enumerate(raw)
        ]
    if raw.shape[1] == 3:
        return [
            OperatingPoint(label=f"OP{idx + 1}", v=float(row[0]), p=float(row[1]), q=float(row[2]))
            for idx, row in enumerate(raw)
        ]
    raise ValueError("--op-csv must contain columns P,Q or V,P,Q.")


def find_ibr_path(paths: list[str], target_ibr: str) -> Path:
    target = target_ibr.lower()
    for path in paths:
        p = Path(path)
        if p.stem.lower().split("_")[0] == target:
            return p
    raise FileNotFoundError(f"Could not find {target_ibr} path in condition file list.")


def load_freqs_from_dataset(path: Path) -> np.ndarray:
    data = load_dataset_mat(path, LABEL_KEY)
    freqs = np.unique(np.asarray(data["X"][:, 3], dtype=np.float64))
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    if freqs.size == 0:
        raise ValueError(f"No positive frequency column found in {path}.")
    return np.sort(freqs)


def load_freqs_from_mat(path: Path) -> np.ndarray:
    d = loadmat(path)
    keys = {key.lower(): key for key in d.keys() if not key.startswith("__")}
    if "fgnc" in keys:
        freqs = np.asarray(d[keys["fgnc"]], dtype=np.float64).reshape(-1)
    elif "wgnc" in keys:
        freqs = np.asarray(d[keys["wgnc"]], dtype=np.float64).reshape(-1) / (2.0 * np.pi)
    elif "w" in keys:
        freqs = np.asarray(d[keys["w"]], dtype=np.float64).reshape(-1) / (2.0 * np.pi)
    else:
        raise KeyError(f"{path} must contain fGNC, wGNC, or w.")
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    if freqs.size == 0:
        raise ValueError(f"No positive frequency values found in {path}.")
    return np.asarray(freqs, dtype=np.float64)


def read_frequencies(args: argparse.Namespace, first_condition) -> np.ndarray:
    if args.freqs:
        freqs = np.asarray(args.freqs, dtype=np.float64)
    elif args.freq_mat:
        freqs = load_freqs_from_mat(Path(args.freq_mat))
    elif args.freq_source == "logspace":
        freqs = np.logspace(np.log10(args.freq_min), np.log10(args.freq_max), args.freq_count)
    else:
        target_test_path = find_ibr_path(first_condition.test_paths, args.target_ibr)
        freqs = load_freqs_from_dataset(target_test_path)

    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    if freqs.size == 0:
        raise ValueError("No positive frequencies were provided.")
    return freqs


def y_vector_to_matrix(y_phys: np.ndarray) -> np.ndarray:
    y_phys = np.asarray(y_phys, dtype=np.float64)
    if y_phys.ndim != 2 or y_phys.shape[1] != 8:
        raise ValueError(f"Expected y_phys with shape (n, 8); got {y_phys.shape}.")

    n_freq = y_phys.shape[0]
    y_matrix = np.zeros((2, 2, n_freq), dtype=np.complex128)
    for _, row_idx, col_idx, re_idx, im_idx in COMPONENT_COLUMNS:
        y_matrix[row_idx, col_idx, :] = y_phys[:, re_idx] + 1j * y_phys[:, im_idx]
    return y_matrix


def predict_yibr_for_op(
    bundle: ModelBundle,
    op: OperatingPoint,
    freqs_hz: np.ndarray,
    device: torch.device,
    divide_zbase: bool,
    zbase_ohm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_query = np.column_stack(
        [
            np.full_like(freqs_hz, op.v, dtype=np.float64),
            np.full_like(freqs_hz, op.p, dtype=np.float64),
            np.full_like(freqs_hz, op.q, dtype=np.float64),
            freqs_hz.astype(np.float64),
        ]
    )
    x_scaled = transform_with_arrays(x_query, bundle.x_mean, bundle.x_scale)

    model = FNN(bundle.input_dim, bundle.hidden_dim, bundle.output_dim).to(device)
    model.load_state_dict(bundle.model_state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(x_scaled).to(device)
        y_scaled = model(xb).cpu().numpy()

    y_phys = inverse_transform_with_arrays(y_scaled, bundle.y_mean, bundle.y_scale)
    if divide_zbase:
        y_phys = y_phys / float(zbase_ohm)

    return y_vector_to_matrix(y_phys), x_query, y_phys


def mat_string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(1, -1)


def save_exports(
    output_dir: Path,
    bundles: list[ModelBundle],
    operating_points: list[OperatingPoint],
    freqs_hz: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    w_rad_s = 2.0 * np.pi * freqs_hz

    all_resp = np.zeros(
        (2, 2, len(freqs_hz), len(operating_points), len(bundles)),
        dtype=np.complex128,
    )
    csv_rows = []
    manifest_rows = []

    for noise_idx, bundle in enumerate(bundles):
        noise_tag = percent_tag(bundle.noise_percent)
        for op_idx, op in enumerate(operating_points):
            y_resp, x_query, y_vector = predict_yibr_for_op(
                bundle=bundle,
                op=op,
                freqs_hz=freqs_hz,
                device=device,
                divide_zbase=args.divide_zbase,
                zbase_ohm=args.zbase_ohm,
            )
            all_resp[:, :, :, op_idx, noise_idx] = y_resp

            op_tag = safe_tag(op.label)
            mat_path = output_dir / f"scenario_h_noise{noise_tag}_{op_tag}_Yibr_resp.mat"
            savemat(
                mat_path,
                {
                    "Yibr_resp": y_resp,
                    "fGNC": freqs_hz.reshape(1, -1),
                    "wGNC": w_rad_s.reshape(1, -1),
                    "X_query": x_query,
                    "operating_point": np.array([[op.v, op.p, op.q]], dtype=np.float64),
                    "op_label": op.label,
                    "noise_label": bundle.noise_label,
                    "noise_percent": np.array([[bundle.noise_percent]], dtype=np.float64),
                    "target_ibr": args.target_ibr,
                    "component_order": "Ydd,Ydq,Yqd,Yqq with Re/Im pairs from Y_Y",
                    "units_note": "raw Y_Y units" if not args.divide_zbase else f"Y_Y divided by {args.zbase_ohm:g} ohm",
                },
                do_compression=True,
            )

            manifest_rows.append(
                {
                    "noise_label": bundle.noise_label,
                    "noise_percent": bundle.noise_percent,
                    "op_label": op.label,
                    "V": op.v,
                    "P": op.p,
                    "Q": op.q,
                    "n_freq": len(freqs_hz),
                    "mat_file": str(mat_path),
                    "checkpoint": str(bundle.checkpoint_path),
                }
            )

            for freq_idx, freq in enumerate(freqs_hz):
                row = {
                    "noise_label": bundle.noise_label,
                    "noise_percent": bundle.noise_percent,
                    "op_label": op.label,
                    "V": op.v,
                    "P": op.p,
                    "Q": op.q,
                    "f_Hz": float(freq),
                    "w_rad_s": float(w_rad_s[freq_idx]),
                }
                for name, row_idx, col_idx, re_idx, im_idx in COMPONENT_COLUMNS:
                    value = y_resp[row_idx, col_idx, freq_idx]
                    row[f"{name}_re"] = float(value.real)
                    row[f"{name}_im"] = float(value.imag)
                    row[f"{name}_complex"] = f"{value.real:.16g}{value.imag:+.16g}i"
                    # Also keep the vector position used before matrix assembly.
                    row[f"{name}_vec_re"] = float(y_vector[freq_idx, re_idx])
                    row[f"{name}_vec_im"] = float(y_vector[freq_idx, im_idx])
                csv_rows.append(row)

            print(f"Saved Matlab-ready Yibr_resp: {mat_path}")

    combined_mat_path = output_dir / "scenario_h_all_Yibr_resp.mat"
    savemat(
        combined_mat_path,
        {
            "Yibr_resp_all": all_resp,
            "fGNC": freqs_hz.reshape(1, -1),
            "wGNC": w_rad_s.reshape(1, -1),
            "operating_points": np.array([[op.v, op.p, op.q] for op in operating_points], dtype=np.float64),
            "op_labels": mat_string_array([op.label for op in operating_points]),
            "noise_percents": np.array([bundle.noise_percent for bundle in bundles], dtype=np.float64).reshape(1, -1),
            "noise_labels": mat_string_array([bundle.noise_label for bundle in bundles]),
            "target_ibr": args.target_ibr,
            "dimension_note": "Yibr_resp_all is 2 x 2 x nGNC x nOP x nNoise",
            "component_order": "Ydd,Ydq,Yqd,Yqq with Re/Im pairs from Y_Y",
            "units_note": "raw Y_Y units" if not args.divide_zbase else f"Y_Y divided by {args.zbase_ohm:g} ohm",
        },
        do_compression=True,
    )

    csv_path = output_dir / "scenario_h_yibr_resp_long.csv"
    manifest_path = output_dir / "scenario_h_manifest.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\nSaved combined MAT: {combined_mat_path}")
    print(f"Saved long CSV:     {csv_path}")
    print(f"Saved manifest:     {manifest_path}")


def run_scenario_h(args: argparse.Namespace) -> None:
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = filter_conditions(
        root=root,
        noise_levels=args.noise_levels,
        include_clean=not args.no_clean,
    )
    operating_points = read_operating_points(args)
    freqs_hz = read_frequencies(args, first_condition=conditions[0])

    print("Scenario H conditions:")
    for condition in conditions:
        print(f"  noise={condition.label} train_files={len(condition.train_paths)} test_files={len(condition.test_paths)}")
    print("Operating points interpreted as [V, P, Q]:")
    for op in operating_points:
        print(f"  {op.label}: [{op.v:g}, {op.p:g}, {op.q:g}]")
    print(f"Frequency points: n={len(freqs_hz)}, f_min={freqs_hz.min():g} Hz, f_max={freqs_hz.max():g} Hz")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    bundles = [load_or_train_condition(condition, args, device, output_dir) for condition in conditions]
    save_exports(output_dir, bundles, operating_points, freqs_hz, device, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario H: export FedAvg-estimated IBR1 admittance for Matlab GNC/Nyquist."
    )
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="scenario_h_gnc_exports")
    parser.add_argument("--noise-levels", type=float, nargs="*", default=[0.0, 2.0])
    parser.add_argument("--no-clean", action="store_true", help="Use only discovered noise folders; exclude root clean data.")
    parser.add_argument("--target-ibr", type=str, default="gfli1")

    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--local-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force-retrain", action="store_true")

    parser.add_argument(
        "--op-csv",
        type=str,
        default=None,
        help="Optional OP CSV. Header columns P,Q or V,P,Q are supported; no-header 2/3-column CSV also works.",
    )
    parser.add_argument("--v-ref", type=float, default=1.0, help="Default V when OPs are provided as [P,Q].")

    parser.add_argument(
        "--freq-source",
        choices=["test", "logspace"],
        default="test",
        help="test: use target IBR test-data frequencies; logspace: use --freq-min/--freq-max/--freq-count.",
    )
    parser.add_argument("--freqs", type=float, nargs="*", default=None, help="Explicit fGNC frequencies in Hz.")
    parser.add_argument("--freq-mat", type=str, default=None, help="MAT file containing fGNC, wGNC, or w.")
    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--freq-max", type=float, default=200.0)
    parser.add_argument("--freq-count", type=int, default=20)

    parser.add_argument(
        "--divide-zbase",
        action="store_true",
        help="Export Y_Y / zbase_ohm. Leave off for raw Y_Y values matching previous model plots.",
    )
    parser.add_argument("--zbase-ohm", type=float, default=95.2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_scenario_h(parse_args())
