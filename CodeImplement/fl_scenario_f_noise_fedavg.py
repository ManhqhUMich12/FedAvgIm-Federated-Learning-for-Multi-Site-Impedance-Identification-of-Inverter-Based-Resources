# -*- coding: utf-8 -*-
"""
Scenario F: validate standard FedAvg under different measurement-noise levels.

The script runs FedAvg independently for each discovered noise condition and
creates:
  1. Final scaled Test MSE vs Noise Level
  2. FedAvg physical-accuracy Learning Curves Across Noise Levels
  3. Per-IBR final scaled Test MSE plus physical prediction metrics

Supported data layouts:
  - Folder layout:
      one_percent_noise/gfli1_impedance_dataset.mat
      one_percent_noise/gfli1_test_impedance_dataset.mat
      five_percent_noise/gfli1_impedance_dataset.mat
      ...

  - Filename layout:
      gfli1_impedance_dataset_1percent_noise.mat
      gfli1_test_impedance_dataset_1percent_noise.mat
      gfli2_impedance_dataset_1percent_noise.mat
      ...
"""

from __future__ import annotations

import argparse
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fl_scenario_ae_fedavg import (
    LABEL_KEY,
    TEST_FILE_PATHS,
    TRAIN_FILE_PATHS,
    FNN,
    aggregate_fedavg,
    clone_state_dict,
    evaluate_state,
    load_scenario_clients,
    savefig_no_crash,
    select_clients,
    set_seed,
    train_client_sgd,
)


WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


SCENARIO_F_NOISE_LEVELS = (0.0, 1.0, 2.0, 3.0)


def apply_paper_style() -> None:
    """Use the same compact publication style as the Scenario A/E, B, and D figures."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def mse_to_accuracy_percent(mse: float) -> float:
    return float(np.clip(100.0 * (1.0 - mse), 0.0, 100.0))


def rel_error_to_accuracy_percent(rel_error_percent: float) -> float:
    return float(np.clip(100.0 - rel_error_percent, 0.0, 100.0))


def evaluate_state_physical(global_state, datasets, input_dim, hidden_dim, output_dim, batch_size, device, y_scaler):
    """Evaluate predictions after inverse-scaling Y back to physical admittance coordinates."""
    model = FNN(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(global_state, strict=True)
    model.eval()

    eps = np.finfo(np.float64).eps
    rows = []
    with torch.no_grad():
        for item in datasets:
            loader = DataLoader(item.dataset, batch_size=batch_size, shuffle=False)
            se_sum, true_sq_sum, value_count = 0.0, 0.0, 0
            real_se_sum, real_true_sq_sum, real_value_count = 0.0, 0.0, 0
            imag_se_sum, imag_true_sq_sum, imag_value_count = 0.0, 0.0, 0
            rel_error_sum, sample_count = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(device)
                pred_scaled = model(xb).cpu().numpy()
                true_scaled = yb.cpu().numpy()

                pred_phys = y_scaler.inverse_transform(pred_scaled)
                true_phys = y_scaler.inverse_transform(true_scaled)
                err = pred_phys - true_phys

                se_sum += float(np.sum(err**2))
                true_sq_sum += float(np.sum(true_phys**2))
                value_count += int(err.size)

                real_cols = np.arange(0, err.shape[1], 2)
                imag_cols = np.arange(1, err.shape[1], 2)
                err_real = err[:, real_cols]
                true_real = true_phys[:, real_cols]
                err_imag = err[:, imag_cols]
                true_imag = true_phys[:, imag_cols]

                real_se_sum += float(np.sum(err_real**2))
                real_true_sq_sum += float(np.sum(true_real**2))
                real_value_count += int(err_real.size)
                imag_se_sum += float(np.sum(err_imag**2))
                imag_true_sq_sum += float(np.sum(true_imag**2))
                imag_value_count += int(err_imag.size)

                # The 8 Y_Y columns are Re/Im pairs for a 2x2 complex matrix.
                # Row-wise L2 norm is therefore the Frobenius norm of that complex matrix.
                err_norm = np.sqrt(np.sum(err**2, axis=1))
                true_norm = np.sqrt(np.sum(true_phys**2, axis=1))
                rel_error_sum += float(np.sum(err_norm / np.maximum(true_norm, eps)))
                sample_count += int(err.shape[0])

            physical_rmse = float(np.sqrt(se_sum / max(value_count, 1)))
            physical_rel_error_percent = 100.0 * rel_error_sum / max(sample_count, 1)
            physical_nrmse_rel_error_percent = 100.0 * float(np.sqrt(se_sum / max(true_sq_sum, eps)))
            physical_real_rmse = float(np.sqrt(real_se_sum / max(real_value_count, 1)))
            physical_imag_rmse = float(np.sqrt(imag_se_sum / max(imag_value_count, 1)))
            physical_real_rel_error_percent = 100.0 * float(
                np.sqrt(real_se_sum / max(real_true_sq_sum, eps))
            )
            physical_imag_rel_error_percent = 100.0 * float(
                np.sqrt(imag_se_sum / max(imag_true_sq_sum, eps))
            )
            rows.append(
                {
                    "name": item.name,
                    "physical_rmse": physical_rmse,
                    "physical_rel_error_percent": physical_rel_error_percent,
                    "physical_accuracy_percent": rel_error_to_accuracy_percent(physical_rel_error_percent),
                    "physical_nrmse_rel_error_percent": physical_nrmse_rel_error_percent,
                    "physical_nrmse_accuracy_percent": rel_error_to_accuracy_percent(
                        physical_nrmse_rel_error_percent
                    ),
                    "physical_real_rmse": physical_real_rmse,
                    "physical_real_rel_error_percent": physical_real_rel_error_percent,
                    "physical_real_accuracy_percent": rel_error_to_accuracy_percent(
                        physical_real_rel_error_percent
                    ),
                    "physical_imag_rmse": physical_imag_rmse,
                    "physical_imag_rel_error_percent": physical_imag_rel_error_percent,
                    "physical_imag_accuracy_percent": rel_error_to_accuracy_percent(
                        physical_imag_rel_error_percent
                    ),
                    "n": sample_count,
                }
            )
    return rows


@dataclass
class NoiseCondition:
    label: str
    percent: float
    train_paths: list[str]
    test_paths: list[str]


def parse_noise_percent(text: str) -> float | None:
    lower = text.lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*percent", lower)
    if match:
        return float(match.group(1))

    match = re.search(r"([a-z]+)_percent", lower)
    if match and match.group(1) in WORD_NUMBERS:
        return float(WORD_NUMBERS[match.group(1)])

    return None


def format_noise_label(percent: float) -> str:
    if float(percent).is_integer():
        return f"{int(percent)}%"
    return f"{percent:g}%"


def expected_gfli_paths(base_dir: Path) -> tuple[list[str], list[str]]:
    train_paths = [str(base_dir / f"gfli{i}_impedance_dataset.mat") for i in range(1, 10)]
    test_paths = [str(base_dir / f"gfli{i}_test_impedance_dataset.mat") for i in range(1, 10)]
    return train_paths, test_paths


def paths_exist(paths: list[str]) -> bool:
    return all(Path(path).exists() for path in paths)


def discover_folder_conditions(root: Path) -> list[NoiseCondition]:
    conditions = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if "differentibrstructure" in name:
            continue
        if "percent_noise" not in name:
            continue

        percent = parse_noise_percent(name)
        if percent is None:
            continue

        train_paths, test_paths = expected_gfli_paths(folder)
        if paths_exist(train_paths) and paths_exist(test_paths):
            conditions.append(
                NoiseCondition(
                    label=format_noise_label(percent),
                    percent=percent,
                    train_paths=train_paths,
                    test_paths=test_paths,
                )
            )
    return conditions


def discover_filename_conditions(root: Path) -> list[NoiseCondition]:
    grouped = {}
    pattern = re.compile(
        r"^(gfli\d+)_(test_)?impedance_dataset_(.+percent)_noise\.mat$",
        re.IGNORECASE,
    )

    for path in root.glob("gfli*_impedance_dataset_*percent_noise.mat"):
        match = pattern.match(path.name)
        if not match:
            continue
        percent = parse_noise_percent(match.group(3))
        if percent is None:
            continue
        entry = grouped.setdefault(percent, {"train": {}, "test": {}})
        ibr = match.group(1).lower()
        is_test = bool(match.group(2))
        entry["test" if is_test else "train"][ibr] = str(path)

    conditions = []
    for percent, files in grouped.items():
        train_paths, test_paths = [], []
        complete = True
        for i in range(1, 10):
            ibr = f"gfli{i}"
            if ibr not in files["train"] or ibr not in files["test"]:
                complete = False
                break
            train_paths.append(files["train"][ibr])
            test_paths.append(files["test"][ibr])
        if complete:
            conditions.append(
                NoiseCondition(
                    label=format_noise_label(percent),
                    percent=percent,
                    train_paths=train_paths,
                    test_paths=test_paths,
                )
            )
    return conditions


def discover_conditions(root: Path, include_clean: bool) -> list[NoiseCondition]:
    conditions = []
    if include_clean:
        clean_train = [str(root / path) for path in TRAIN_FILE_PATHS]
        clean_test = [str(root / path) for path in TEST_FILE_PATHS]
        if paths_exist(clean_train) and paths_exist(clean_test):
            conditions.append(
                NoiseCondition(
                    label="0%",
                    percent=0.0,
                    train_paths=clean_train,
                    test_paths=clean_test,
                )
            )

    found = discover_folder_conditions(root) + discover_filename_conditions(root)
    by_percent = {condition.percent: condition for condition in conditions}
    for condition in found:
        by_percent.setdefault(condition.percent, condition)
    return sorted(by_percent.values(), key=lambda item: item.percent)


def run_one_noise_condition(condition: NoiseCondition, args, device: torch.device):
    set_seed(args.seed)
    clients, test_sets, input_dim, output_dim, _, y_scaler = load_scenario_clients(
        condition.train_paths,
        condition.test_paths,
        LABEL_KEY,
    )

    global_model = FNN(input_dim, args.hidden_dim, output_dim).to(device)
    global_state = clone_state_dict(global_model)
    rng = random.Random(args.seed)
    history_rows = []

    print(f"\n===== Scenario F | noise={condition.label} =====")
    print(f"Loaded {len(clients)} clients and {len(test_sets)} test sets.")

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
        test_scaled_score_percent = mse_to_accuracy_percent(test_mse)

        test_physical_eval = evaluate_state_physical(
            global_state,
            test_sets,
            input_dim,
            args.hidden_dim,
            output_dim,
            args.eval_batch_size,
            device,
            y_scaler,
        )
        test_physical_rmse_mean = float(np.mean([row["physical_rmse"] for row in test_physical_eval]))
        test_physical_rel_error_percent = float(
            np.mean([row["physical_rel_error_percent"] for row in test_physical_eval])
        )
        test_physical_accuracy_percent = rel_error_to_accuracy_percent(test_physical_rel_error_percent)
        test_physical_nrmse_rel_error_percent = float(
            np.mean([row["physical_nrmse_rel_error_percent"] for row in test_physical_eval])
        )
        test_physical_nrmse_accuracy_percent = rel_error_to_accuracy_percent(
            test_physical_nrmse_rel_error_percent
        )
        test_physical_real_rmse_mean = float(np.mean([row["physical_real_rmse"] for row in test_physical_eval]))
        test_physical_real_rel_error_percent = float(
            np.mean([row["physical_real_rel_error_percent"] for row in test_physical_eval])
        )
        test_physical_real_accuracy_percent = rel_error_to_accuracy_percent(
            test_physical_real_rel_error_percent
        )
        test_physical_imag_rmse_mean = float(np.mean([row["physical_imag_rmse"] for row in test_physical_eval]))
        test_physical_imag_rel_error_percent = float(
            np.mean([row["physical_imag_rel_error_percent"] for row in test_physical_eval])
        )
        test_physical_imag_accuracy_percent = rel_error_to_accuracy_percent(
            test_physical_imag_rel_error_percent
        )

        history_rows.append(
            {
                "noise_label": condition.label,
                "noise_percent": condition.percent,
                "round": round_idx,
                "local_train_mse_mean": float(np.mean(local_losses)),
                "train_mse_mean": train_mse,
                "test_mse_mean": test_mse,
                "test_scaled_mse_mean": test_mse,
                "test_scaled_score_percent": test_scaled_score_percent,
                "test_accuracy_percent": test_physical_accuracy_percent,
                "test_physical_rmse_mean": test_physical_rmse_mean,
                "test_physical_rel_error_percent": test_physical_rel_error_percent,
                "test_physical_accuracy_percent": test_physical_accuracy_percent,
                "test_physical_nrmse_rel_error_percent": test_physical_nrmse_rel_error_percent,
                "test_physical_nrmse_accuracy_percent": test_physical_nrmse_accuracy_percent,
                "test_physical_real_rmse_mean": test_physical_real_rmse_mean,
                "test_physical_real_rel_error_percent": test_physical_real_rel_error_percent,
                "test_physical_real_accuracy_percent": test_physical_real_accuracy_percent,
                "test_physical_imag_rmse_mean": test_physical_imag_rmse_mean,
                "test_physical_imag_rel_error_percent": test_physical_imag_rel_error_percent,
                "test_physical_imag_accuracy_percent": test_physical_imag_accuracy_percent,
            }
        )

        if round_idx == 1 or round_idx == args.rounds or round_idx % args.log_every == 0:
            print(
                f"[FedAvg noise={condition.label}] round={round_idx:03d} "
                f"train={train_mse:.4e} test={test_mse:.4e} "
                f"phys_rel_err={test_physical_rel_error_percent:.2f}% "
                f"phys_acc={test_physical_accuracy_percent:.2f}% "
                f"nrmse_acc={test_physical_nrmse_accuracy_percent:.2f}% "
                f"real_acc={test_physical_real_accuracy_percent:.2f}% "
                f"imag_acc={test_physical_imag_accuracy_percent:.2f}%"
            )

    final_test_scaled = pd.DataFrame(
        evaluate_state(global_state, test_sets, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device)
    )
    final_test_scaled = final_test_scaled.rename(columns={"mse": "scaled_mse"})
    final_test_scaled["mse"] = final_test_scaled["scaled_mse"]

    final_test_physical = pd.DataFrame(
        evaluate_state_physical(
            global_state,
            test_sets,
            input_dim,
            args.hidden_dim,
            output_dim,
            args.eval_batch_size,
            device,
            y_scaler,
        )
    ).drop(columns=["n"])
    final_test = final_test_scaled.merge(final_test_physical, on="name", how="left")
    final_test["physical_accuracy_percent"] = final_test["physical_rel_error_percent"].map(
        rel_error_to_accuracy_percent
    )
    final_test["noise_label"] = condition.label
    final_test["noise_percent"] = condition.percent

    return pd.DataFrame(history_rows), final_test, deepcopy(global_state)


def plot_final_mse(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(
        summary_df["noise_percent"],
        summary_df["final_test_mse"],
        marker="o",
        linewidth=1.6,
        color="#1f77b4",
    )
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Final scaled macro test MSE")
    ax.set_title("Scenario F: Final Scaled Test MSE vs Noise Level")
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    savefig_no_crash(fig, output_dir / "fig_scenario_f_final_test_mse_vs_noise.pdf")
    savefig_no_crash(fig, output_dir / "fig_scenario_f_final_test_mse_vs_noise.svg")
    plt.close(fig)


def plot_final_physical_accuracy(summary_df: pd.DataFrame, output_dir: Path) -> None:
    required = {
        "final_test_physical_nrmse_accuracy_percent",
        "final_test_physical_real_accuracy_percent",
        "final_test_physical_imag_accuracy_percent",
    }
    if not required.issubset(summary_df.columns):
        return

    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(
        summary_df["noise_percent"],
        summary_df["final_test_physical_nrmse_accuracy_percent"],
        marker="o",
        linewidth=1.5,
        label="Complex",
    )
    ax.plot(
        summary_df["noise_percent"],
        summary_df["final_test_physical_real_accuracy_percent"],
        marker="s",
        linewidth=1.5,
        label="Real",
    )
    ax.plot(
        summary_df["noise_percent"],
        summary_df["final_test_physical_imag_accuracy_percent"],
        marker="^",
        linewidth=1.5,
        label="Imag",
    )
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel("Final physical NRMSE accuracy (%)")
    ax.set_title("Scenario F: Final Physical Accuracy vs Noise Level")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    savefig_no_crash(fig, output_dir / "fig_scenario_f_final_physical_accuracy_vs_noise.pdf")
    savefig_no_crash(fig, output_dir / "fig_scenario_f_final_physical_accuracy_vs_noise.svg")
    plt.close(fig)


def plot_learning_curves(history_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    has_physical_accuracy = "test_physical_accuracy_percent" in history_df.columns
    accuracy_col = "test_physical_accuracy_percent" if has_physical_accuracy else "test_accuracy_percent"
    for _, group in history_df.groupby("noise_percent", sort=True):
        label = group["noise_label"].iloc[0]
        ax.plot(
            group["round"],
            group[accuracy_col],
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Physical prediction accuracy (%)" if has_physical_accuracy else "Scaled-MSE score (%)")
    ax.set_title("FedAvgIm learning curves across noise levels")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(
    title="Noise",
    frameon=False,
    loc="lower right",
    fontsize=8,
    title_fontsize=8,
    )
    fig.tight_layout()
    savefig_no_crash(fig, output_dir / "fig_scenario_f_learning_curves_noise.pdf")
    savefig_no_crash(fig, output_dir / "fig_scenario_f_learning_curves_noise.svg")
    plt.close(fig)


def plot_per_ibr_mse(final_per_ibr_df: pd.DataFrame, output_dir: Path) -> None:
    pivot = final_per_ibr_df.pivot(index="name", columns="noise_label", values="mse")
    ordered_cols = (
        final_per_ibr_df[["noise_label", "noise_percent"]]
        .drop_duplicates()
        .sort_values("noise_percent")["noise_label"]
        .tolist()
    )
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(5.2, 0.65 * len(ordered_cols) + 2.5), 3.2))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([name.replace("gfli", "IBR") for name in pivot.index])
    ax.set_xlabel("Noise level")
    ax.set_ylabel("IBR")
    ax.set_title("Scenario F: Per-IBR Final Scaled Test MSE")

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(
                col,
                row,
                f"{pivot.values[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if pivot.values[row, col] > np.nanmean(pivot.values) else "black",
                fontsize=7,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Final scaled test MSE")
    fig.tight_layout()
    savefig_no_crash(fig, output_dir / "fig_scenario_f_per_ibr_final_test_mse.pdf")
    savefig_no_crash(fig, output_dir / "fig_scenario_f_per_ibr_final_test_mse.svg")
    plt.close(fig)


def run_scenario_f(args) -> None:
    apply_paper_style()

    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(root, include_clean=not args.no_clean)
    if not args.all_noise_levels:
        wanted = {float(item) for item in (args.noise_levels or SCENARIO_F_NOISE_LEVELS)}
        conditions = [condition for condition in conditions if condition.percent in wanted]

    if not conditions:
        raise RuntimeError(
            "No complete noise conditions found. Expected folders like "
            "'one_percent_noise'/'five_percent_noise' or filenames like "
            "'gfli1_impedance_dataset_1percent_noise.mat'."
        )

    print("Scenario F conditions:")
    for condition in conditions:
        print(f"  noise={condition.label} train_files={len(condition.train_paths)} test_files={len(condition.test_paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    histories, final_rows, summary_rows = [], [], []
    for condition in conditions:
        history_df, final_test_df, _ = run_one_noise_condition(condition, args, device)
        histories.append(history_df)
        final_rows.append(final_test_df)
        summary_rows.append(
            {
                "noise_label": condition.label,
                "noise_percent": condition.percent,
                "final_train_mse": history_df["train_mse_mean"].iloc[-1],
                "final_test_mse": history_df["test_mse_mean"].iloc[-1],
                "final_test_scaled_mse": history_df["test_scaled_mse_mean"].iloc[-1],
                "final_test_scaled_score_percent": history_df["test_scaled_score_percent"].iloc[-1],
                "final_test_physical_rmse": history_df["test_physical_rmse_mean"].iloc[-1],
                "final_test_physical_rel_error_percent": history_df[
                    "test_physical_rel_error_percent"
                ].iloc[-1],
                "final_test_physical_accuracy_percent": history_df[
                    "test_physical_accuracy_percent"
                ].iloc[-1],
                "final_test_physical_nrmse_rel_error_percent": history_df[
                    "test_physical_nrmse_rel_error_percent"
                ].iloc[-1],
                "final_test_physical_nrmse_accuracy_percent": history_df[
                    "test_physical_nrmse_accuracy_percent"
                ].iloc[-1],
                "final_test_physical_real_rmse": history_df["test_physical_real_rmse_mean"].iloc[-1],
                "final_test_physical_real_rel_error_percent": history_df[
                    "test_physical_real_rel_error_percent"
                ].iloc[-1],
                "final_test_physical_real_accuracy_percent": history_df[
                    "test_physical_real_accuracy_percent"
                ].iloc[-1],
                "final_test_physical_imag_rmse": history_df["test_physical_imag_rmse_mean"].iloc[-1],
                "final_test_physical_imag_rel_error_percent": history_df[
                    "test_physical_imag_rel_error_percent"
                ].iloc[-1],
                "final_test_physical_imag_accuracy_percent": history_df[
                    "test_physical_imag_accuracy_percent"
                ].iloc[-1],
                "final_test_accuracy_percent": history_df["test_physical_accuracy_percent"].iloc[-1],
                "auc_test_mse": float(np.trapezoid(history_df["test_mse_mean"], history_df["round"])),
                "auc_test_physical_rel_error_percent": float(
                    np.trapezoid(history_df["test_physical_rel_error_percent"], history_df["round"])
                ),
                "auc_test_physical_nrmse_rel_error_percent": float(
                    np.trapezoid(history_df["test_physical_nrmse_rel_error_percent"], history_df["round"])
                ),
                "auc_test_physical_real_rel_error_percent": float(
                    np.trapezoid(history_df["test_physical_real_rel_error_percent"], history_df["round"])
                ),
                "auc_test_physical_imag_rel_error_percent": float(
                    np.trapezoid(history_df["test_physical_imag_rel_error_percent"], history_df["round"])
                ),
            }
        )

    history_all = pd.concat(histories, ignore_index=True)
    final_per_ibr = pd.concat(final_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("noise_percent")

    history_all.to_csv(output_dir / "scenario_f_noise_fedavg_history.csv", index=False)
    final_per_ibr.to_csv(output_dir / "scenario_f_per_ibr_final_test_mse.csv", index=False)
    summary.to_csv(output_dir / "scenario_f_noise_summary.csv", index=False)

    plot_final_mse(summary, output_dir)
    plot_final_physical_accuracy(summary, output_dir)
    plot_learning_curves(history_all, output_dir)
    plot_per_ibr_mse(final_per_ibr, output_dir)

    print("\nScenario F summary:")
    print(summary.to_string(index=False))
    final_metrics = summary[
        [
            "noise_label",
            "noise_percent",
            "final_test_mse",
            "final_test_physical_rel_error_percent",
            "final_test_physical_accuracy_percent",
            "final_test_physical_nrmse_accuracy_percent",
            "final_test_physical_real_accuracy_percent",
            "final_test_physical_imag_accuracy_percent",
        ]
    ].copy()
    final_metrics["final_test_physical_rel_error_percent"] = final_metrics[
        "final_test_physical_rel_error_percent"
    ].map(lambda value: f"{value:.2f}%")
    final_metrics["final_test_physical_accuracy_percent"] = final_metrics[
        "final_test_physical_accuracy_percent"
    ].map(
        lambda value: f"{value:.2f}%"
    )
    final_metrics["final_test_physical_nrmse_accuracy_percent"] = final_metrics[
        "final_test_physical_nrmse_accuracy_percent"
    ].map(lambda value: f"{value:.2f}%")
    final_metrics["final_test_physical_real_accuracy_percent"] = final_metrics[
        "final_test_physical_real_accuracy_percent"
    ].map(lambda value: f"{value:.2f}%")
    final_metrics["final_test_physical_imag_accuracy_percent"] = final_metrics[
        "final_test_physical_imag_accuracy_percent"
    ].map(lambda value: f"{value:.2f}%")
    print("\nScenario F final test metrics:")
    print(final_metrics.to_string(index=False))
    print(f"\nSaved Scenario F results to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Scenario F: FedAvg robustness across noisy data levels.")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="scenario_f_noise_fedavg_results")
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--local-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="*",
        default=None,
        help="Noise levels to run. Default: 0 1 2 3.",
    )
    parser.add_argument(
        "--all-noise-levels",
        action="store_true",
        help="Run every discovered noise level instead of the default 0, 1, 2, and 3 percent conditions.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Exclude the clean root-level datasets as 0% noise.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_scenario_f(parse_args())
