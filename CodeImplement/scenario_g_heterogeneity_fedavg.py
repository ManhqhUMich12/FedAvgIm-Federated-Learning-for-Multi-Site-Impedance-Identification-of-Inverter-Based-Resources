# -*- coding: utf-8 -*-
"""
Scenario G: FedAvg effectiveness versus output-space heterogeneity.

This script compares multiple IBR admittance cases ordered by heterogeneity.
For each case it:
  1. Computes within-case output heterogeneity H_Y from Y_Y only.
  2. Trains the same FedAvg FNN framework used by Scenario A/E.
  3. Optionally trains local-only FNN baselines for each client.
  4. Exports summary CSVs and figures linking heterogeneity to performance.

H_Y uses only the marginal output/admittance distributions and does not assume
that clients share identical operating points or frequency samples.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fl_scenario_ae_fedavg import (
    FNN,
    LABEL_KEY,
    aggregate_fedavg,
    clone_state_dict,
    evaluate_model,
    evaluate_state,
    get_lr,
    load_scenario_clients,
    select_clients,
    set_seed,
    train_client_sgd,
)
from quantify_y_y_heterogeneity import (
    compute_hy_results,
    fit_global_y_scaler,
    load_scenario_clients as load_hy_scenario_clients,
    normalize_against_baseline,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_SCENARIOS = [
    ("G1", "zero_percent_noise"),
    ("G2", "one_percent_noise"),
    ("G3", "differentIBRstructure_IBR9_weak"),
    ("G4", "differentIBRstructure_IBR4_strong"),
]
IEEE_FIGSIZE = (3.5, 1.8)


plt.rcParams.update(
    {
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
        "lines.markersize": 3.2,
    }
)


@dataclass
class ScenarioSpec:
    name: str
    folder: Path
    description: str = ""


@dataclass
class ScenarioTrainingResult:
    scenario: str
    folder: Path
    fedavg_history: pd.DataFrame
    fedavg_final_train: pd.DataFrame
    fedavg_final_test: pd.DataFrame
    local_results: pd.DataFrame
    per_client_results: pd.DataFrame


def parse_scenario_item(item: str) -> ScenarioSpec:
    """Parse name=folder or name=folder:description."""
    if "=" not in item:
        raise ValueError(f"Invalid --scenario value {item!r}. Expected name=folder.")
    name, rest = item.split("=", 1)
    description = ""
    folder = rest
    if ":" in rest:
        folder, description = rest.split(":", 1)
    name = name.strip()
    folder = folder.strip()
    description = description.strip()
    if not name or not folder:
        raise ValueError(f"Invalid --scenario value {item!r}. Expected name=folder.")
    return ScenarioSpec(name=name, folder=Path(folder), description=description)


def parse_scenarios(args: argparse.Namespace) -> list[ScenarioSpec]:
    if args.scenario:
        scenarios = [parse_scenario_item(item) for item in args.scenario]
    else:
        scenarios = [ScenarioSpec(name=name, folder=Path(folder)) for name, folder in DEFAULT_SCENARIOS]

    seen = set()
    for scenario in scenarios:
        if scenario.name in seen:
            raise ValueError(f"Duplicate scenario name: {scenario.name}")
        seen.add(scenario.name)
    return scenarios


def expected_gfli_paths(folder: Path) -> tuple[list[str], list[str]]:
    train_paths = [str(folder / f"gfli{i}_impedance_dataset.mat") for i in range(1, 10)]
    test_paths = [str(folder / f"gfli{i}_test_impedance_dataset.mat") for i in range(1, 10)]
    return train_paths, test_paths


def metric_or_nan(summary: dict[str, float] | None, key: str) -> float:
    if not summary:
        return float("nan")
    return float(summary.get(key, float("nan")))


def build_heterogeneity_summary(
    hy_results,
    scenario_specs: list[ScenarioSpec],
    baseline_name: str,
) -> pd.DataFrame:
    description_by_name = {item.name: item.description for item in scenario_specs}
    rows = []
    for result in hy_results:
        total_samples = sum(client.y.shape[0] for client in result.clients)
        rows.append(
            {
                "scenario": result.scenario,
                "description": description_by_name.get(result.scenario, ""),
                "folder": str(result.folder),
                "n_clients": len(result.clients),
                "total_samples": total_samples,
                "H_Y_stat_mean": result.stat_summary["mean"],
                "H_Y_stat_median": result.stat_summary["median"],
                "H_Y_stat_std": result.stat_summary["std"],
                "H_Y_stat_min": result.stat_summary["min"],
                "H_Y_stat_max": result.stat_summary["max"],
                "H_Y_stat_p90": result.stat_summary["p90"],
                "relative_H_Y_stat": result.relative_stat,
                "H_Y_stat_increase_percent": result.stat_increase_percent,
                "H_Y_MMD_mean": metric_or_nan(result.mmd_summary, "mean"),
                "H_Y_MMD_median": metric_or_nan(result.mmd_summary, "median"),
                "H_Y_MMD_std": metric_or_nan(result.mmd_summary, "std"),
                "H_Y_MMD_min": metric_or_nan(result.mmd_summary, "min"),
                "H_Y_MMD_max": metric_or_nan(result.mmd_summary, "max"),
                "H_Y_MMD_p90": metric_or_nan(result.mmd_summary, "p90"),
                "relative_H_Y_MMD": result.relative_mmd,
                "H_Y_MMD_increase_percent": result.mmd_increase_percent,
                "baseline_scenario": baseline_name,
            }
        )
    return pd.DataFrame(rows)


def train_local_fnn(
    client,
    test_set,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    optimizer_name: str,
    device: torch.device,
) -> dict[str, float]:
    model = FNN(input_dim, hidden_dim, output_dim).to(device)
    loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_curve = []
    test_curve = []
    for _ in range(epochs):
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
        train_curve.append(total_loss / max(total_n, 1))
        test_curve.append(evaluate_model(model, test_set.dataset, eval_batch_size, device))

    test_arr = np.asarray(test_curve, dtype=float)
    best_idx = int(np.argmin(test_arr)) if test_arr.size else -1
    return {
        "local_final_train_mse": float(train_curve[-1]) if train_curve else float("nan"),
        "local_final_test_mse": float(test_curve[-1]) if test_curve else float("nan"),
        "local_best_test_mse": float(test_arr[best_idx]) if best_idx >= 0 else float("nan"),
        "local_best_epoch": float(best_idx + 1) if best_idx >= 0 else float("nan"),
    }


def run_fedavg_for_scenario(
    scenario: ScenarioSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> ScenarioTrainingResult:
    train_paths, test_paths = expected_gfli_paths(scenario.folder)
    clients, test_sets, input_dim, output_dim, _, _ = load_scenario_clients(
        train_paths, test_paths, args.label_key
    )

    print(f"\n===== Scenario {scenario.name}: FedAvg =====")
    print(f"Folder: {scenario.folder}")
    print(f"Loaded {len(clients)} train clients and {len(test_sets)} test sets.")

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
        train_eval = evaluate_state(
            global_state, clients, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device
        )
        test_eval = evaluate_state(
            global_state, test_sets, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device
        )
        train_macro_mse = float(np.mean([row["mse"] for row in train_eval]))
        test_macro_mse = float(np.mean([row["mse"] for row in test_eval]))
        test_values = np.asarray([row["mse"] for row in test_eval], dtype=float)

        history_rows.append(
            {
                "scenario": scenario.name,
                "round": round_idx,
                "lr": lr,
                "selected_clients": len(selected),
                "local_train_mse_mean": float(np.mean(local_losses)),
                "train_macro_mse": train_macro_mse,
                "test_macro_mse": test_macro_mse,
                "test_mse_std_across_clients": float(np.std(test_values, ddof=0)),
                "test_mse_worst_client": float(np.max(test_values)),
                "test_mse_best_client": float(np.min(test_values)),
                "test_mse_p90": float(np.percentile(test_values, 90)),
            }
        )

        if round_idx == 1 or round_idx == args.rounds or round_idx % args.log_every == 0:
            print(
                f"[{scenario.name} FedAvg] round={round_idx:03d} "
                f"train={train_macro_mse:.4e} test={test_macro_mse:.4e}"
            )

    fedavg_history = pd.DataFrame(history_rows)
    fedavg_final_train = pd.DataFrame(
        evaluate_state(global_state, clients, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device)
    )
    fedavg_final_test = pd.DataFrame(
        evaluate_state(global_state, test_sets, input_dim, args.hidden_dim, output_dim, args.eval_batch_size, device)
    )

    local_rows = []
    if not args.skip_local_baseline:
        local_epochs = args.local_baseline_epochs if args.local_baseline_epochs > 0 else args.rounds
        print(f"\n===== Scenario {scenario.name}: Local-only FNN ({local_epochs} epochs) =====")
        for client, test_set in zip(clients, test_sets):
            row = train_local_fnn(
                client=client,
                test_set=test_set,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                epochs=local_epochs,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                lr=args.lr,
                optimizer_name=args.optimizer,
                device=device,
            )
            row.update({"scenario": scenario.name, "client": client.name})
            local_rows.append(row)
            print(
                f"[{scenario.name} Local] client={client.name} "
                f"test={row['local_final_test_mse']:.4e}"
            )
    local_results = pd.DataFrame(local_rows)

    per_client = fedavg_final_test.rename(columns={"name": "client", "mse": "fedavg_final_test_mse", "n": "n_test"})
    final_train = fedavg_final_train.rename(
        columns={"name": "client", "mse": "fedavg_final_train_mse", "n": "n_train"}
    )
    per_client = per_client.merge(final_train, on="client", how="left")
    per_client["scenario"] = scenario.name
    per_client["folder"] = str(scenario.folder)
    if not local_results.empty:
        per_client = per_client.merge(local_results, on=["scenario", "client"], how="left")
    if "local_final_test_mse" in per_client:
        per_client["fedavg_minus_local_test_mse"] = (
            per_client["fedavg_final_test_mse"] - per_client["local_final_test_mse"]
        )
        per_client["fedavg_improvement_percent_vs_local"] = (
            (per_client["local_final_test_mse"] - per_client["fedavg_final_test_mse"])
            / per_client["local_final_test_mse"]
            * 100.0
        )

    return ScenarioTrainingResult(
        scenario=scenario.name,
        folder=scenario.folder,
        fedavg_history=fedavg_history,
        fedavg_final_train=fedavg_final_train,
        fedavg_final_test=fedavg_final_test,
        local_results=local_results,
        per_client_results=per_client,
    )


def build_fedavg_summary(
    training_results: list[ScenarioTrainingResult],
    heterogeneity_summary: pd.DataFrame,
    target_mse: float | None,
) -> pd.DataFrame:
    hetero_by_scenario = heterogeneity_summary.set_index("scenario")
    rows = []
    for result in training_results:
        history = result.fedavg_history
        final_test = result.fedavg_final_test["mse"].to_numpy(dtype=float)
        final_train = result.fedavg_final_train["mse"].to_numpy(dtype=float)
        final_row = history.iloc[-1]
        rounds = history["round"].to_numpy(dtype=float)
        test_curve = history["test_macro_mse"].to_numpy(dtype=float)
        if len(rounds) > 1:
            test_auc = float(np.trapz(test_curve, rounds) / (rounds[-1] - rounds[0]))
        else:
            test_auc = float(test_curve[-1])

        round_to_target = float("nan")
        if target_mse is not None:
            reached = history.loc[history["test_macro_mse"] <= target_mse, "round"]
            if not reached.empty:
                round_to_target = float(reached.iloc[0])

        local_mean = float("nan")
        fedavg_improvement = float("nan")
        if not result.local_results.empty:
            local_mean = float(result.local_results["local_final_test_mse"].mean())
            fedavg_improvement = (local_mean - float(final_row["test_macro_mse"])) / local_mean * 100.0

        hetero = hetero_by_scenario.loc[result.scenario]
        rows.append(
            {
                "scenario": result.scenario,
                "folder": str(result.folder),
                "H_Y_stat_mean": hetero["H_Y_stat_mean"],
                "H_Y_stat_increase_percent": hetero["H_Y_stat_increase_percent"],
                "H_Y_MMD_mean": hetero.get("H_Y_MMD_mean", float("nan")),
                "H_Y_MMD_increase_percent": hetero.get("H_Y_MMD_increase_percent", float("nan")),
                "final_train_mse_mean": float(np.mean(final_train)),
                "final_test_mse_mean": float(final_row["test_macro_mse"]),
                "final_test_mse_std_across_clients": float(np.std(final_test, ddof=0)),
                "final_test_mse_worst_client": float(np.max(final_test)),
                "final_test_mse_best_client": float(np.min(final_test)),
                "final_test_mse_p90": float(np.percentile(final_test, 90)),
                "final_test_mse_max_min_gap": float(np.max(final_test) - np.min(final_test)),
                "round_to_target_mse": round_to_target,
                "test_mse_auc_over_rounds": test_auc,
                "local_final_test_mse_mean": local_mean,
                "fedavg_improvement_percent_vs_local_mean": fedavg_improvement,
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    heterogeneity_summary: pd.DataFrame,
    hy_results,
    fedavg_summary: pd.DataFrame,
    training_results: list[ScenarioTrainingResult],
    different_topology_clients: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heterogeneity_summary.to_csv(output_dir / "scenario_g_heterogeneity_summary.csv", index=False)
    fedavg_summary.to_csv(output_dir / "scenario_g_fedavg_summary.csv", index=False)
    for result in hy_results:
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in result.scenario)
        result.pairwise_stat.to_csv(output_dir / f"{safe_name}_pairwise_DY_stat.csv")
        if result.pairwise_mmd is not None:
            result.pairwise_mmd.to_csv(output_dir / f"{safe_name}_pairwise_MMD_Y.csv")

    learning_curves = pd.concat([item.fedavg_history for item in training_results], ignore_index=True)
    learning_curves = learning_curves.merge(
        heterogeneity_summary[
            [
                "scenario",
                "H_Y_stat_mean",
                "H_Y_stat_increase_percent",
                "H_Y_MMD_mean",
                "H_Y_MMD_increase_percent",
            ]
        ],
        on="scenario",
        how="left",
    )
    learning_curves.to_csv(output_dir / "scenario_g_learning_curves.csv", index=False)

    per_client = pd.concat([item.per_client_results for item in training_results], ignore_index=True)
    if different_topology_clients:
        per_client["topology_group"] = np.where(
            per_client["client"].isin(different_topology_clients), "different", "normal"
        )
    else:
        per_client["topology_group"] = ""
    per_client.to_csv(output_dir / "scenario_g_per_client_results.csv", index=False)
    return learning_curves, per_client


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def has_local_final_test(per_client: pd.DataFrame) -> bool:
    return (
        "local_final_test_mse" in per_client
        and not per_client["local_final_test_mse"].isna().all()
    )


def plot_fedavg_vs_local_final_bars(
    plots_dir: Path,
    fedavg_summary: pd.DataFrame,
    per_client: pd.DataFrame,
) -> None:
    """Grouped final-test-MSE bars for FedAvg and Local-only baselines."""
    if not has_local_final_test(per_client):
        return

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    scenarios = fedavg_summary["scenario"].tolist()
    x = np.arange(len(scenarios))
    width = 0.36
    ax.bar(
        x - width / 2,
        fedavg_summary["final_test_mse_mean"],
        width,
        label="FedAvg",
        color="#4C72B0",
    )
    ax.bar(
        x + width / 2,
        fedavg_summary["local_final_test_mse_mean"],
        width,
        label="Local-only",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Final test MSE")
    ax.set_title("FedAvg vs Local-only")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.2, 1.05),
    )
    save_fig(fig, plots_dir / "fig_g7_case_final_test_mse_fedavg_vs_local_bars.png")

    for scenario, group in per_client.groupby("scenario", sort=False):
        group = group.sort_values("client")
        labels = group["client"].astype(str).tolist()
        x = np.arange(len(group))
        fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
        ax.bar(
            x - width / 2,
            group["fedavg_final_test_mse"],
            width,
            label="FedAvg",
            color="#4C72B0",
        )
        ax.bar(
            x + width / 2,
            group["local_final_test_mse"],
            width,
            label="Local-only",
            color="#C44E52",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Final test MSE")
        ax.set_title(f"{scenario}: final test MSE")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(scenario))
        save_fig(fig, plots_dir / f"fig_g8_{safe_name}_client_final_test_mse_fedavg_vs_local_bars.png")


def make_plots(
    output_dir: Path,
    heterogeneity_summary: pd.DataFrame,
    fedavg_summary: pd.DataFrame,
    learning_curves: pd.DataFrame,
    per_client: pd.DataFrame,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    scenarios = heterogeneity_summary["scenario"].tolist()
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    width = 0.35
    ax.bar(x - width / 2, heterogeneity_summary["H_Y_stat_mean"], width, label="H_Y_stat")
    if not heterogeneity_summary["H_Y_MMD_mean"].isna().all():
        ax.bar(x + width / 2, heterogeneity_summary["H_Y_MMD_mean"], width, label="H_Y_MMD")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel("Heterogeneity")
    ax.set_title("Scenario G heterogeneity levels")
    ax.legend()
    save_fig(fig, plots_dir / "fig_g1_heterogeneity_levels.png")

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    for scenario, group in learning_curves.groupby("scenario"):
        ax.plot(group["round"], group["test_macro_mse"], label=scenario)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("FedAvg test MSE")
    ax.set_title("FedAvg learning curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, plots_dir / "fig_g2_learning_curves.png")

    hetero_x = (
        fedavg_summary["H_Y_MMD_mean"]
        if "H_Y_MMD_mean" in fedavg_summary and not fedavg_summary["H_Y_MMD_mean"].isna().all()
        else fedavg_summary["H_Y_stat_mean"]
    )
    hetero_label = "H_Y_MMD" if not fedavg_summary["H_Y_MMD_mean"].isna().all() else "H_Y_stat"

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    ax.scatter(hetero_x, fedavg_summary["final_test_mse_mean"], s=50)
    for _, row in fedavg_summary.iterrows():
        ax.annotate(row["scenario"], (row[hetero_label + "_mean"], row["final_test_mse_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(hetero_label)
    ax.set_ylabel("Final FedAvg test MSE")
    ax.set_title("Final MSE vs heterogeneity")
    ax.grid(True, alpha=0.3)
    save_fig(fig, plots_dir / "fig_g3_final_mse_vs_HY_MMD.png")

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    ax.scatter(hetero_x, fedavg_summary["final_test_mse_worst_client"], s=50, color="#C44E52")
    for _, row in fedavg_summary.iterrows():
        ax.annotate(row["scenario"], (row[hetero_label + "_mean"], row["final_test_mse_worst_client"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(hetero_label)
    ax.set_ylabel("Worst-client final test MSE")
    ax.set_title("Worst-client MSE vs heterogeneity")
    ax.grid(True, alpha=0.3)
    save_fig(fig, plots_dir / "fig_g4_worst_client_vs_HY_MMD.png")

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    ax.bar(fedavg_summary["scenario"], fedavg_summary["final_test_mse_std_across_clients"], color="#55A868")
    ax.set_ylabel("Std of client test MSE")
    ax.set_title("Fairness degradation vs heterogeneity")
    ax.tick_params(axis="x", rotation=15)
    save_fig(fig, plots_dir / "fig_g5_fairness_vs_heterogeneity.png")

    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    labels = per_client["scenario"] + "-" + per_client["client"]
    ax.bar(labels, per_client["fedavg_final_test_mse"], label="FedAvg")
    if "local_final_test_mse" in per_client and not per_client["local_final_test_mse"].isna().all():
        ax.plot(labels, per_client["local_final_test_mse"], color="#C44E52", marker="o", linestyle="", label="Local FNN")
    ax.set_ylabel("Final test MSE")
    ax.set_title("Per-client final test MSE")
    ax.tick_params(axis="x", rotation=90, labelsize=4.5)
    ax.legend()
    save_fig(fig, plots_dir / "fig_g6_per_client_mse_bars.png")

    plot_fedavg_vs_local_final_bars(plots_dir, fedavg_summary, per_client)


def parse_client_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    clients = {item.strip().lower() for item in value.split(",") if item.strip()}
    return clients or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scenario G: FedAvg performance across multiple H_Y heterogeneity levels."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable case definition: name=folder or name=folder:description.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline scenario name. Default: first scenario.")
    parser.add_argument("--label-key", default=LABEL_KEY)
    parser.add_argument("--output-dir", type=Path, default=Path("scenario_g_results"))
    parser.add_argument("--compute-mmd-y", action="store_true", help="Compute optional H_Y_MMD.")
    parser.add_argument("--mmd-max-samples-per-client", type=int, default=1000)
    parser.add_argument("--mmd-gamma", type=float, default=None)
    parser.add_argument("--target-mse", type=float, default=None)
    parser.add_argument("--different-topology-clients", default=None)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from existing Scenario G CSVs in --output-dir.",
    )
    parser.add_argument("--skip-local-baseline", action="store_true")
    parser.add_argument("--local-baseline-epochs", type=int, default=0, help="0 means use --rounds.")

    parser.add_argument("--rounds", type=int, default=250)
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
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        heterogeneity_path = args.output_dir / "scenario_g_heterogeneity_summary.csv"
        summary_path = args.output_dir / "scenario_g_fedavg_summary.csv"
        curves_path = args.output_dir / "scenario_g_learning_curves.csv"
        per_client_path = args.output_dir / "scenario_g_per_client_results.csv"
        required_paths = [heterogeneity_path, summary_path, curves_path, per_client_path]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Plot-only mode requires existing CSV files: " + ", ".join(missing))
        heterogeneity_summary = pd.read_csv(heterogeneity_path)
        fedavg_summary = pd.read_csv(summary_path)
        learning_curves = pd.read_csv(curves_path)
        per_client = pd.read_csv(per_client_path)
        make_plots(args.output_dir, heterogeneity_summary, fedavg_summary, learning_curves, per_client)
        print(f"Regenerated Scenario G plots from existing CSVs in: {args.output_dir}")
        return

    scenario_specs = parse_scenarios(args)
    baseline_name = args.baseline or scenario_specs[0].name
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)
    print("Scenario order:", ", ".join(item.name for item in scenario_specs))
    print("Baseline:", baseline_name)

    hy_scenarios = []
    for spec in scenario_specs:
        clients = load_hy_scenario_clients(spec.folder, split="train", label_key=args.label_key)
        hy_scenarios.append(type("HYScenario", (), {"name": spec.name, "folder": spec.folder, "clients": clients})())
    scaler_y = fit_global_y_scaler(hy_scenarios)
    hy_results, gamma_y = compute_hy_results(
        hy_scenarios,
        scaler_y=scaler_y,
        compute_mmd_y=args.compute_mmd_y,
        mmd_max_samples_per_client=args.mmd_max_samples_per_client,
        mmd_gamma=args.mmd_gamma,
        random_seed=args.random_seed,
    )
    normalize_against_baseline(hy_results, baseline_name)
    if gamma_y is not None:
        print(f"H_Y_MMD gamma: {gamma_y:.6g}")

    heterogeneity_summary = build_heterogeneity_summary(hy_results, scenario_specs, baseline_name)

    training_results = []
    for spec in scenario_specs:
        training_results.append(run_fedavg_for_scenario(spec, args=args, device=device))

    fedavg_summary = build_fedavg_summary(training_results, heterogeneity_summary, args.target_mse)
    learning_curves, per_client = save_outputs(
        args.output_dir,
        heterogeneity_summary=heterogeneity_summary,
        hy_results=hy_results,
        fedavg_summary=fedavg_summary,
        training_results=training_results,
        different_topology_clients=parse_client_set(args.different_topology_clients),
    )
    if args.make_plots:
        make_plots(args.output_dir, heterogeneity_summary, fedavg_summary, learning_curves, per_client)

    print("\n===== Scenario G Summary =====")
    print(fedavg_summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
