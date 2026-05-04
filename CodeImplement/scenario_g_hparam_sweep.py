# -*- coding: utf-8 -*-
"""
Scenario G hyperparameter sweep, focused on local epochs.

The main use is to quantify client-drift sensitivity under increasing
heterogeneity:

    local_epochs = 1, 2, 4, 8, 16

For each local-epoch setting, the script runs Scenario G FedAvg over all
specified cases, aggregates the results into one table, and creates compact
IEEE-style figures. Heterogeneity H_Y is computed once from Y_Y only and then
merged into every hyperparameter result.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fl_scenario_ae_fedavg import LABEL_KEY, set_seed
from quantify_y_y_heterogeneity import (
    ScenarioData,
    compute_hy_results,
    fit_global_y_scaler,
    load_scenario_clients as load_hy_scenario_clients,
    normalize_against_baseline,
)
from scenario_g_heterogeneity_fedavg import (
    DEFAULT_SCENARIOS,
    ScenarioSpec,
    build_fedavg_summary,
    build_heterogeneity_summary,
    make_plots as make_single_run_plots,
    parse_scenario_item,
    run_fedavg_for_scenario,
    save_outputs,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IEEE_FIGSIZE = (3.5, 1.5)
COLOR_CYCLE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def configure_ieee_plots() -> None:
    """Compact IEEE-style plotting setup, matching the AE plotting style."""
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
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
        }
    )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError("All local epoch values must be positive.")
        items.append(parsed)
    if not items:
        raise ValueError("At least one local epoch value is required.")
    return items


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


def compute_global_heterogeneity(
    scenario_specs: list[ScenarioSpec],
    baseline_name: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list]:
    hy_scenarios = []
    for spec in scenario_specs:
        clients = load_hy_scenario_clients(spec.folder, split="train", label_key=args.label_key)
        hy_scenarios.append(ScenarioData(name=spec.name, folder=spec.folder, clients=clients))

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
    return build_heterogeneity_summary(hy_results, scenario_specs, baseline_name), hy_results


def local_steps_summary(per_client: pd.DataFrame, local_epochs: int, batch_size: int) -> dict[str, float]:
    if "n_train" not in per_client:
        return {
            "local_update_steps_mean": float("nan"),
            "local_update_steps_min": float("nan"),
            "local_update_steps_max": float("nan"),
        }
    steps = local_epochs * np.ceil(per_client["n_train"].to_numpy(dtype=float) / float(batch_size))
    return {
        "local_update_steps_mean": float(np.mean(steps)),
        "local_update_steps_min": float(np.min(steps)),
        "local_update_steps_max": float(np.max(steps)),
    }


def run_one_setting(
    base_args: argparse.Namespace,
    scenario_specs: list[ScenarioSpec],
    heterogeneity_summary: pd.DataFrame,
    hy_results,
    local_epochs: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_args = copy.copy(base_args)
    run_args.local_epochs = local_epochs
    run_args.skip_local_baseline = not base_args.run_local_baseline
    run_args.output_dir = base_args.output_dir / f"local_epochs_{local_epochs}"

    print(f"\n========== Sweep local_epochs={local_epochs} ==========")
    set_seed(run_args.seed)
    training_results = [
        run_fedavg_for_scenario(spec, args=run_args, device=device) for spec in scenario_specs
    ]
    fedavg_summary = build_fedavg_summary(
        training_results, heterogeneity_summary=heterogeneity_summary, target_mse=run_args.target_mse
    )
    learning_curves, per_client = save_outputs(
        run_args.output_dir,
        heterogeneity_summary=heterogeneity_summary,
        hy_results=hy_results,
        fedavg_summary=fedavg_summary,
        training_results=training_results,
        different_topology_clients=None,
    )
    if run_args.make_single_run_plots:
        make_single_run_plots(run_args.output_dir, heterogeneity_summary, fedavg_summary, learning_curves, per_client)

    for frame in (fedavg_summary, learning_curves, per_client):
        frame["local_epochs"] = local_epochs
        frame["batch_size"] = run_args.batch_size
        frame["lr"] = run_args.lr
        frame["optimizer"] = run_args.optimizer
        frame["client_fraction"] = run_args.client_fraction
        frame["rounds"] = run_args.rounds

    step_rows = []
    for scenario, group in per_client.groupby("scenario"):
        row = {"scenario": scenario, "local_epochs": local_epochs}
        row.update(local_steps_summary(group, local_epochs, run_args.batch_size))
        step_rows.append(row)
    steps_df = pd.DataFrame(step_rows)
    fedavg_summary = fedavg_summary.merge(steps_df, on=["scenario", "local_epochs"], how="left")
    return fedavg_summary, learning_curves, per_client


def plot_metric_vs_local_epochs(
    summary: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
    for idx, (scenario, group) in enumerate(summary.groupby("scenario", sort=False)):
        group = group.sort_values("local_epochs")
        ax.plot(
            group["local_epochs"],
            group[y_col],
            marker="o",
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=scenario,
        )
    ax.set_xlabel("Local epochs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(summary["local_epochs"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, path)


def make_hparam_plots(
    output_dir: Path,
    summary: pd.DataFrame,
    learning_curves: pd.DataFrame,
    per_client: pd.DataFrame | None = None,
    compare_client: str | None = None,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_vs_local_epochs(
        summary,
        "final_test_mse_mean",
        "Final test MSE",
        "Final MSE vs local epochs",
        plots_dir / "fig_g_hparam_final_mse_vs_local_epochs.png",
    )
    plot_metric_vs_local_epochs(
        summary,
        "final_test_mse_worst_client",
        "Worst-client MSE",
        "Worst client vs local epochs",
        plots_dir / "fig_g_hparam_worst_client_vs_local_epochs.png",
    )
    plot_metric_vs_local_epochs(
        summary,
        "final_test_mse_std_across_clients",
        "Client MSE std.",
        "Fairness vs local epochs",
        plots_dir / "fig_g_hparam_fairness_vs_local_epochs.png",
    )
    if "fedavg_improvement_percent_vs_local_mean" in summary and not summary[
        "fedavg_improvement_percent_vs_local_mean"
    ].isna().all():
        plot_metric_vs_local_epochs(
            summary,
            "fedavg_improvement_percent_vs_local_mean",
            "Improvement (%)",
            "FedAvg vs local FNN",
            plots_dir / "fig_g_hparam_fedavg_vs_local_vs_local_epochs.png",
        )

    for scenario, scenario_curves in learning_curves.groupby("scenario", sort=False):
        fig, ax = plt.subplots(figsize=IEEE_FIGSIZE)
        for idx, (local_epochs, group) in enumerate(scenario_curves.groupby("local_epochs")):
            group = group.sort_values("round")
            ax.plot(
                group["round"],
                group["test_macro_mse"],
                color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
                label=f"E={local_epochs}",
            )
        ax.set_xlabel("Round")
        ax.set_ylabel("Test MSE")
        ax.set_title(f"{scenario}: learning curves")
        ax.legend(frameon=False, ncol=2)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(scenario))
        save_figure(fig, plots_dir / f"fig_g_hparam_learning_curves_{safe}.png")

    plot_compare_client_fedavg_vs_local(plots_dir, per_client, compare_client)


def plot_compare_client_fedavg_vs_local(
    plots_dir: Path,
    per_client: pd.DataFrame | None,
    compare_client: str | None,
) -> None:
    if per_client is None or not compare_client:
        return
    required = {"scenario", "client", "local_epochs", "fedavg_final_test_mse", "local_final_test_mse"}
    if not required.issubset(per_client.columns):
        return

    client_key = compare_client.strip().lower()
    data = per_client.loc[per_client["client"].astype(str).str.lower() == client_key].copy()
    data = data.dropna(subset=["fedavg_final_test_mse", "local_final_test_mse", "local_epochs"])
    if data.empty:
        return

    width = 0.36
    for scenario, group in data.groupby("scenario", sort=False):
        group = group.sort_values("local_epochs")
        x = np.arange(len(group))
        labels = group["local_epochs"].astype(int).astype(str)
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
        ax.set_xticklabels(labels)
        ax.set_xlabel("Local epochs")
        ax.set_ylabel("Final test MSE")
        ax.set_title(f"{scenario}: {compare_client}")
        ax.legend(frameon=False, ncol=2)
        safe_scenario = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(scenario))
        safe_client = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(compare_client))
        save_figure(
            fig,
            plots_dir / f"fig_g_hparam_{safe_scenario}_{safe_client}_fedavg_vs_local_by_epochs.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Scenario G over multiple local-epoch settings and aggregate results."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable case definition: name=folder or name=folder:description.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline scenario. Default: first scenario.")
    parser.add_argument("--local-epochs-list", default="1,2,4,8,16")
    parser.add_argument("--label-key", default=LABEL_KEY)
    parser.add_argument("--output-dir", type=Path, default=Path("scenario_g_hparam_sweep_results"))
    parser.add_argument("--compute-mmd-y", action="store_true")
    parser.add_argument("--mmd-max-samples-per-client", type=int, default=1000)
    parser.add_argument("--mmd-gamma", type=float, default=None)
    parser.add_argument("--target-mse", type=float, default=None)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from existing CSVs in --output-dir; do not retrain models.",
    )
    parser.add_argument(
        "--make-single-run-plots",
        action="store_true",
        help="Also create the regular Scenario G plots inside each local_epochs_* folder.",
    )
    parser.add_argument(
        "--run-local-baseline",
        action="store_true",
        help="Train local-only FNN baselines for every sweep setting. This is slower.",
    )
    parser.add_argument("--local-baseline-epochs", type=int, default=0, help="0 means use --rounds.")
    parser.add_argument(
        "--compare-client",
        default="gfli4",
        help="Client name for FedAvg-vs-Local-only bars across local epochs.",
    )

    parser.add_argument("--rounds", type=int, default=250)
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
    configure_ieee_plots()

    if args.plot_only:
        summary_path = args.output_dir / "scenario_g_hparam_summary.csv"
        curves_path = args.output_dir / "scenario_g_hparam_learning_curves.csv"
        if not summary_path.exists() or not curves_path.exists():
            raise FileNotFoundError(
                "Plot-only mode requires existing CSV files: "
                f"{summary_path} and {curves_path}"
            )
        hparam_summary = pd.read_csv(summary_path)
        hparam_learning_curves = pd.read_csv(curves_path)
        per_client_path = args.output_dir / "scenario_g_hparam_per_client_results.csv"
        hparam_per_client = pd.read_csv(per_client_path) if per_client_path.exists() else None
        make_hparam_plots(
            args.output_dir,
            hparam_summary,
            hparam_learning_curves,
            hparam_per_client,
            args.compare_client,
        )
        print(f"Regenerated plots from existing CSVs in: {args.output_dir}")
        return

    scenario_specs = parse_scenarios(args)
    baseline_name = args.baseline or scenario_specs[0].name
    local_epochs_values = parse_int_list(args.local_epochs_list)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Using device:", device)
    print("Scenario order:", ", ".join(item.name for item in scenario_specs))
    print("Baseline:", baseline_name)
    print("Local epochs sweep:", local_epochs_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heterogeneity_summary, hy_results = compute_global_heterogeneity(scenario_specs, baseline_name, args)
    heterogeneity_summary.to_csv(args.output_dir / "scenario_g_hparam_heterogeneity_summary.csv", index=False)

    summary_frames = []
    curve_frames = []
    client_frames = []
    for local_epochs in local_epochs_values:
        summary, curves, clients = run_one_setting(
            args,
            scenario_specs=scenario_specs,
            heterogeneity_summary=heterogeneity_summary,
            hy_results=hy_results,
            local_epochs=local_epochs,
            device=device,
        )
        summary_frames.append(summary)
        curve_frames.append(curves)
        client_frames.append(clients)

    hparam_summary = pd.concat(summary_frames, ignore_index=True)
    hparam_learning_curves = pd.concat(curve_frames, ignore_index=True)
    hparam_per_client = pd.concat(client_frames, ignore_index=True)

    hparam_summary.to_csv(args.output_dir / "scenario_g_hparam_summary.csv", index=False)
    hparam_learning_curves.to_csv(args.output_dir / "scenario_g_hparam_learning_curves.csv", index=False)
    hparam_per_client.to_csv(args.output_dir / "scenario_g_hparam_per_client_results.csv", index=False)

    if args.make_plots:
        make_hparam_plots(
            args.output_dir,
            hparam_summary,
            hparam_learning_curves,
            hparam_per_client,
            args.compare_client,
        )

    print("\n===== Scenario G Hyperparameter Sweep Summary =====")
    display_cols = [
        "scenario",
        "local_epochs",
        "H_Y_stat_mean",
        "H_Y_MMD_mean",
        "final_test_mse_mean",
        "final_test_mse_worst_client",
        "final_test_mse_std_across_clients",
        "local_update_steps_mean",
        "fedavg_improvement_percent_vs_local_mean",
    ]
    display_cols = [col for col in display_cols if col in hparam_summary.columns]
    print(hparam_summary[display_cols].to_string(index=False))
    print(f"\nSaved sweep outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
