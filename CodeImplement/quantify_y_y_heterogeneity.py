# -*- coding: utf-8 -*-
"""
Within-case output-space heterogeneity H_Y for IBR admittance datasets.

H_Y measures how heterogeneous the marginal admittance/output distributions
Y_Y are among clients inside the same scenario/case. It uses only Y_Y, so it
does not require clients to share the same input samples, operating points, or
frequency samples, and it avoids dilution by input-space heterogeneity.

Two Y-only metrics are supported:
  - H_Y_stat: average pairwise distance between client mean/std vectors after
    global Y_Y standardization.
  - H_Y_MMD: optional average pairwise RBF-kernel MMD^2 between standardized
    client Y_Y distributions.

All scenarios share one StandardScaler fitted on all loaded Y_Y samples. This
makes H_Y values comparable across cases. Percentage increase is computed by
normalizing each scenario's H_Y by the selected baseline scenario.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


LABEL_KEY = "Y_Y"
DEFAULT_SCENARIOS = {
    "same_topology": "zero_percent_noise",
    "different_topology": "differentIBRstrcuture_IBR4_strong",
}
SUMMARY_STATS = ("mean", "median", "std", "min", "max", "p90")
MMD_COLUMNS = (
    "H_Y_MMD_mean",
    "H_Y_MMD_median",
    "H_Y_MMD_std",
    "H_Y_MMD_min",
    "H_Y_MMD_max",
    "H_Y_MMD_p90",
    "relative_H_Y_MMD",
    "H_Y_MMD_increase_percent",
)


@dataclass
class ClientOutput:
    name: str
    y: np.ndarray
    path: str


@dataclass
class ScenarioData:
    name: str
    folder: Path
    clients: list[ClientOutput]


@dataclass
class HYResult:
    scenario: str
    folder: Path
    clients: list[ClientOutput]
    client_stats: pd.DataFrame
    pairwise_stat: pd.DataFrame
    stat_summary: dict[str, float]
    pairwise_mmd: pd.DataFrame | None = None
    mmd_summary: dict[str, float] | None = None
    relative_stat: float = float("nan")
    stat_increase_percent: float = float("nan")
    relative_mmd: float = float("nan")
    mmd_increase_percent: float = float("nan")


def fix_shape(arr: np.ndarray, expected_cols: int | None = None) -> np.ndarray:
    """Return a 2D float64 array, transposing MATLAB-style data when needed."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("Expected a vector or matrix, got a scalar.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if expected_cols is not None:
        if arr.shape[1] != expected_cols and arr.shape[0] == expected_cols:
            arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got shape {arr.shape}.")
    return arr.astype(np.float64, copy=False)


def _read_h5_matrix(path: Path, key: str, expected_cols: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        for group_name in ("Dataset", "dataset"):
            if group_name in f and key in f[group_name]:
                return fix_shape(f[group_name][key][()], expected_cols=expected_cols)
        if key in f:
            return fix_shape(f[key][()], expected_cols=expected_cols)
    raise KeyError(f"Could not find key {key!r} in HDF5 file: {path}")


def _read_mat_struct_field(dataset: np.ndarray, key: str) -> np.ndarray | None:
    if not (hasattr(dataset, "dtype") and dataset.dtype.names):
        return None
    if key not in dataset.dtype.names:
        return None
    return np.asarray(dataset[key]).squeeze()


def _read_mat_matrix(path: Path, key: str, expected_cols: int) -> np.ndarray:
    mat = loadmat(str(path))
    data = {k: v for k, v in mat.items() if not k.startswith("__")}

    for struct_name in ("Dataset", "dataset"):
        if struct_name in data:
            raw = _read_mat_struct_field(data[struct_name], key)
            if raw is not None:
                return fix_shape(raw, expected_cols=expected_cols)

    if key in data:
        return fix_shape(data[key], expected_cols=expected_cols)
    raise KeyError(f"Could not find key {key!r} in MAT file: {path}")


def load_matrix_from_file(path: Path, key: str, expected_cols: int) -> np.ndarray:
    """Load Dataset.<key> or top-level <key> from MATLAB/HDF5 files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h5_error: Exception | None = None
    try:
        return _read_h5_matrix(path, key, expected_cols)
    except Exception as exc:
        h5_error = exc

    try:
        return _read_mat_matrix(path, key, expected_cols)
    except Exception as mat_error:
        raise RuntimeError(
            f"Error loading key {key!r} from {path}: h5py -> {h5_error}; loadmat -> {mat_error}"
        ) from mat_error


def load_y_y(path: Path, label_key: str = LABEL_KEY) -> np.ndarray:
    """Load the 8-column Y_Y output matrix used by the H_Y metrics."""
    return load_matrix_from_file(path, label_key, expected_cols=8)


def expected_client_paths(folder: Path, split: str) -> list[Path]:
    paths: list[Path] = []
    for idx in range(1, 10):
        if split == "train":
            paths.append(folder / f"gfli{idx}_impedance_dataset.mat")
        elif split == "test":
            paths.append(folder / f"gfli{idx}_test_impedance_dataset.mat")
        elif split == "both":
            paths.append(folder / f"gfli{idx}_impedance_dataset.mat")
            paths.append(folder / f"gfli{idx}_test_impedance_dataset.mat")
        else:
            raise ValueError(f"Unsupported split: {split}")
    return paths


def _client_sort_key(name: str) -> tuple[int, str]:
    suffix = name.replace("gfli", "")
    return (int(suffix), name) if suffix.isdigit() else (10_000, name)


def load_scenario_clients(folder: Path, split: str, label_key: str) -> list[ClientOutput]:
    """Load Y_Y for each client; for split='both', concatenate train and test."""
    clients_by_name: dict[str, list[np.ndarray]] = {}
    paths_by_name: dict[str, list[str]] = {}

    for path in expected_client_paths(folder, split):
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        client_name = path.name.split("_")[0].lower()
        clients_by_name.setdefault(client_name, []).append(load_y_y(path, label_key=label_key))
        paths_by_name.setdefault(client_name, []).append(str(path))

    clients = []
    for name in sorted(clients_by_name, key=_client_sort_key):
        y = np.vstack(clients_by_name[name])
        clients.append(ClientOutput(name=name, y=y, path=" + ".join(paths_by_name[name])))
    return clients


def parse_scenarios(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.scenario:
        scenarios = []
        seen_names = set()
        for item in args.scenario:
            if "=" not in item:
                raise ValueError(f"Invalid --scenario value {item!r}. Expected name=folder.")
            name, folder = item.split("=", 1)
            name = name.strip()
            folder = folder.strip()
            if not name or not folder:
                raise ValueError(f"Invalid --scenario value {item!r}. Expected name=folder.")
            if name in seen_names:
                raise ValueError(f"Duplicate scenario name: {name}")
            seen_names.add(name)
            scenarios.append((name, Path(folder)))
        return scenarios

    return [
        ("same_topology", args.same_folder),
        ("different_topology", args.different_folder),
    ]


def load_scenarios(args: argparse.Namespace) -> list[ScenarioData]:
    scenarios = []
    for name, folder in parse_scenarios(args):
        clients = load_scenario_clients(folder, split=args.split, label_key=args.label_key)
        if len(clients) < 2:
            raise ValueError(f"Need at least two clients for scenario {name}: {folder}")
        scenarios.append(ScenarioData(name=name, folder=folder, clients=clients))
    return scenarios


def fit_global_y_scaler(scenarios: list[ScenarioData]) -> StandardScaler:
    all_y = np.vstack([client.y for scenario in scenarios for client in scenario.clients])
    return StandardScaler().fit(all_y)


def summarize_upper_triangle(pairwise_matrix: pd.DataFrame) -> dict[str, float]:
    values = pairwise_matrix.values[np.triu_indices_from(pairwise_matrix.values, k=1)]
    if values.size == 0:
        return {stat: float("nan") for stat in SUMMARY_STATS}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
    }


def compute_hy_stat_for_scenario(
    scenario_clients: list[ClientOutput],
    scaler_y: StandardScaler,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Compute moment-based H_Y_stat from standardized Y_Y distributions."""
    names = [client.name for client in scenario_clients]
    means = []
    stds = []
    stats_rows = []

    for client in scenario_clients:
        y_scaled = scaler_y.transform(client.y)
        mu = y_scaled.mean(axis=0)
        sigma = y_scaled.std(axis=0, ddof=0)
        means.append(mu)
        stds.append(sigma)
        stats_rows.append(
            {
                "client": client.name,
                "n_samples": client.y.shape[0],
                "y_mu_norm": float(np.linalg.norm(mu, ord=2)),
                "y_sigma_norm": float(np.linalg.norm(sigma, ord=2)),
            }
        )

    pairwise = np.zeros((len(scenario_clients), len(scenario_clients)), dtype=np.float64)
    for i in range(len(scenario_clients)):
        for j in range(i + 1, len(scenario_clients)):
            d_ij = float(
                np.linalg.norm(means[i] - means[j], ord=2)
                + np.linalg.norm(stds[i] - stds[j], ord=2)
            )
            pairwise[i, j] = d_ij
            pairwise[j, i] = d_ij

    matrix = pd.DataFrame(pairwise, index=names, columns=names)
    return matrix, summarize_upper_triangle(matrix), pd.DataFrame(stats_rows)


def _sample_rows(values: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or values.shape[0] <= max_samples:
        return values
    indices = rng.choice(values.shape[0], size=max_samples, replace=False)
    return values[indices]


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.sum(a * a, axis=1)[:, None]
    b_norm = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a_norm + b_norm - 2.0 * (a @ b.T), 0.0)


def _rbf_mean(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    return float(np.exp(-gamma * _squared_distances(a, b)).mean())


def estimate_mmd_gamma_y(
    scenarios: list[ScenarioData],
    scaler_y: StandardScaler,
    max_samples_per_client: int,
    seed: int,
    max_pooled_samples: int = 2000,
    eps: float = 1e-12,
) -> float:
    rng = np.random.default_rng(seed)
    sampled = []
    for scenario in scenarios:
        for client in scenario.clients:
            sampled.append(_sample_rows(scaler_y.transform(client.y), max_samples_per_client, rng))

    pooled = np.vstack(sampled)
    pooled = _sample_rows(pooled, max_pooled_samples, rng)
    if pooled.shape[0] < 2:
        return 1.0

    distances = _squared_distances(pooled, pooled)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > eps]
    median_squared_distance = float(np.median(positive)) if positive.size else eps
    return 1.0 / (2.0 * median_squared_distance + eps)


def compute_hy_mmd_for_scenario(
    scenario_clients: list[ClientOutput],
    scaler_y: StandardScaler,
    max_samples_per_client: int,
    gamma: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute pairwise RBF-kernel MMD^2 on standardized Y_Y distributions."""
    rng = np.random.default_rng(seed)
    names = [client.name for client in scenario_clients]
    sampled = [
        _sample_rows(scaler_y.transform(client.y), max_samples_per_client, rng)
        for client in scenario_clients
    ]
    self_terms = [_rbf_mean(values, values, gamma) for values in sampled]

    pairwise = np.zeros((len(scenario_clients), len(scenario_clients)), dtype=np.float64)
    for i in range(len(scenario_clients)):
        for j in range(i + 1, len(scenario_clients)):
            cross = _rbf_mean(sampled[i], sampled[j], gamma)
            mmd_squared = max(self_terms[i] + self_terms[j] - 2.0 * cross, 0.0)
            pairwise[i, j] = mmd_squared
            pairwise[j, i] = mmd_squared

    matrix = pd.DataFrame(pairwise, index=names, columns=names)
    return matrix, summarize_upper_triangle(matrix)


def compute_hy_results(
    scenarios: list[ScenarioData],
    scaler_y: StandardScaler,
    compute_mmd_y: bool,
    mmd_max_samples_per_client: int,
    mmd_gamma: float | None,
    random_seed: int,
) -> tuple[list[HYResult], float | None]:
    gamma_y = None
    if compute_mmd_y:
        gamma_y = (
            float(mmd_gamma)
            if mmd_gamma is not None
            else estimate_mmd_gamma_y(
                scenarios,
                scaler_y=scaler_y,
                max_samples_per_client=mmd_max_samples_per_client,
                seed=random_seed,
            )
        )

    results = []
    for scenario in scenarios:
        pairwise_stat, stat_summary, client_stats = compute_hy_stat_for_scenario(
            scenario.clients, scaler_y=scaler_y
        )
        pairwise_mmd = None
        mmd_summary = None
        if compute_mmd_y and gamma_y is not None:
            pairwise_mmd, mmd_summary = compute_hy_mmd_for_scenario(
                scenario.clients,
                scaler_y=scaler_y,
                max_samples_per_client=mmd_max_samples_per_client,
                gamma=gamma_y,
                seed=random_seed,
            )

        results.append(
            HYResult(
                scenario=scenario.name,
                folder=scenario.folder,
                clients=scenario.clients,
                client_stats=client_stats,
                pairwise_stat=pairwise_stat,
                stat_summary=stat_summary,
                pairwise_mmd=pairwise_mmd,
                mmd_summary=mmd_summary,
            )
        )
    return results, gamma_y


def normalize_against_baseline(results: list[HYResult], baseline_name: str) -> None:
    by_name = {result.scenario: result for result in results}
    if baseline_name not in by_name:
        available = ", ".join(by_name)
        raise ValueError(f"Baseline scenario {baseline_name!r} was not found. Available: {available}")

    baseline = by_name[baseline_name]
    baseline_stat = baseline.stat_summary["mean"]
    baseline_mmd = baseline.mmd_summary["mean"] if baseline.mmd_summary else float("nan")

    for result in results:
        if result.scenario == baseline_name:
            result.relative_stat = 1.0
            result.stat_increase_percent = 0.0
            if result.mmd_summary:
                result.relative_mmd = 1.0
                result.mmd_increase_percent = 0.0
            continue

        result.relative_stat = result.stat_summary["mean"] / baseline_stat if baseline_stat else float("nan")
        result.stat_increase_percent = (result.relative_stat - 1.0) * 100.0

        if result.mmd_summary:
            result.relative_mmd = (
                result.mmd_summary["mean"] / baseline_mmd if baseline_mmd else float("nan")
            )
            result.mmd_increase_percent = (result.relative_mmd - 1.0) * 100.0


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "scenario"


def _summary_columns(prefix: str, summary: dict[str, float] | None) -> dict[str, float]:
    if summary is None:
        return {f"{prefix}_{stat}": float("nan") for stat in SUMMARY_STATS}
    return {f"{prefix}_{stat}": summary[stat] for stat in SUMMARY_STATS}


def save_hy_results(results: list[HYResult], output_dir: Path, baseline_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for result in results:
        safe_name = sanitize_filename(result.scenario)
        total_samples = sum(client.y.shape[0] for client in result.clients)

        result.pairwise_stat.to_csv(output_dir / f"{safe_name}_pairwise_DY_stat.csv")
        if result.pairwise_mmd is not None:
            result.pairwise_mmd.to_csv(output_dir / f"{safe_name}_pairwise_MMD_Y.csv")
        result.client_stats.to_csv(output_dir / f"{safe_name}_client_Y_stats.csv", index=False)

        row = {
            "scenario": result.scenario,
            "folder": str(result.folder),
            "n_clients": len(result.clients),
            "total_samples": total_samples,
        }
        row.update(_summary_columns("H_Y_stat", result.stat_summary))
        row["relative_H_Y_stat"] = result.relative_stat
        row["H_Y_stat_increase_percent"] = result.stat_increase_percent
        row.update(_summary_columns("H_Y_MMD", result.mmd_summary))
        row["relative_H_Y_MMD"] = result.relative_mmd
        row["H_Y_MMD_increase_percent"] = result.mmd_increase_percent
        row["baseline_scenario"] = baseline_name
        rows.append(row)

    columns = [
        "scenario",
        "folder",
        "n_clients",
        "total_samples",
        "H_Y_stat_mean",
        "H_Y_stat_median",
        "H_Y_stat_std",
        "H_Y_stat_min",
        "H_Y_stat_max",
        "H_Y_stat_p90",
        "relative_H_Y_stat",
        "H_Y_stat_increase_percent",
        *MMD_COLUMNS,
        "baseline_scenario",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        output_dir / "heterogeneity_HY_summary.csv", index=False
    )


def _format_stats(summary: dict[str, float]) -> str:
    return " / ".join(f"{summary[stat]:.6f}" for stat in SUMMARY_STATS)


def print_hy_summary(results: list[HYResult], baseline_name: str, gamma_y: float | None) -> None:
    print("\n===== Within-Case Output-Space Heterogeneity H_Y =====")
    print(f"\nBaseline scenario: {baseline_name}")
    if gamma_y is not None:
        print(f"MMD Y gamma: {gamma_y:.6g}")

    for result in results:
        total_samples = sum(client.y.shape[0] for client in result.clients)
        print(f"\nScenario: {result.scenario}")
        print(f"  Folder: {result.folder}")
        print(f"  Clients: {len(result.clients)}")
        print(f"  Total samples: {total_samples}")
        print("  H_Y_stat:")
        print(f"    mean / median / std / min / max / p90 = {_format_stats(result.stat_summary)}")
        print(f"    Relative H_Y_stat: {result.relative_stat:.6f}")
        print(f"    H_Y_stat increase: {result.stat_increase_percent:+.2f}%")

        if result.mmd_summary:
            print("  H_Y_MMD:")
            print(f"    mean / median / std / min / max / p90 = {_format_stats(result.mmd_summary)}")
            print(f"    Relative H_Y_MMD: {result.relative_mmd:.6f}")
            print(f"    H_Y_MMD increase: {result.mmd_increase_percent:+.2f}%")

    print("\nComparison against baseline:")
    for result in results:
        if result.scenario == baseline_name:
            continue
        print(f"  {result.scenario}: H_Y_stat {result.stat_increase_percent:+.2f}%")
        if result.mmd_summary:
            print(f"  {result.scenario}: H_Y_MMD {result.mmd_increase_percent:+.2f}%")


def make_hy_plots(results: list[HYResult], output_dir: Path, baseline_name: str) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        safe_name = sanitize_filename(result.scenario)
        for matrix, suffix, title in (
            (result.pairwise_stat, "DY_stat", "D_Y stat"),
            (result.pairwise_mmd, "MMD_Y", "MMD^2 Y"),
        ):
            if matrix is None:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            image = ax.imshow(matrix.values, cmap="viridis")
            ax.set_title(f"{result.scenario}: {title}")
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{safe_name}_pairwise_{suffix}_heatmap.png", dpi=300)
            plt.close(fig)

    labels = [result.scenario for result in results]
    stat_means = [result.stat_summary["mean"] for result in results]
    fig, ax = plt.subplots(figsize=(max(6.0, len(results) * 1.2), 4.0))
    ax.bar(labels, stat_means, color="#4C72B0")
    ax.set_ylabel("H_Y_stat mean")
    ax.set_title("Within-case output heterogeneity")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plots_dir / "H_Y_stat_mean_bar.png", dpi=300)
    plt.close(fig)

    non_baseline = [result for result in results if result.scenario != baseline_name]
    if non_baseline:
        fig, ax = plt.subplots(figsize=(max(6.0, len(non_baseline) * 1.2), 4.0))
        ax.bar(
            [result.scenario for result in non_baseline],
            [result.stat_increase_percent for result in non_baseline],
            color="#DD8452",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("H_Y_stat increase (%)")
        ax.set_title("Increase vs baseline")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "H_Y_stat_increase_percent_bar.png", dpi=300)
        plt.close(fig)

    if any(result.mmd_summary for result in results):
        mmd_means = [
            result.mmd_summary["mean"] if result.mmd_summary else np.nan for result in results
        ]
        fig, ax = plt.subplots(figsize=(max(6.0, len(results) * 1.2), 4.0))
        ax.bar(labels, mmd_means, color="#55A868")
        ax.set_ylabel("H_Y_MMD mean")
        ax.set_title("Within-case MMD output heterogeneity")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(plots_dir / "H_Y_MMD_mean_bar.png", dpi=300)
        plt.close(fig)

        mmd_non_baseline = [
            result for result in non_baseline if result.mmd_summary is not None
        ]
        if mmd_non_baseline:
            fig, ax = plt.subplots(figsize=(max(6.0, len(mmd_non_baseline) * 1.2), 4.0))
            ax.bar(
                [result.scenario for result in mmd_non_baseline],
                [result.mmd_increase_percent for result in mmd_non_baseline],
                color="#C44E52",
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("H_Y_MMD increase (%)")
            ax.set_title("MMD increase vs baseline")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(plots_dir / "H_Y_MMD_increase_percent_bar.png", dpi=300)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute within-case output-space heterogeneity H_Y from Y_Y admittance data."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeatable scenario definition in the form name=folder.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline scenario name. Default: first scenario in the scenario list.",
    )
    parser.add_argument(
        "--same-folder",
        type=Path,
        default=Path(DEFAULT_SCENARIOS["same_topology"]),
        help="Legacy folder for same-topology scenario if --scenario is not used.",
    )
    parser.add_argument(
        "--different-folder",
        type=Path,
        default=Path(DEFAULT_SCENARIOS["different_topology"]),
        help="Legacy folder for different-topology scenario if --scenario is not used.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="train",
        help="Use train files, test files, or both per client. Default: train.",
    )
    parser.add_argument(
        "--label-key",
        default=LABEL_KEY,
        help="MAT/HDF5 output key to load. Default: Y_Y.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("heterogeneity_HY_results"),
        help="Directory for summary, pairwise CSV files, and optional plots.",
    )
    parser.add_argument(
        "--compute-mmd-y",
        action="store_true",
        help="Also compute H_Y_MMD using RBF-kernel MMD^2 on standardized Y_Y.",
    )
    parser.add_argument(
        "--compute-mmd",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mmd-max-samples-per-client",
        type=int,
        default=1000,
        help="Maximum samples per client used for MMD. Default: 1000.",
    )
    parser.add_argument(
        "--mmd-gamma",
        type=float,
        default=None,
        help="RBF gamma for MMD. If omitted, use the median heuristic.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for MMD subsampling and median heuristic. Default: 42.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate pairwise heatmaps and H_Y comparison bar charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_mmd_y = args.compute_mmd_y or args.compute_mmd

    scenarios = load_scenarios(args)
    baseline_name = args.baseline if args.baseline else scenarios[0].name
    scaler_y = fit_global_y_scaler(scenarios)
    results, gamma_y = compute_hy_results(
        scenarios,
        scaler_y=scaler_y,
        compute_mmd_y=compute_mmd_y,
        mmd_max_samples_per_client=args.mmd_max_samples_per_client,
        mmd_gamma=args.mmd_gamma,
        random_seed=args.random_seed,
    )
    normalize_against_baseline(results, baseline_name)
    print_hy_summary(results, baseline_name, gamma_y)
    save_hy_results(results, args.output_dir, baseline_name)
    if args.make_plots:
        make_hy_plots(results, args.output_dir, baseline_name)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
