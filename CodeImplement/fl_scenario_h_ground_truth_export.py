# -*- coding: utf-8 -*-
"""
Export Scenario H ground-truth Yibr_resp files from a test dataset.

The exporter is intentionally exact-match by default: it writes a MAT file only
when the requested [V, P, Q] operating point exists in the source test set.
Missing OPs are listed in a manifest instead of being filled by predictions or
interpolation.

Example:
    .\\.venv\\Scripts\\python.exe fl_scenario_h_ground_truth_export.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from fl_scenario_ae_fedavg import LABEL_KEY, load_dataset_mat
from fl_scenario_h_gnc_export import (
    COMPONENT_COLUMNS,
    OperatingPoint,
    mat_string_array,
    read_operating_points,
    safe_tag,
    y_vector_to_matrix,
)


def load_reference_freqs(output_dir: Path, operating_points: list[OperatingPoint]) -> np.ndarray | None:
    """Use existing Scenario H fGNC if prediction files are present."""
    for op in operating_points:
        path = output_dir / f"scenario_h_noise0pct_{safe_tag(op.label)}_Yibr_resp.mat"
        if not path.exists():
            continue
        data = loadmat(path)
        if "fGNC" in data:
            freqs = np.asarray(data["fGNC"], dtype=np.float64).reshape(-1)
            freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
            if freqs.size:
                return freqs
    return None


def extract_exact_op(
    x: np.ndarray,
    y: np.ndarray,
    op: OperatingPoint,
    freqs_hz: np.ndarray | None,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.array([op.v, op.p, op.q], dtype=np.float64)
    mask = (
        np.isclose(x[:, 0], target[0], atol=atol, rtol=0.0)
        & np.isclose(x[:, 1], target[1], atol=atol, rtol=0.0)
        & np.isclose(x[:, 2], target[2], atol=atol, rtol=0.0)
    )
    if not np.any(mask):
        raise KeyError(f"{op.label} [V,P,Q]={target.tolist()} was not found in the test set.")

    x_op = np.asarray(x[mask], dtype=np.float64)
    y_op = np.asarray(y[mask], dtype=np.float64)
    order = np.argsort(x_op[:, 3])
    x_op = x_op[order]
    y_op = y_op[order]

    if freqs_hz is not None:
        keep_indices = []
        for freq in freqs_hz:
            matches = np.flatnonzero(np.isclose(x_op[:, 3], freq, atol=atol, rtol=0.0))
            if matches.size == 0:
                raise KeyError(f"{op.label} is missing f={freq:g} Hz in the test set.")
            keep_indices.append(int(matches[0]))
        x_op = x_op[keep_indices]
        y_op = y_op[keep_indices]

    return y_vector_to_matrix(y_op), x_op, y_op


def export_ground_truth(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    op_args = argparse.Namespace(op_csv=args.op_csv, v_ref=args.v_ref)
    operating_points = read_operating_points(op_args)

    data = load_dataset_mat(args.test_mat, LABEL_KEY)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["Y"], dtype=np.float64)

    freqs_hz = None
    if args.use_scenario_h_freqs:
        freqs_hz = load_reference_freqs(output_dir, operating_points)
        if freqs_hz is None:
            print("[WARN] Existing Scenario H fGNC not found; exporting source test-set frequencies.")

    manifest_rows = []
    long_rows = []
    exported_resp = []
    exported_freqs = []
    exported_labels = []

    for op in operating_points:
        op_tag = safe_tag(op.label)
        mat_path = output_dir / f"scenario_h_groundtruth_{op_tag}_Yibr_resp.mat"
        try:
            y_resp, x_op, y_vector = extract_exact_op(
                x=x,
                y=y,
                op=op,
                freqs_hz=freqs_hz,
                atol=args.atol,
            )
        except KeyError as exc:
            print(f"[MISSING] {exc}")
            manifest_rows.append(
                {
                    "op_label": op.label,
                    "V": op.v,
                    "P": op.p,
                    "Q": op.q,
                    "status": "missing_exact_op_or_frequency",
                    "mat_file": "",
                    "source_test_mat": str(args.test_mat),
                    "message": str(exc),
                }
            )
            continue

        f_out = x_op[:, 3].astype(np.float64)
        w_out = 2.0 * np.pi * f_out
        savemat(
            mat_path,
            {
                "Yibr_resp": y_resp,
                "fGNC": f_out.reshape(1, -1),
                "wGNC": w_out.reshape(1, -1),
                "X_query": x_op,
                "operating_point": np.array([[op.v, op.p, op.q]], dtype=np.float64),
                "op_label": op.label,
                "target_ibr": args.target_ibr,
                "source_test_mat": str(args.test_mat),
                "source_kind": "ground_truth_test_set_exact_match",
                "component_order": "Ydd,Ydq,Yqd,Yqq with Re/Im pairs from Y_Y",
                "units_note": "raw Y_Y units",
            },
            do_compression=True,
        )

        exported_resp.append(y_resp)
        exported_freqs.append(f_out.copy())
        exported_labels.append(op.label)
        manifest_rows.append(
            {
                "op_label": op.label,
                "V": op.v,
                "P": op.p,
                "Q": op.q,
                "status": "exported_exact",
                "mat_file": str(mat_path),
                "source_test_mat": str(args.test_mat),
                "message": "",
            }
        )

        for freq_idx, freq in enumerate(f_out):
            row = {
                "op_label": op.label,
                "V": op.v,
                "P": op.p,
                "Q": op.q,
                "f_Hz": float(freq),
                "w_rad_s": float(w_out[freq_idx]),
            }
            for name, row_idx, col_idx, re_idx, im_idx in COMPONENT_COLUMNS:
                value = y_resp[row_idx, col_idx, freq_idx]
                row[f"{name}_re"] = float(value.real)
                row[f"{name}_im"] = float(value.imag)
                row[f"{name}_complex"] = f"{value.real:.16g}{value.imag:+.16g}i"
                row[f"{name}_vec_re"] = float(y_vector[freq_idx, re_idx])
                row[f"{name}_vec_im"] = float(y_vector[freq_idx, im_idx])
            long_rows.append(row)

        print(f"Saved ground-truth Yibr_resp: {mat_path}")

    manifest_path = output_dir / "scenario_h_groundtruth_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Saved ground-truth manifest: {manifest_path}")

    if long_rows:
        csv_path = output_dir / "scenario_h_groundtruth_yibr_resp_long.csv"
        pd.DataFrame(long_rows).to_csv(csv_path, index=False)
        print(f"Saved ground-truth long CSV: {csv_path}")

    if exported_resp:
        n_freqs = {resp.shape[2] for resp in exported_resp}
        same_freq_grid = len(n_freqs) == 1 and all(np.array_equal(exported_freqs[0], item) for item in exported_freqs)
        if same_freq_grid:
            combined = np.stack(exported_resp, axis=3)
            f_combined = exported_freqs[0].reshape(1, -1)
            combined_path = output_dir / "scenario_h_groundtruth_all_Yibr_resp.mat"
            savemat(
                combined_path,
                {
                    "Yibr_resp_all": combined,
                    "fGNC": f_combined,
                    "wGNC": (2.0 * np.pi * f_combined),
                    "op_labels": mat_string_array(exported_labels),
                    "target_ibr": args.target_ibr,
                    "source_test_mat": str(args.test_mat),
                    "dimension_note": "Yibr_resp_all is 2 x 2 x nGNC x nExportedOP",
                    "component_order": "Ydd,Ydq,Yqd,Yqq with Re/Im pairs from Y_Y",
                    "units_note": "raw Y_Y units",
                },
                do_compression=True,
            )
            print(f"Saved combined ground-truth MAT: {combined_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Scenario H ground-truth Yibr_resp MAT files.")
    parser.add_argument("--test-mat", type=str, default="gfli1_test_impedance_dataset.mat")
    parser.add_argument("--output-dir", type=str, default="scenario_h_gnc_exports")
    parser.add_argument("--target-ibr", type=str, default="gfli1")
    parser.add_argument("--op-csv", type=str, default=None)
    parser.add_argument("--v-ref", type=float, default=1.0)
    parser.add_argument(
        "--use-scenario-h-freqs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fGNC from existing Scenario H prediction MAT files when available.",
    )
    parser.add_argument("--atol", type=float, default=1e-9)
    return parser.parse_args()


if __name__ == "__main__":
    export_ground_truth(parse_args())
