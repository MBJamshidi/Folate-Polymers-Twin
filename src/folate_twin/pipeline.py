"""End-to-end training pipeline: train A-D once and write every result table.

A single pass produces both the headline tables and the training record that
the figures are drawn from:

====================================  ==========================================
``metric_evaluation_matrix.csv/.xlsx`` MSE / RMSE / R^2 per configuration
``numerical_trajectories.csv``         dense Configuration D trajectories
``rk4_digital_twin.csv``               differentiable RK4 rollout (Config. D)
``training_history.csv``               losses, multipliers and rates per epoch
``dense_predictions_all.csv``          dense trajectories of A-D (200 points)
``node_predictions.csv``               A-D evaluated at the 8 sampling times
``physics_diagnostics_D.csv``          r_p, r_t and RBA weights (Config. D)
``figure_data_check.csv``              this run next to the reference metrics
====================================  ==========================================
"""

from __future__ import annotations

import csv
import os
from importlib import resources
from typing import Dict

import torch

from . import config as C
from .config import DEVICE, DTYPE
from .data import Normaliser, TargetMatrix, load_target
from .physics import physics_residuals, rba_weights
from .training import TrainResult, run_ablation
from .twin import DifferentiableRK4

HISTORY_KEYS = ["epoch", "total", "data", "physics", "boundary",
                "lambda_d", "lambda_p", "lambda_b"]


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #
def print_report(results: Dict[str, TrainResult]) -> None:
    print("\n" + "=" * 78)
    print("TRACKING METRICS (ug/mL, evaluated at the 8 sampling times)")
    print("=" * 78)
    header = f"{'Config':<16}{'Cp R2':>10}{'Cp RMSE':>11}{'Ct R2':>10}{'Ct RMSE':>11}{'mean R2':>10}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        p, t = res.metrics["plasma"], res.metrics["tumour"]
        mean_r2 = 0.5 * (p["R2"] + t["R2"])
        print(f"{name:<16}{p['R2']:>10.4f}{p['RMSE']:>11.3f}"
              f"{t['R2']:>10.4f}{t['RMSE']:>11.3f}{mean_r2:>10.4f}")

    print("\nRecovered effective kinetic parameters (per configuration):")
    for name, res in results.items():
        rates = ", ".join(f"{k}={v:.4f}" for k, v in res.final_rates.items())
        print(f"  {name:<16} {rates}")
        if res.config.adaptive:
            lam = ", ".join(f"{k}={v:.3e}" for k, v in res.lambdas.items())
            print(f"  {'':<16} self-adaptive weights: {lam}")


# --------------------------------------------------------------------------- #
# Headline tables
# --------------------------------------------------------------------------- #
def write_metric_tables(results: Dict[str, TrainResult], tables: str) -> None:
    """``metric_evaluation_matrix.csv`` and the Excel workbook of the same table."""
    rows = [[name, comp, res.metrics[comp]["MSE"], res.metrics[comp]["RMSE"], res.metrics[comp]["R2"]]
            for name, res in results.items() for comp in ("plasma", "tumour")]
    header = ["config", "compartment", "MSE", "RMSE", "R2"]

    with open(os.path.join(tables, "metric_evaluation_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    ws.append(header)
    for row in rows:
        ws.append(row)
    wb.save(os.path.join(tables, "metric_evaluation_matrix.xlsx"))


def write_twin_tables(
    results: Dict[str, TrainResult],
    target: TargetMatrix,
    normaliser: Normaliser,
    tables: str,
    rk4_steps: int = C.RK4_STEPS,
) -> Dict[str, float]:
    """Dense Configuration D trajectories and the differentiable RK4 rollout.

    Returns the terminal tumour concentration (ug/mL) and dC_t/dC_0.
    """
    best = results["Configuration D"]
    t_dense = torch.linspace(0.0, 1.0, C.DENSE_POINTS, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    with torch.no_grad():
        pred = best.surrogate(t_dense)
    with open(os.path.join(tables, "numerical_trajectories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_hours", "Cp_pred_ugml", "Ct_pred_ugml"])
        for i in range(t_dense.shape[0]):
            w.writerow([
                float(t_dense[i, 0]) * normaliser.t_max,
                float(normaliser.denorm_conc(pred[i, 0])),
                float(normaliser.denorm_conc(pred[i, 1])),
            ])

    # C_0 is the learned systemic-input (dose) amplitude of Configuration D.
    twin = DifferentiableRK4(best.params, t_max=normaliser.t_max, steps=rk4_steps)
    c0_value = float(best.params.c0().detach())
    with torch.no_grad():
        t_grid, traj = twin.rollout(torch.tensor(c0_value, dtype=DTYPE, device=DEVICE))
    ct_final, dct_dc0 = twin.state_sensitivity(c0_value)
    with open(os.path.join(tables, "rk4_digital_twin.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_hours", "Cp_rk4_ugml", "Ct_rk4_ugml"])
        for i in range(t_grid.shape[0]):
            w.writerow([
                float(t_grid[i]),
                float(normaliser.denorm_conc(traj[i, 0])),
                float(normaliser.denorm_conc(traj[i, 1])),
            ])
    return {"Ct_final_ugml": ct_final * normaliser.c_max, "dCt_dC0": dct_dc0}


# --------------------------------------------------------------------------- #
# Training record used by the figures
# --------------------------------------------------------------------------- #
def write_figure_data(
    results: Dict[str, TrainResult],
    target: TargetMatrix,
    normaliser: Normaliser,
    tables: str,
    n_collocation: int = C.N_COLLOCATION,
) -> None:
    """Training history, dense / node predictions of A-D and the Configuration D
    physics diagnostics."""
    rate_keys = sorted({k for r in results.values() for h in r.history for k in h} - set(HISTORY_KEYS))
    with open(os.path.join(tables, "training_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + HISTORY_KEYS + rate_keys)
        for name, res in results.items():
            for h in res.history:
                w.writerow([name] + [h.get(k, "") for k in HISTORY_KEYS + rate_keys])

    t_dense = torch.linspace(0.0, 1.0, C.DENSE_POINTS, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    with open(os.path.join(tables, "dense_predictions_all.csv"), "w", newline="") as fd, \
         open(os.path.join(tables, "node_predictions.csv"), "w", newline="") as fn:
        wd, wn = csv.writer(fd), csv.writer(fn)
        wd.writerow(["config", "t_hours", "Cp_pred_ugml", "Ct_pred_ugml"])
        wn.writerow(["config", "t_hours", "Cp_obs_ugml", "Ct_obs_ugml",
                     "Cp_pred_ugml", "Ct_pred_ugml"])
        for name, res in results.items():
            res.surrogate.eval()
            with torch.no_grad():
                pd = res.surrogate(t_dense)
                pn = res.surrogate(target.t_bar)
            res.surrogate.train()
            for i in range(t_dense.shape[0]):
                wd.writerow([name, float(t_dense[i, 0]) * normaliser.t_max,
                             float(normaliser.denorm_conc(pd[i, 0])),
                             float(normaliser.denorm_conc(pd[i, 1]))])
            for i in range(target.n):
                wn.writerow([name, float(target.t_hours[i]),
                             float(target.cp_raw[i]), float(target.ct_raw[i]),
                             float(normaliser.denorm_conc(pn[i, 0])),
                             float(normaliser.denorm_conc(pn[i, 1]))])

    D = results["Configuration D"]
    t_coll = torch.linspace(0.0, 1.0, n_collocation, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    r_p, r_t, _, _ = physics_residuals(D.surrogate, D.params, t_coll, D.config.paradigm, normaliser.t_max)
    w_rba = rba_weights(r_p, r_t)
    with open(os.path.join(tables, "physics_diagnostics_D.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_hours", "r_p", "r_t", "rba_w"])
        for i in range(n_collocation):
            w.writerow([float(t_coll[i, 0]) * normaliser.t_max, float(r_p[i, 0].detach()),
                        float(r_t[i, 0].detach()), float(w_rba[i, 0])])


def reference_metrics() -> Dict[tuple, Dict[str, str]]:
    """The reported metric matrix shipped with the package."""
    ref = resources.files("folate_twin") / "reference" / "published_metrics.csv"
    with ref.open("r", newline="") as f:
        return {(row["config"], row["compartment"]): row for row in csv.DictReader(f)}


def write_reference_check(results: Dict[str, TrainResult], tables: str, verbose: bool = True) -> None:
    """``figure_data_check.csv``: this run's RMSE / R^2 next to the reported ones."""
    published = reference_metrics()
    with open(os.path.join(tables, "figure_data_check.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "compartment", "RMSE_published", "RMSE_retrained",
                    "R2_published", "R2_retrained"])
        for name, res in results.items():
            for comp in ("plasma", "tumour"):
                pub = published.get((name, comp), {})
                m = res.metrics[comp]
                w.writerow([name, comp, pub.get("RMSE", ""), m["RMSE"], pub.get("R2", ""), m["R2"]])
                if verbose:
                    print(f"  {name:16s} {comp:6s} RMSE reported={float(pub.get('RMSE', 'nan')):8.3f} "
                          f"this run={m['RMSE']:8.3f}   R2 reported={float(pub.get('R2', 'nan')):.4f} "
                          f"this run={m['R2']:.4f}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_pipeline(
    output_dir: str = "outputs",
    epochs: int = C.EPOCHS,
    n_collocation: int = C.N_COLLOCATION,
    rk4_steps: int = C.RK4_STEPS,
    log_every: int = 0,
    verbose: bool = True,
) -> Dict[str, TrainResult]:
    """Train A-D and write every table to ``<output_dir>/tables``."""
    tables = os.path.join(output_dir, "tables")
    os.makedirs(tables, exist_ok=True)
    normaliser, target = load_target()
    if verbose:
        print(f"Device={DEVICE}, dtype={DTYPE}, T_max={normaliser.t_max} h, "
              f"C_max={normaliser.c_max} ug/mL, epochs={epochs}, N_r={n_collocation}")

    results = run_ablation(target, normaliser, epochs=epochs, n_collocation=n_collocation,
                           log_every=log_every, history_every=C.HISTORY_EVERY, verbose=verbose)
    if verbose:
        print_report(results)

    write_metric_tables(results, tables)
    twin = write_twin_tables(results, target, normaliser, tables, rk4_steps=rk4_steps)
    write_figure_data(results, target, normaliser, tables, n_collocation=n_collocation)
    if verbose:
        print(f"\nDifferentiable RK4 twin: C_t(T_max)={twin['Ct_final_ugml']:.3f} ug/mL, "
              f"sensitivity dC_t/dC_0 = {twin['dCt_dC0']:.5f}")
        print("\nComparison with the reported metric matrix:")
    write_reference_check(results, tables, verbose=verbose)
    if verbose:
        print(f"\nTables written to: {tables}")
    return results
