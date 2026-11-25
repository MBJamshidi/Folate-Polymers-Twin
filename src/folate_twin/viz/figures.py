r"""Figures 1-46: Configuration D performance, ablation comparison and
digital-twin analyses.

Every series is read from the result tables; nothing is synthesised.  Inputs
(under ``<data_dir>/tables``):

=============================  ===============================================
``metric_evaluation_matrix``   ``folate-twin train``
``rk4_digital_twin``           ``folate-twin train``
``training_history``           ``folate-twin train``
``dense_predictions_all``      ``folate-twin train``
``node_predictions``           ``folate-twin train``
``physics_diagnostics_D``      ``folate-twin train``
``multiseed_parameters``       ``folate-twin robustness`` (seed-0 rates, Fig. 35)
=============================  ===============================================

Each figure is written as ``figNN_<name>.png`` (300 dpi) and ``.pdf``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import seaborn as sns

from ..config import C_MAX, CONFIG_ORDER, RK4_STEPS, T_MAX  # noqa: F401
from ..data import target_array
from .style import (
    COL, CONFIG_COLORS, L_CONC, L_CP, L_CT, L_TIME, SYM,
    export, finalize_axis, house_style, new_fig,
)

DTYPE = np.float64

# NumPy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

TRAIN_CMD = "folate-twin train"
ROBUST_CMD = "folate-twin robustness"


# --------------------------------------------------------------------------- #
# Data provider: result tables only
# --------------------------------------------------------------------------- #
@dataclass
class FigureData:
    # empirical
    t_data: np.ndarray
    cp_data: np.ndarray
    ct_data: np.ndarray
    cp_err: np.ndarray | None    # no replicate SDs are available -> None
    ct_err: np.ndarray | None
    # dense predictions per configuration
    t_dense: np.ndarray
    cp_dense: Dict[str, np.ndarray]
    ct_dense: Dict[str, np.ndarray]
    # RK4 differentiable twin
    t_rk4: np.ndarray
    cp_rk4: np.ndarray
    ct_rk4: np.ndarray
    dct_dc0: float
    dct_dc0_t: np.ndarray        # time-resolved sensitivity dC_t(t)/dC_0 on t_rk4
    # metrics
    metrics: Dict[str, Dict[str, Tuple[float, float, float]]]
    params_final: Dict[str, Dict[str, float]]
    node_pred: Dict[str, Dict[str, np.ndarray]]     # model evaluated at the data times
    # histories
    epochs: np.ndarray
    loss_hist: Dict[str, Dict[str, np.ndarray]]
    param_hist: Dict[str, Dict[str, np.ndarray]]
    lambda_hist: Dict[str, np.ndarray]
    # physics diagnostics (Configuration D)
    t_coll: np.ndarray
    r_p: np.ndarray
    r_t: np.ndarray
    rba_w: np.ndarray


class DataProvider:
    """Loads the result tables of a run directory into :class:`FigureData`."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = self._build()

    # -- file helpers -- #
    def _find(self, fname: str) -> str | None:
        for cand in (os.path.join(self.data_dir, "tables", fname),
                     os.path.join(self.data_dir, fname)):
            if os.path.isfile(cand):
                return cand
        return None

    def _require(self, fname: str, producer: str):
        import pandas as pd
        path = self._find(fname)
        if path is None:
            raise FileNotFoundError(f"{fname} not found under {self.data_dir}; run `{producer}` first.")
        return pd.read_csv(path)

    # -- assembly -- #
    def _build(self) -> FigureData:
        target = target_array()
        t_data = target[:, 0].copy()
        cp_data = target[:, 1].copy()
        ct_data = target[:, 2].copy()

        dense = self._require("dense_predictions_all.csv", TRAIN_CMD)
        nodes = self._require("node_predictions.csv", TRAIN_CMD)
        cp_dense: Dict[str, np.ndarray] = {}
        ct_dense: Dict[str, np.ndarray] = {}
        node_pred: Dict[str, Dict[str, np.ndarray]] = {}
        t_dense = None
        for cfg in CONFIG_ORDER:
            g = dense[dense["config"] == cfg].sort_values("t_hours")
            t_dense = g["t_hours"].to_numpy(DTYPE)
            cp_dense[cfg] = g["Cp_pred_ugml"].to_numpy(DTYPE)
            ct_dense[cfg] = g["Ct_pred_ugml"].to_numpy(DTYPE)
            n = nodes[nodes["config"] == cfg].sort_values("t_hours")
            node_pred[cfg] = {"plasma": n["Cp_pred_ugml"].to_numpy(DTYPE),
                              "tumour": n["Ct_pred_ugml"].to_numpy(DTYPE)}

        rk4 = self._require("rk4_digital_twin.csv", TRAIN_CMD)
        t_rk4 = rk4["t_hours"].to_numpy(DTYPE)
        cp_rk4 = rk4["Cp_rk4_ugml"].to_numpy(DTYPE)
        ct_rk4 = rk4["Ct_rk4_ugml"].to_numpy(DTYPE)

        metrics = self._load_metrics()

        hist = self._require("training_history.csv", TRAIN_CMD)
        loss_hist, param_hist, params_final = {}, {}, {}
        for cfg in CONFIG_ORDER:
            g = hist[hist["config"] == cfg].sort_values("epoch")
            loss_hist[cfg] = {k: g[k].to_numpy(DTYPE)
                              for k in ("epoch", "total", "data", "physics", "boundary")}
            rate_cols = [c for c in ("k10", "k_in", "k_out", "k_a", "C0", "k12", "k21")
                         if c in g and g[c].notna().all()]
            param_hist[cfg] = {k: g[k].to_numpy(DTYPE) for k in rate_cols}
            params_final[cfg] = {k: float(v[-1]) for k, v in param_hist[cfg].items()}
        gD = hist[hist["config"] == "Configuration D"].sort_values("epoch")
        epochs = gD["epoch"].to_numpy(DTYPE)
        lambda_hist = {"lam_d": gD["lambda_d"].to_numpy(DTYPE),
                       "lam_p": gD["lambda_p"].to_numpy(DTYPE),
                       "lam_b": gD["lambda_b"].to_numpy(DTYPE)}

        phys = self._require("physics_diagnostics_D.csv", TRAIN_CMD)
        t_coll = phys["t_hours"].to_numpy(DTYPE)
        r_p = phys["r_p"].to_numpy(DTYPE)
        r_t = phys["r_t"].to_numpy(DTYPE)
        rba_w = phys["rba_w"].to_numpy(DTYPE)

        dct_dc0_t = self._sensitivity_field(t_rk4)
        dct_dc0 = float(dct_dc0_t[-1])

        return FigureData(
            t_data=t_data, cp_data=cp_data, ct_data=ct_data, cp_err=None, ct_err=None,
            t_dense=t_dense, cp_dense=cp_dense, ct_dense=ct_dense,
            t_rk4=t_rk4, cp_rk4=cp_rk4, ct_rk4=ct_rk4, dct_dc0=dct_dc0, dct_dc0_t=dct_dc0_t,
            metrics=metrics, params_final=params_final, node_pred=node_pred,
            epochs=epochs, loss_hist=loss_hist, param_hist=param_hist, lambda_hist=lambda_hist,
            t_coll=t_coll, r_p=r_p, r_t=r_t, rba_w=rba_w,
        )

    def _sensitivity_field(self, t_grid):
        """dC_t(t)/dC_0 of the calibrated Configuration D twin.

        The absorption-driven field is linear in the state and in C_0, so the
        sensitivity obeys the same field with C_0 = 1 and zero initial state.
        It is integrated with the same 240-step RK4 scheme as
        :class:`~folate_twin.twin.DifferentiableRK4`, using the rates of the
        seed-0 Configuration D run.
        """
        tab = self._require("multiseed_parameters.csv", ROBUST_CMD)
        row = tab[(tab["config"] == "D") & (tab["seed"] == 0)].iloc[0]
        k10, kin, kout, ka = (float(row[k]) for k in ("k10", "k_in", "k_out", "k_a"))

        def f(t, y):
            return np.array([ka * np.exp(-ka * t) - (k10 + kin) * y[0],
                             kin * y[0] - kout * y[1]])

        steps = RK4_STEPS
        dt = T_MAX / steps
        y, t = np.zeros(2), 0.0
        grid, sens = [0.0], [0.0]
        for _ in range(steps):
            a = f(t, y)
            b = f(t + 0.5 * dt, y + 0.5 * dt * a)
            c = f(t + 0.5 * dt, y + 0.5 * dt * b)
            d = f(t + dt, y + dt * c)
            y = y + (dt / 6.0) * (a + 2 * b + 2 * c + d)
            t += dt
            grid.append(t)
            sens.append(y[1])
        return np.interp(t_grid, grid, sens).astype(DTYPE)

    def _load_metrics(self):
        df = self._require("metric_evaluation_matrix.csv", TRAIN_CMD)
        out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        for _, row in df.iterrows():
            cfg, comp = str(row["config"]), str(row["compartment"])
            out.setdefault(cfg, {})[comp] = (float(row["MSE"]), float(row["RMSE"]), float(row["R2"]))
        return out

    def pred_at_data(self, cfg: str, comp: str) -> np.ndarray:
        """Trained surrogate evaluated exactly at the eight sampling times."""
        return self.data.node_pred[cfg][comp]


# --------------------------------------------------------------------------- #
# Reusable plotting primitives
# --------------------------------------------------------------------------- #
def _fit_panel(ax, t_data, y_data, yerr, t_dense, y_dense, color, ylabel):
    ax.errorbar(t_data, y_data, yerr=yerr, fmt="o", color=color, ecolor=color,
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, label="Empirical", zorder=3)
    ax.plot(t_dense, y_dense, "-", color=color, linewidth=1.0, label="PINN prediction", zorder=2)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    finalize_axis(ax)


def _loss_panel(ax, epochs, series, color, label):
    ax.plot(epochs, series, "-", color=color, linewidth=1.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss (a.u.)")
    ax.legend(loc="best")
    finalize_axis(ax)


# --------------------------------------------------------------------------- #
# Figures 01-20: Configuration D in-depth performance
# --------------------------------------------------------------------------- #
def fig01(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _fit_panel(ax, d.t_data, d.cp_data, d.cp_err, d.t_dense, d.cp_dense["Configuration D"],
               COL["plasma"], L_CP)
    export(fig, "fig01_trajectory_plasma_D", out)


def fig02(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _fit_panel(ax, d.t_data, d.ct_data, d.ct_err, d.t_dense, d.ct_dense["Configuration D"],
               COL["tumour"], L_CT)
    export(fig, "fig02_trajectory_tumour_D", out)


def fig03(p, out):
    d = p.data
    fig, ax = new_fig("double", 3.0)
    ax.errorbar(d.t_data, d.cp_data, yerr=d.cp_err, fmt="o", color=COL["plasma"],
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, label=r"$C_p$ empirical")
    ax.plot(d.t_dense, d.cp_dense["Configuration D"], "-", color=COL["plasma"], label=r"$C_p$ PINN")
    ax.errorbar(d.t_data, d.ct_data, yerr=d.ct_err, fmt="s", color=COL["tumour"],
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, label=r"$C_t$ empirical")
    ax.plot(d.t_dense, d.ct_dense["Configuration D"], "-", color=COL["tumour"], label=r"$C_t$ PINN")
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(L_CONC)
    ax.legend(loc="best", ncol=2)
    finalize_axis(ax)
    export(fig, "fig03_trajectory_combined_D", out)


def fig04(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _loss_panel(ax, d.epochs, d.loss_hist["Configuration D"]["total"], COL["plasma"], SYM["L_total"])
    export(fig, "fig04_loss_total_D", out)


def fig05(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _loss_panel(ax, d.epochs, d.loss_hist["Configuration D"]["physics"], COL["accent"], SYM["L_phys"])
    export(fig, "fig05_loss_physics_D", out)


def fig06(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _loss_panel(ax, d.epochs, d.loss_hist["Configuration D"]["data"], COL["tumour"], SYM["L_data"])
    export(fig, "fig06_loss_data_D", out)


def fig07(p, out):
    d = p.data
    fig, ax = new_fig("single")
    _loss_panel(ax, d.epochs, d.loss_hist["Configuration D"]["boundary"], COL["rk4"], SYM["L_bound"])
    export(fig, "fig07_loss_boundary_D", out)


def fig08(p, out):
    d = p.data
    h = d.loss_hist["Configuration D"]
    fig, ax = new_fig("double", 3.0)
    ax.plot(d.epochs, h["total"], "-", color=COL["plasma"], label=SYM["L_total"], linewidth=1.1)
    ax.plot(d.epochs, h["data"], "-", color=COL["tumour"], label=SYM["L_data"])
    ax.plot(d.epochs, h["physics"], "-", color=COL["accent"], label=SYM["L_phys"])
    ax.plot(d.epochs, h["boundary"], "-", color=COL["rk4"], label=SYM["L_bound"])
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss (a.u.)")
    ax.legend(loc="best", ncol=2)
    finalize_axis(ax)
    export(fig, "fig08_loss_components_D", out)


def _param_track(p, out, key, sym, name):
    d = p.data
    hist = d.param_hist["Configuration D"][key]
    final = d.params_final["Configuration D"][key]
    fig, ax = new_fig("single")
    ax.plot(d.epochs, hist, "-", color=COL["plasma"], linewidth=1.0, label="Estimate")
    ax.axhline(final, color=COL["baseline"], linestyle="--", linewidth=0.7,
               label=f"Final = {final:.4f}")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(sym + r" (h$^{-1}$)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, name, out)


def fig09(p, out): _param_track(p, out, "k10", SYM["k10"], "fig09_param_k10_D")
def fig10(p, out): _param_track(p, out, "k_in", SYM["k_in"], "fig10_param_kin_D")
def fig11(p, out): _param_track(p, out, "k_out", SYM["k_out"], "fig11_param_kout_D")


def fig12(p, out):
    d = p.data
    h = d.param_hist["Configuration D"]
    fig, ax = new_fig("double", 3.0)
    for key, sym, c in (("k10", SYM["k10"], COL["plasma"]),
                        ("k_in", SYM["k_in"], COL["tumour"]),
                        ("k_out", SYM["k_out"], COL["accent"])):
        ax.plot(d.epochs, h[key], "-", color=c, linewidth=1.0, label=sym)
        ax.axhline(d.params_final["Configuration D"][key], color=c, linestyle=":", linewidth=0.6)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"Rate constant (h$^{-1}$)")
    ax.legend(loc="best", ncol=3)
    finalize_axis(ax)
    export(fig, "fig12_param_overlay_D", out)


def _lambda_fig(p, out, key, sym, name, color):
    d = p.data
    fig, ax = new_fig("single")
    ax.plot(d.epochs, d.lambda_hist[key], "-", color=color, linewidth=1.0, label=sym)
    ax.fill_between(d.epochs, d.lambda_hist[key], color=color, alpha=0.18)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Self-adaptive weight")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, name, out)


def fig13(p, out): _lambda_fig(p, out, "lam_d", SYM["lam_d"], "fig13_lambda_data_D", COL["tumour"])
def fig14(p, out): _lambda_fig(p, out, "lam_p", SYM["lam_p"], "fig14_lambda_physics_D", COL["accent"])
def fig15(p, out): _lambda_fig(p, out, "lam_b", SYM["lam_b"], "fig15_lambda_boundary_D", COL["rk4"])


def fig16(p, out):
    d = p.data
    fig, ax = new_fig("single")
    ax.plot(d.cp_dense["Configuration D"], d.ct_dense["Configuration D"], "-",
            color=COL["plasma"], linewidth=1.0, label="Twin trajectory")
    ax.scatter(d.cp_data, d.ct_data, s=14, facecolor="white", edgecolor=COL["tumour"],
               linewidth=0.8, zorder=3, label="Empirical")
    ax.set_xlabel(L_CP)
    ax.set_ylabel(L_CT)
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig16_phase_space_D", out)


def fig17(p, out):
    d = p.data
    fig, ax = new_fig("single")
    ax.axhline(0, color=COL["baseline"], linewidth=0.6)
    ax.plot(d.t_coll, d.r_p, "-", color=COL["plasma"], linewidth=0.9, label=SYM["rp"])
    ax.fill_between(d.t_coll, d.r_p, color=COL["plasma"], alpha=0.2)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(r"Residual $r_p$ (a.u.)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig17_residual_rp_D", out)


def fig18(p, out):
    d = p.data
    fig, ax = new_fig("single")
    ax.axhline(0, color=COL["baseline"], linewidth=0.6)
    ax.plot(d.t_coll, d.r_t, "-", color=COL["tumour"], linewidth=0.9, label=SYM["rt"])
    ax.fill_between(d.t_coll, d.r_t, color=COL["tumour"], alpha=0.2)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(r"Residual $r_t$ (a.u.)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig18_residual_rt_D", out)


def fig19(p, out):
    d = p.data
    mag = (np.abs(d.r_p) + np.abs(d.r_t)).reshape(1, -1)
    fig, ax = new_fig("double", 1.9)
    im = ax.imshow(mag, aspect="auto", cmap="rocket",
                   extent=[0, T_MAX, 0, 1], origin="lower")
    ax.set_yticks([])
    ax.set_xlabel(L_TIME)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Magnitude (a.u.)", fontsize=12)
    export(fig, "fig19_residual_magnitude_map_D", out)


def fig20(p, out):
    d = p.data
    fig, ax = new_fig("double", 3.0)
    ax.fill_between(d.t_coll, d.rba_w, color=COL["accent"], alpha=0.3)
    ax.plot(d.t_coll, d.rba_w, "-", color=COL["accent"], linewidth=0.9, label=r"$w_j$")
    ax.set_xlabel(L_TIME)
    ax.set_ylabel("RBA weight $w_j$")
    ax.legend(loc="best")
    finalize_axis(ax)
    # inset histogram of the weight distribution
    iax = ax.inset_axes([0.55, 0.45, 0.4, 0.45])
    iax.hist(d.rba_w, bins=24, color=COL["accent"], alpha=0.7, edgecolor="white", linewidth=0.3)
    iax.set_xlabel(r"$w_j$", fontsize=10)
    iax.set_ylabel("Count", fontsize=10)
    iax.tick_params(labelsize=9)
    finalize_axis(iax)
    export(fig, "fig20_rba_weights_D", out)


# --------------------------------------------------------------------------- #
# Figures 21-34: ablation study and cross-model comparison
# --------------------------------------------------------------------------- #
def _config_fit(p, out, cfg, comp, color, ylabel, y_data, y_err, name):
    d = p.data
    dense = d.cp_dense[cfg] if comp == "plasma" else d.ct_dense[cfg]
    fig, ax = new_fig("single")
    _fit_panel(ax, d.t_data, y_data, y_err, d.t_dense, dense, color, ylabel)
    export(fig, name, out)


def fig21(p, out):
    d = p.data
    _config_fit(p, out, "Configuration A", "plasma", COL["baseline"], L_CP,
                d.cp_data, d.cp_err, "fig21_configA_plasma")


def fig22(p, out):
    d = p.data
    _config_fit(p, out, "Configuration A", "tumour", COL["baseline"], L_CT,
                d.ct_data, d.ct_err, "fig22_configA_tumour")


def fig23(p, out):
    d = p.data
    _config_fit(p, out, "Configuration B", "plasma", CONFIG_COLORS["Configuration B"], L_CP,
                d.cp_data, d.cp_err, "fig23_configB_plasma")


def fig24(p, out):
    d = p.data
    _config_fit(p, out, "Configuration B", "tumour", CONFIG_COLORS["Configuration B"], L_CT,
                d.ct_data, d.ct_err, "fig24_configB_tumour")


def fig25(p, out):
    d = p.data
    _config_fit(p, out, "Configuration C", "plasma", CONFIG_COLORS["Configuration C"], L_CP,
                d.cp_data, d.cp_err, "fig25_configC_plasma")


def fig26(p, out):
    d = p.data
    _config_fit(p, out, "Configuration C", "tumour", CONFIG_COLORS["Configuration C"], L_CT,
                d.ct_data, d.ct_err, "fig26_configC_tumour")


def _overlay(p, out, comp, y_data, y_err, ylabel, name):
    d = p.data
    fig, ax = new_fig("double", 3.0)
    ax.errorbar(d.t_data, y_data, yerr=y_err, fmt="o", color="black", ecolor="black",
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, label="Empirical", zorder=5)
    for cfg in CONFIG_ORDER:
        dense = d.cp_dense[cfg] if comp == "plasma" else d.ct_dense[cfg]
        ax.plot(d.t_dense, dense, "-", color=CONFIG_COLORS[cfg], linewidth=1.0,
                label=cfg.replace("Configuration ", "Config "))
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", ncol=2)
    finalize_axis(ax)
    export(fig, name, out)


def fig27(p, out):
    d = p.data
    _overlay(p, out, "plasma", d.cp_data, d.cp_err, L_CP, "fig27_overlay_plasma")


def fig28(p, out):
    d = p.data
    _overlay(p, out, "tumour", d.ct_data, d.ct_err, L_CT, "fig28_overlay_tumour")


def _metric_bar(p, out, idx, ylabel, name):
    d = p.data
    configs = CONFIG_ORDER
    x = np.arange(len(configs))
    pl = [d.metrics[c]["plasma"][idx] for c in configs]
    tu = [d.metrics[c]["tumour"][idx] for c in configs]
    fig, ax = new_fig("double", 3.0)
    w = 0.38
    ax.bar(x - w / 2, pl, w, color=COL["plasma"], label="Plasma", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, tu, w, color=COL["tumour"], label="Tumour", edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Configuration ", "") for c in configs])
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, name, out)


def fig29(p, out): _metric_bar(p, out, 2, r"$R^2$", "fig29_bar_r2")
def fig30(p, out): _metric_bar(p, out, 1, "RMSE ($\\mu$g/mL)", "fig30_bar_rmse")
def fig31(p, out): _metric_bar(p, out, 0, r"MSE ($\mu$g$^2$/mL$^2$)", "fig31_bar_mse")


def _metric_heatmap(p, out, idx, name, fmt, cmap):
    d = p.data
    configs = CONFIG_ORDER
    mat = np.array([[d.metrics[c]["plasma"][idx], d.metrics[c]["tumour"][idx]] for c in configs], dtype=DTYPE)
    fig, ax = new_fig("single", 3.0)
    hm = sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, cbar_kws={"shrink": 0.7},
                     xticklabels=["Plasma", "Tumour"],
                     yticklabels=[c.replace("Configuration ", "Cfg ") for c in configs],
                     annot_kws={"size": 11}, linewidths=0.4, linecolor="white", ax=ax)
    ax.tick_params(labelsize=11)
    hm.collections[0].colorbar.ax.tick_params(labelsize=10)
    export(fig, name, out)


def fig32(p, out): _metric_heatmap(p, out, 2, "fig32_heatmap_r2", ".3f", "mako")
def fig33(p, out): _metric_heatmap(p, out, 1, "fig33_heatmap_rmse", ".1f", "rocket_r")


def fig34(p, out):
    d = p.data
    res = np.concatenate([
        p.pred_at_data("Configuration D", "plasma") - d.cp_data,
        p.pred_at_data("Configuration D", "tumour") - d.ct_data,
    ])
    fig, ax = new_fig("single")
    ax.hist(res, bins=10, color=COL["plasma"], alpha=0.65, edgecolor="white", linewidth=0.4, density=True)
    mu, sigma = float(res.mean()), float(res.std() + 1e-9)
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
            "-", color=COL["tumour"], linewidth=1.0, label="Gaussian fit")
    ax.set_xlabel(r"Residual ($\mu$g/mL)")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig34_residual_histogram_D", out)


# --------------------------------------------------------------------------- #
# Figures 35-46: digital-twin applications and agreement statistics
# --------------------------------------------------------------------------- #
def fig35(p, out):
    d = p.data
    field = d.dct_dc0_t
    fig, ax = new_fig("single")
    ax.plot(d.t_rk4, field, "-", color=COL["rk4"], linewidth=1.0)
    ax.fill_between(d.t_rk4, field, color=COL["rk4"], alpha=0.2)
    ax.plot(d.t_rk4[-1], d.dct_dc0, "o", color=COL["rk4"], markersize=3)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(r"$\partial \hat{C}_t / \partial C_0$")
    ax.text(0.97, 0.06, r"final $\partial \hat{C}_t/\partial C_0 = %.3f$" % d.dct_dc0,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
    finalize_axis(ax)
    export(fig, "fig35_sensitivity_field", out)


def fig36(p, out):
    d = p.data
    fig, ax = new_fig("double", 3.0)
    ax.plot(d.t_rk4, d.cp_rk4, "-", color=COL["plasma"], linewidth=1.0, label=r"$C_p$ RK4")
    ax.plot(d.t_rk4, d.ct_rk4, "-", color=COL["tumour"], linewidth=1.0, label=r"$C_t$ RK4")
    ax.scatter(d.t_data, d.cp_data, s=12, facecolor="white", edgecolor=COL["plasma"], linewidth=0.7, zorder=3)
    ax.scatter(d.t_data, d.ct_data, s=12, facecolor="white", edgecolor=COL["tumour"], linewidth=0.7, zorder=3)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(L_CONC)
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig36_rk4_trajectory", out)


def fig37(p, out):
    d = p.data
    obs = np.concatenate([d.cp_data, d.ct_data])
    pred = np.concatenate([p.pred_at_data("Configuration D", "plasma"),
                           p.pred_at_data("Configuration D", "tumour")])
    fig, ax = new_fig("single")
    lim = [0, max(obs.max(), pred.max()) * 1.08]
    ax.plot(lim, lim, "--", color=COL["baseline"], linewidth=0.7, label="Identity")
    ax.scatter(d.cp_data, p.pred_at_data("Configuration D", "plasma"), s=16,
               facecolor=COL["plasma"], edgecolor="white", linewidth=0.4, label="Plasma")
    ax.scatter(d.ct_data, p.pred_at_data("Configuration D", "tumour"), s=16,
               facecolor=COL["tumour"], edgecolor="white", linewidth=0.4, label="Tumour")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observed ($\mu$g/mL)")
    ax.set_ylabel(r"Predicted ($\mu$g/mL)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig37_observed_vs_predicted", out)


def fig38(p, out):
    d = p.data
    fig, ax = new_fig("double", 3.0)
    for cfg in CONFIG_ORDER:
        ax.plot(d.loss_hist[cfg]["epoch"], d.loss_hist[cfg]["total"], "-", color=CONFIG_COLORS[cfg],
                linewidth=1.0, label=cfg.replace("Configuration ", "Config "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(SYM["L_total"] + " (a.u.)")
    ax.legend(loc="best", ncol=2)
    finalize_axis(ax)
    export(fig, "fig38_loglog_loss_comparison", out)


def fig39(p, out):
    d = p.data
    rp = p.pred_at_data("Configuration D", "plasma") - d.cp_data
    rt = p.pred_at_data("Configuration D", "tumour") - d.ct_data
    fig, ax = new_fig("double", 3.0)
    ax.axhline(0, color=COL["baseline"], linewidth=0.6)
    ax.plot(d.t_data, rp, "-o", color=COL["plasma"], markersize=3, markerfacecolor="white",
            markeredgewidth=0.7, label="Plasma residual")
    ax.plot(d.t_data, rt, "-s", color=COL["tumour"], markersize=3, markerfacecolor="white",
            markeredgewidth=0.7, label="Tumour residual")
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(r"Residual ($\mu$g/mL)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig39_residual_timeseries", out)


def error_stats(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    """MAE, range-normalised RMSE and SMAPE (%)."""
    err = pred - obs
    mae = float(np.mean(np.abs(err)))
    nrmse = float(np.sqrt(np.mean(err ** 2)) / (obs.max() - obs.min()))
    smape = float(np.mean(2 * np.abs(err) / (np.abs(obs) + np.abs(pred))) * 100)
    return mae, nrmse, smape


def fig40(p, out):
    d = p.data
    pl = error_stats(d.cp_data, p.pred_at_data("Configuration D", "plasma"))
    tu = error_stats(d.ct_data, p.pred_at_data("Configuration D", "tumour"))
    labels = ["MAE", "NRMSE", "SMAPE (%)"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = new_fig("double", 3.0)
    ax.bar(x - w / 2, pl, w, color=COL["plasma"], label="Plasma", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, tu, w, color=COL["tumour"], label="Tumour", edgecolor="white", linewidth=0.4)
    for i, (a, b) in enumerate(zip(pl, tu)):
        ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error metric")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig40_grouped_error_metrics", out)


def information_criteria(p) -> Tuple[List[float], List[float]]:
    """Gaussian AIC and BIC per configuration from the pooled residual sum of
    squares (k = 3 rates for A/B, 4 rates + C_0 = 5 for C/D)."""
    d = p.data
    n = 2 * len(d.t_data)
    aic, bic = [], []
    for cfg in CONFIG_ORDER:
        k = 3 if cfg in ("Configuration A", "Configuration B") else 5
        rss = float(np.sum((p.pred_at_data(cfg, "plasma") - d.cp_data) ** 2)
                    + np.sum((p.pred_at_data(cfg, "tumour") - d.ct_data) ** 2))
        ll = n * np.log(rss / n)
        aic.append(ll + 2 * k)
        bic.append(ll + k * np.log(n))
    return aic, bic


def fig41(p, out):
    aic, bic = information_criteria(p)
    x = np.arange(len(CONFIG_ORDER))
    w = 0.38
    fig, ax = new_fig("double", 3.0)
    ax.bar(x - w / 2, aic, w, color=COL["plasma"], label="AIC", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, bic, w, color=COL["accent"], label="BIC", edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("Configuration ", "") for c in CONFIG_ORDER])
    ax.set_ylabel("Information criterion")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig41_aic_bic", out)


def _bland_altman(p, out, comp, color, y_data, name):
    pred = p.pred_at_data("Configuration D", comp)
    mean = (pred + y_data) / 2.0
    diff = pred - y_data
    md, sd = float(diff.mean()), float(diff.std() + 1e-9)
    fig, ax = new_fig("single")
    ax.scatter(mean, diff, s=16, facecolor=color, edgecolor="white", linewidth=0.4)
    ax.axhline(md, color=COL["baseline"], linewidth=0.8, label=f"Bias = {md:.1f}")
    ax.axhline(md + 1.96 * sd, color=color, linestyle="--", linewidth=0.7, label=r"$\pm$1.96 SD")
    ax.axhline(md - 1.96 * sd, color=color, linestyle="--", linewidth=0.7)
    ax.set_xlabel(r"Mean of obs & pred ($\mu$g/mL)")
    ax.set_ylabel(r"Pred $-$ obs ($\mu$g/mL)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, name, out)


def fig42(p, out):
    _bland_altman(p, out, "plasma", COL["plasma"], p.data.cp_data, "fig42_bland_altman_plasma")


def fig43(p, out):
    _bland_altman(p, out, "tumour", COL["tumour"], p.data.ct_data, "fig43_bland_altman_tumour")


def fig44(p, out):
    d = p.data
    abs_err = np.abs(np.concatenate([
        p.pred_at_data("Configuration D", "plasma") - d.cp_data,
        p.pred_at_data("Configuration D", "tumour") - d.ct_data,
    ]))
    xs = np.sort(abs_err)
    ecdf = np.arange(1, xs.size + 1) / xs.size
    fig, ax = new_fig("single")
    ax.step(xs, ecdf, where="post", color=COL["plasma"], linewidth=1.0)
    ax.fill_between(xs, ecdf, step="post", color=COL["plasma"], alpha=0.18)
    ax.set_xlabel(r"Absolute tracking error ($\mu$g/mL)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    finalize_axis(ax)
    export(fig, "fig44_ecdf_abs_error", out)


def lin_ccc(obs: np.ndarray, pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    cov = np.mean((obs - obs.mean()) * (pred - pred.mean()))
    return float(2 * cov / (obs.var() + pred.var() + (obs.mean() - pred.mean()) ** 2))


def fig45(p, out):
    d = p.data
    cp_hat = p.pred_at_data("Configuration D", "plasma")
    ct_hat = p.pred_at_data("Configuration D", "tumour")
    ccc_p, ccc_t = lin_ccc(d.cp_data, cp_hat), lin_ccc(d.ct_data, ct_hat)
    fig, ax = new_fig("single")
    lim = [0, max(d.cp_data.max(), d.ct_data.max(), cp_hat.max(), ct_hat.max()) * 1.08]
    ax.plot(lim, lim, "--", color=COL["baseline"], linewidth=0.7, label="Identity")
    ax.scatter(d.cp_data, cp_hat, s=16, facecolor=COL["plasma"], edgecolor="white",
               linewidth=0.4, label="Plasma", zorder=3)
    ax.scatter(d.ct_data, ct_hat, s=16, marker="s", facecolor=COL["tumour"], edgecolor="white",
               linewidth=0.4, label="Tumour", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observed ($\mu$g/mL)")
    ax.set_ylabel(r"Predicted ($\mu$g/mL)")
    ax.text(0.05, 0.95, "Lin's CCC\nplasma $= %.3f$\ntumour $= %.3f$" % (ccc_p, ccc_t),
            transform=ax.transAxes, ha="left", va="top", fontsize=11)
    ax.legend(loc="lower right")
    finalize_axis(ax)
    export(fig, "fig45_concordance_ccc", out)


def fig46(p, out):
    d = p.data
    # Exposure (trapezoidal AUC) of the twin trajectories vs the data.
    auc_cp_pred = float(_trapz(d.cp_dense["Configuration D"], d.t_dense))
    auc_ct_pred = float(_trapz(d.ct_dense["Configuration D"], d.t_dense))
    auc_cp_obs = float(_trapz(d.cp_data, d.t_data))
    auc_ct_obs = float(_trapz(d.ct_data, d.t_data))
    labels = [r"$AUC_{C_p}$", r"$AUC_{C_t}$"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = new_fig("double", 3.0)
    ax.bar(x - w / 2, [auc_cp_obs, auc_ct_obs], w, color=COL["baseline"],
           label="Empirical (trapz)", edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, [auc_cp_pred, auc_ct_pred], w, color=COL["plasma"],
           label="Twin (Configuration D)", edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"AUC ($\mu$g$\cdot$h/mL)")
    ax.legend(loc="best")
    finalize_axis(ax)
    export(fig, "fig46_auc_summary", out)


# --------------------------------------------------------------------------- #
# Registry & driver
# --------------------------------------------------------------------------- #
FIGURES: List[Callable] = [
    fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10,
    fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19, fig20,
    fig21, fig22, fig23, fig24, fig25, fig26, fig27, fig28, fig29, fig30,
    fig31, fig32, fig33, fig34, fig35, fig36, fig37, fig38, fig39, fig40,
    fig41, fig42, fig43, fig44, fig45, fig46,
]


def render_figures(data_dir: str = "outputs", output_dir: str | None = None, verbose: bool = True) -> int:
    """Render figures 1-46 from ``<data_dir>/tables`` into ``output_dir``
    (default ``<data_dir>/figures``).  Returns the number of figures written."""
    output_dir = output_dir or os.path.join(data_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    with house_style():
        provider = DataProvider(data_dir)
        if verbose:
            print(f"Rendering {len(FIGURES)} figures -> {output_dir}")
        for i, fn in enumerate(FIGURES, start=1):
            fn(provider, output_dir)
            if verbose:
                print(f"  [{i:02d}/{len(FIGURES)}] {fn.__name__} -> png + pdf")
    return len(FIGURES)
