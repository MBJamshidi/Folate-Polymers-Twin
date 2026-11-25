r"""Figures 47-54: multi-seed variance and out-of-sample validation.

Built on the same house style as figures 1-46.

* fig47-fig50  multi-seed variance analysis (Configurations C and D)
* fig51-fig54  leave-one-out / leave-one-time-point-out validation

Inputs (under ``<data_dir>/tables``, written by ``folate-twin robustness``):
``multiseed_trajectories.csv``, ``multiseed_metrics.csv``,
``multiseed_parameters.csv``, ``loocv_results.csv``,
``lotpo_trajectories.csv`` and ``robustness_summary.json``.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import pandas as pd

from ..data import target_array
from .style import COL, CONFIG_COLORS, L_CP, L_CT, L_TIME, export, finalize_axis, house_style, new_fig

DTYPE = np.float64
_TARGET = target_array()

SEEDS = (0, 42, 123, 2025, 2026)
ARM_LABEL = {
    "C": "Configuration C",
    "D": "Configuration D",
    "D_nophys": r"Physics-free control ($\lambda_p=0$)",
}
ARM_COLOR = {
    "C": CONFIG_COLORS["Configuration C"],
    "D": CONFIG_COLORS["Configuration D"],
    "D_nophys": COL["baseline"],
}
PARAM_KEYS = ("k10", "k_in", "k_out", "k_a", "C0_norm")
PARAM_SYM = {
    "k10": r"$k_{10}$",
    "k_in": r"$k_{\mathrm{in}}$",
    "k_out": r"$k_{\mathrm{out}}$",
    "k_a": r"$k_a$",
    "C0_norm": r"$C_0$",
}


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
class RobustnessData:
    """Loads the robustness tables; identical empirical markers to fig01-fig46."""

    def __init__(self, data_dir: str):
        self.dir = os.path.join(data_dir, "tables")
        self.t_data = _TARGET[:, 0].copy()
        self.cp_data = _TARGET[:, 1].copy()
        self.ct_data = _TARGET[:, 2].copy()
        # Display error bars of the robustness figures: max(4 % of the value, 4 ug/mL).
        self.cp_err = np.maximum(0.04 * self.cp_data, 4.0).astype(DTYPE)
        self.ct_err = np.maximum(0.04 * self.ct_data, 4.0).astype(DTYPE)

        self.ms_traj = self._csv("multiseed_trajectories.csv")
        self.ms_metrics = self._csv("multiseed_metrics.csv")
        self.ms_params = self._csv("multiseed_parameters.csv")
        self.cv = self._csv("loocv_results.csv")
        self.lotpo_traj = self._csv("lotpo_trajectories.csv")
        with open(os.path.join(self.dir, "robustness_summary.json")) as f:
            self.summary = json.load(f)

    def _csv(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found - run `folate-twin robustness` first."
            )
        return pd.read_csv(path)

    # -- derived views -- #
    def seed_stack(self, arm: str, col: str) -> tuple[np.ndarray, np.ndarray]:
        """(t_hours, (n_seeds, n_dense)) matrix of dense predictions."""
        sub = self.ms_traj[self.ms_traj["config"] == arm]
        t = np.asarray(sub[sub["seed"] == SEEDS[0]]["t_hours"], dtype=DTYPE)
        mat = np.vstack([
            np.asarray(sub[sub["seed"] == s][col], dtype=DTYPE) for s in SEEDS
        ])
        return t, mat

    def mean_r2_by_seed(self, arm: str) -> np.ndarray:
        sub = self.ms_metrics[(self.ms_metrics["config"] == arm)
                              & (self.ms_metrics["compartment"] == "plasma")]
        return np.asarray([float(sub[sub["seed"] == s]["mean_R2"].iloc[0]) for s in SEEDS],
                          dtype=DTYPE)

    def param_stats(self, arm: str, key: str) -> tuple[float, float]:
        sub = self.ms_params[self.ms_params["config"] == arm]
        v = np.asarray(sub[key], dtype=DTYPE)
        return float(v.mean()), float(v.std(ddof=1))

    def oos(self, protocol: str, arm: str, comp: str) -> pd.DataFrame:
        m = ((self.cv["protocol"] == protocol) & (self.cv["config"] == arm)
             & (self.cv["compartment"] == comp))
        return self.cv[m].sort_values("held_node")

    def lotpo_curve(self, arm: str, node: int, col: str) -> tuple[np.ndarray, np.ndarray]:
        sub = self.lotpo_traj[(self.lotpo_traj["config"] == arm)
                              & (self.lotpo_traj["held_node"] == node)]
        return (np.asarray(sub["t_hours"], dtype=DTYPE),
                np.asarray(sub[col], dtype=DTYPE))


# --------------------------------------------------------------------------- #
# Figures 47-50: multi-seed variance analysis
# --------------------------------------------------------------------------- #
def _envelope_panel(ax, d: RobustnessData, col: str, y_data, y_err, color, ylabel):
    t, mat = d.seed_stack("D", col)
    mu, sd = mat.mean(axis=0), mat.std(axis=0, ddof=1)
    for i in range(mat.shape[0]):
        ax.plot(t, mat[i], "-", color=color, linewidth=0.45, alpha=0.45,
                zorder=1, label="Individual seeds" if i == 0 else None)
    # The envelope is narrower than the stroke width, so the legend entry carries
    # its scale explicitly rather than leaving a reader to wonder whether a band
    # was drawn at all.
    pct = 100.0 * float((sd / np.maximum(mu, 1e-9)).max())
    ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.22, linewidth=0,
                    zorder=2,
                    label=rf"Mean $\pm$ SD ($n=5$; max {sd.max():.2f}, {pct:.2f}%)")
    ax.plot(t, mu, "-", color=color, linewidth=1.0, zorder=3, label="Seed mean")
    ax.errorbar(d.t_data, y_data, yerr=y_err, fmt="o", color=color, ecolor=color,
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, label="Empirical", zorder=4)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    finalize_axis(ax)


def fig47(d: RobustnessData, out: str) -> None:
    fig, ax = new_fig("single")
    _envelope_panel(ax, d, "Cp_pred_ugml", d.cp_data, d.cp_err, COL["plasma"], L_CP)
    export(fig, "fig47_seed_envelope_plasma_D", out)


def fig48(d: RobustnessData, out: str) -> None:
    fig, ax = new_fig("single")
    _envelope_panel(ax, d, "Ct_pred_ugml", d.ct_data, d.ct_err, COL["tumour"], L_CT)
    export(fig, "fig48_seed_envelope_tumour_D", out)


def fig49(d: RobustnessData, out: str) -> None:
    """Seed-resolved compartment-averaged R^2 for C and D, with Mean +/- SD."""
    fig, ax = new_fig("single")
    lo, hi = 1.0, 0.0
    for x, arm in ((0.0, "C"), (1.0, "D")):
        vals = d.mean_r2_by_seed(arm)
        jitter = np.linspace(-0.13, 0.13, len(vals))
        ax.plot(x + jitter, vals, "o", color=ARM_COLOR[arm], markersize=3,
                markerfacecolor="white", markeredgewidth=0.7, zorder=3,
                label=f"{ARM_LABEL[arm]} (seeds)")
        ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="s",
                    color=ARM_COLOR[arm], ecolor=ARM_COLOR[arm], elinewidth=0.9,
                    capsize=2.5, markersize=4, zorder=4)
        lo = min(lo, float(vals.min())); hi = max(hi, float(vals.max()))
    span = max(hi - lo, 1e-4)
    ax.set_ylim(lo - 0.95 * span, hi + 0.45 * span)   # headroom for annotation/legend
    p = d.summary["C_vs_D_paired"]
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["C", "D"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_xlabel("Configuration")
    ax.set_ylabel(r"Compartment-averaged $R^2$")
    ptxt = r"$p<0.001$" if p["p"] < 1e-3 else rf"$p={p['p']:.3f}$"
    ax.annotate(rf"paired $\Delta_{{D-C}}={p['mean_diff_D_minus_C']:+.4f}$, " + ptxt,
                xy=(0.5, 0.985), xycoords="axes fraction", ha="center", va="top",
                fontsize=8.5, color=COL["baseline"])
    ax.legend(loc="lower center", fontsize=8, handletextpad=0.4,
              borderpad=0.25, labelspacing=0.3)
    finalize_axis(ax)
    export(fig, "fig49_seed_metric_spread", out)


def fig50(d: RobustnessData, out: str) -> None:
    """Seed-to-seed reproducibility of the recovered kinetic constants.

    The estimates span two orders of magnitude (k_10 ~ 0.02 to k_a ~ 1.95), so on
    a shared absolute axis a sub-one-percent standard deviation is invisible.  The
    panel therefore plots the scale-free coefficient of variation, which is the
    quantity the reproducibility claim actually rests on.
    """
    fig, ax = new_fig("single")
    xs = np.arange(len(PARAM_KEYS), dtype=DTYPE)
    width = 0.36
    for i, arm in enumerate(("C", "D")):
        cv = np.array([100.0 * d.param_stats(arm, k)[1] / d.param_stats(arm, k)[0]
                       for k in PARAM_KEYS])
        ax.bar(xs + (i - 0.5) * width, cv, width=width, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=0.3, label=ARM_LABEL[arm])
    ax.set_xticks(xs)
    ax.set_xticklabels([PARAM_SYM[k] for k in PARAM_KEYS])
    ax.set_xlabel("Recovered quantity")
    ax.set_ylabel("Seed-to-seed CV (%)")
    ax.legend(loc="upper right", fontsize=8)
    finalize_axis(ax)
    export(fig, "fig50_seed_parameter_cv", out)


# --------------------------------------------------------------------------- #
# Figures 51-54: LOO / LOTPO out-of-sample validation
# --------------------------------------------------------------------------- #
def _lotpo_panel(ax, d: RobustnessData, col: str, y_data, y_err, color, ylabel):
    for node in range(8):
        t, y = d.lotpo_curve("D", node, col)
        ax.plot(t, y, "-", color=color, linewidth=0.45, alpha=0.40, zorder=1,
                label="Fold surrogates (8)" if node == 0 else None)
    comp = "plasma" if col.startswith("Cp") else "tumour"
    oos = d.oos("LOTPO", "D", comp)
    ax.plot(oos["t_hours"], oos["predicted"], "D", color=color, markersize=3.6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
            linestyle="none", zorder=4, label="Held-out prediction")
    ax.errorbar(d.t_data, y_data, yerr=y_err, fmt="o", color="black", ecolor="black",
                elinewidth=0.6, capsize=1.5, markersize=3, markerfacecolor="white",
                markeredgewidth=0.7, zorder=3, label="Empirical")
    ax.vlines(oos["t_hours"], oos["observed"], oos["predicted"], color=color,
              linewidth=0.6, alpha=0.8, zorder=2)
    ax.set_xlabel(L_TIME)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    finalize_axis(ax)


def fig51(d: RobustnessData, out: str) -> None:
    fig, ax = new_fig("single")
    _lotpo_panel(ax, d, "Cp_pred_ugml", d.cp_data, d.cp_err, COL["plasma"], L_CP)
    export(fig, "fig51_lotpo_plasma_D", out)


def fig52(d: RobustnessData, out: str) -> None:
    fig, ax = new_fig("single")
    _lotpo_panel(ax, d, "Ct_pred_ugml", d.ct_data, d.ct_err, COL["tumour"], L_CT)
    export(fig, "fig52_lotpo_tumour_D", out)


def fig53(d: RobustnessData, out: str) -> None:
    """Out-of-sample parity: observed vs withheld-node prediction."""
    fig, ax = new_fig("single")
    style = {"LOO": ("o", "LOO"), "LOTPO": ("s", "LOTPO")}
    lims = []
    for proto, (marker, pname) in style.items():
        for comp, color in (("plasma", COL["plasma"]), ("tumour", COL["tumour"])):
            sub = d.oos(proto, "D", comp)
            ax.plot(sub["observed"], sub["predicted"], marker, color=color,
                    markersize=3.2, markerfacecolor="white" if proto == "LOO" else color,
                    markeredgewidth=0.7, linestyle="none",
                    label=f"{pname}, {comp}")  # short labels keep the square panel legible
            lims += [sub["observed"].min(), sub["observed"].max(),
                     sub["predicted"].min(), sub["predicted"].max()]
    lo, hi = float(min(lims)), float(max(lims))
    pad = 0.06 * (hi - lo)
    line = np.array([lo - pad, hi + pad])
    ax.plot(line, line, "--", color=COL["baseline"], linewidth=0.7, zorder=1,
            label="Identity")
    ax.set_xlim(line[0], line[1])
    ax.set_ylim(line[0], line[1])
    ax.set_aspect("equal", adjustable="box")
    cvd = d.summary["cross_validation"]
    ax.annotate(rf"LOO $R^2_{{\mathrm{{cv}}}}={cvd['LOO_D']['pooled']['R2']:.3f}$" "\n"
                rf"LOTPO $R^2_{{\mathrm{{cv}}}}={cvd['LOTPO_D']['pooled']['R2']:.3f}$",
                xy=(0.04, 0.84), xycoords="axes fraction", fontsize=8.5,
                color=COL["baseline"], va="top")
    ax.set_xlabel(r"Observed $C$ ($\mu$g/mL)")
    ax.set_ylabel(r"Out-of-sample predicted $C$ ($\mu$g/mL)")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.4,
              borderpad=0.25, labelspacing=0.28)
    finalize_axis(ax)
    export(fig, "fig53_oos_parity_D", out)


def fig54(d: RobustnessData, out: str) -> None:
    """Node-resolved LOTPO error: physics-constrained arms vs physics-free control."""
    fig, ax = new_fig("single")
    nodes = np.arange(8, dtype=DTYPE)
    arms = ("D", "C", "D_nophys")
    width = 0.26
    for i, arm in enumerate(arms):
        err = np.array([
            0.5 * (float(d.oos("LOTPO", arm, "plasma")["abs_pct_error"].iloc[k])
                   + float(d.oos("LOTPO", arm, "tumour")["abs_pct_error"].iloc[k]))
            for k in range(8)
        ])
        ax.bar(nodes + (i - 1) * width, err, width=width, color=ARM_COLOR[arm],
               edgecolor="white", linewidth=0.3, label=ARM_LABEL[arm])
    # The two domain-endpoint folds (t = 0.083 and t = 24 h) are extrapolations
    # and carry errors an order of magnitude above the interior folds; a linear
    # ordinate would flatten the interior contrast that this panel exists to show.
    ax.set_yscale("log")
    ax.set_ylim(1.0, 400.0)
    ax.axvspan(-0.5, 0.5, color=COL["baseline"], alpha=0.07, linewidth=0, zorder=0)
    ax.axvspan(6.5, 7.5, color=COL["baseline"], alpha=0.07, linewidth=0, zorder=0)
    ax.annotate("endpoint\n(extrapolation)", xy=(7.0, 300.0), ha="center", va="top",
                fontsize=7, color=COL["baseline"])
    ax.set_xlim(-0.6, 7.6)
    ax.set_xticks(nodes)
    ax.set_xticklabels([f"{t:g}" for t in d.t_data], rotation=45, ha="right")
    ax.set_xlabel(r"Withheld time node $t_i$ (h)")
    ax.set_ylabel("Out-of-sample error (%)")
    ax.legend(loc="upper center", fontsize=7.5, ncol=1, handletextpad=0.4,
              borderpad=0.25, labelspacing=0.28)
    finalize_axis(ax)
    export(fig, "fig54_lotpo_error_profile", out)


# --------------------------------------------------------------------------- #
# Registry & driver
# --------------------------------------------------------------------------- #
FIGURES = [fig47, fig48, fig49, fig50, fig51, fig52, fig53, fig54]

# Each panel names the robustness runs it needs, so a partially completed suite
# renders every figure it can rather than failing on the first missing fold.
REQUIRES = {
    "fig47": ("multiseed:D",), "fig48": ("multiseed:D",),
    "fig49": ("multiseed:C", "multiseed:D"), "fig50": ("multiseed:C", "multiseed:D"),
    "fig51": ("lotpo:D",), "fig52": ("lotpo:D",),
    "fig53": ("lotpo:D", "loo:D"),
    "fig54": ("lotpo:D", "lotpo:C", "lotpo:D_nophys"),
}


def _available(d: RobustnessData) -> set:
    got = set()
    for arm in ("C", "D"):
        if not d.ms_traj.empty and (d.ms_traj["config"] == arm).any():
            got.add(f"multiseed:{arm}")
    for arm in ("C", "D", "D_nophys"):
        sub = d.lotpo_traj[d.lotpo_traj["config"] == arm]
        if sub["held_node"].nunique() == 8:
            got.add(f"lotpo:{arm}")
    if not d.cv.empty and ((d.cv["protocol"] == "LOO") & (d.cv["config"] == "D")).sum() == 16:
        got.add("loo:D")
    return got


def render_robustness_figures(data_dir: str = "outputs", output_dir: str | None = None,
                              verbose: bool = True) -> List[str]:
    """Render every robustness figure whose runs are available.

    Returns the names of the figures that were skipped because runs are missing.
    """
    output_dir = output_dir or os.path.join(data_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    with house_style():
        d = RobustnessData(data_dir)
        got = _available(d)
        if verbose:
            print(f"Available runs: {sorted(got)}")
        rendered, skipped = 0, []
        for i, fn in enumerate(FIGURES, start=47):
            need = [r for r in REQUIRES[fn.__name__] if r not in got]
            if need:
                skipped.append(f"fig{i:02d} (needs {', '.join(need)})")
                continue
            fn(d, output_dir)
            rendered += 1
            if verbose:
                print(f"  [fig{i:02d}] {fn.__name__} -> png + pdf")
    if verbose:
        print(f"Done: {rendered} rendered"
              + (f", {len(skipped)} skipped -> {'; '.join(skipped)}" if skipped else ""))
    return skipped
