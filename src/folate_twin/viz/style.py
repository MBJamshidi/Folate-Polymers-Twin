"""House style shared by every figure: column widths, palette, type sizes and
paired PNG (300 dpi) / PDF export."""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from typing import Iterator, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import CONFIG_ORDER  # noqa: F401  (re-exported for the figure modules)

MM = 1.0 / 25.4
SINGLE_COL = 89.0 * MM                       # 3.5 in
DOUBLE_COL = 183.0 * MM                      # 7.2 in

# Colourblind-safe, muted palette.
COL = {
    "plasma": "#13315c",     # deep navy
    "tumour": "#9e2a2b",     # muted crimson
    "baseline": "#5c6b73",   # slate gray
    "rk4": "#2a7f7f",        # muted teal
    "accent": "#a47148",     # muted bronze
    "ci": "#13315c",
}
CONFIG_COLORS = {
    "Configuration A": "#6b705c",   # olive gray
    "Configuration B": "#577590",   # slate blue
    "Configuration C": "#b08968",   # muted tan
    "Configuration D": "#13315c",   # deep navy
}

# Axis labels (matplotlib mathtext, no external LaTeX needed).
L_TIME = r"Time $t$ (h)"
L_CONC = r"Concentration $C$ ($\mu$g/mL)"
L_CP = r"Plasma $C_p$ ($\mu$g/mL)"
L_CT = r"Tumour $C_t$ ($\mu$g/mL)"
SYM = {
    "k10": r"$k_{10}$",
    "k_in": r"$k_{\mathrm{in}}$",
    "k_out": r"$k_{\mathrm{out}}$",
    "lam_d": r"$\lambda_{\mathrm{data}}$",
    "lam_p": r"$\lambda_{\mathrm{physics}}$",
    "lam_b": r"$\lambda_{\mathrm{boundary}}$",
    "L_total": r"$\mathcal{L}_{\mathrm{total}}$",
    "L_data": r"$\mathcal{L}_{\mathrm{data}}$",
    "L_phys": r"$\mathcal{L}_{\mathrm{physics}}$",
    "L_bound": r"$\mathcal{L}_{\mathrm{boundary}}$",
    "rp": r"$r_p$",
    "rt": r"$r_t$",
}


def _resolve_sans() -> str:
    """First available of Helvetica / Arial / DejaVu Sans."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in ("Helvetica", "Arial", "DejaVu Sans"):
        if name in installed:
            return name
    return "DejaVu Sans"


def apply_house_style() -> None:
    """Set the rcParams of the house style (global; prefer :func:`house_style`)."""
    sns.set_theme(context="paper", style="ticks")
    family = _resolve_sans()
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,                  # editable TrueType in PDF
        "ps.fonttype": 42,
    })


@contextmanager
def house_style() -> Iterator[None]:
    """Apply the house style for the duration of a ``with`` block.

    rcParams, warning filters and the matplotlib log level are restored on exit,
    so rendering figures leaves the caller's matplotlib state untouched.
    """
    logger = logging.getLogger("matplotlib")
    level = logger.level
    with warnings.catch_warnings(), matplotlib.rc_context():
        warnings.simplefilter("ignore")
        logger.setLevel(logging.ERROR)
        try:
            apply_house_style()
            yield
        finally:
            logger.setLevel(level)


def new_fig(width: str = "single", height_in: float = 2.6) -> Tuple[plt.Figure, plt.Axes]:
    """Single- (89 mm) or double-column (183 mm) figure with one axis."""
    w = SINGLE_COL if width == "single" else DOUBLE_COL
    fig, ax = plt.subplots(figsize=(w, height_in))
    return fig, ax


def finalize_axis(ax: plt.Axes) -> None:
    sns.despine(ax=ax)


def export(fig: plt.Figure, name: str, output_dir: str) -> None:
    """Save ``<name>.png`` (300 dpi) and ``<name>.pdf`` and close the figure."""
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), format="pdf",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
