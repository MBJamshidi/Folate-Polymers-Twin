"""Assemble the eight multi-panel article figures from the individual panels.

Each article figure is a grid of the single-panel figures produced by
:mod:`folate_twin.viz.figures`; this module tiles their PNG files row by row
(every cell in a row has the same width) into ``figure_1.png`` ... ``figure_8.png``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

from PIL import Image

# Article figure -> rows of panel file stems.
PANEL_LAYOUT: Dict[int, List[List[str]]] = {
    1: [["fig01_trajectory_plasma_D", "fig02_trajectory_tumour_D", "fig03_trajectory_combined_D"]],
    2: [["fig04_loss_total_D", "fig05_loss_physics_D", "fig06_loss_data_D",
         "fig07_loss_boundary_D", "fig08_loss_components_D"]],
    3: [["fig09_param_k10_D", "fig10_param_kin_D", "fig11_param_kout_D", "fig12_param_overlay_D"]],
    4: [["fig13_lambda_data_D", "fig14_lambda_physics_D", "fig15_lambda_boundary_D"]],
    5: [["fig16_phase_space_D", "fig17_residual_rp_D", "fig18_residual_rt_D",
         "fig19_residual_magnitude_map_D", "fig20_rba_weights_D"]],
    6: [["fig21_configA_plasma", "fig22_configA_tumour", "fig23_configB_plasma", "fig24_configB_tumour"],
        ["fig25_configC_plasma", "fig26_configC_tumour", "fig27_overlay_plasma", "fig28_overlay_tumour"]],
    7: [["fig29_bar_r2", "fig30_bar_rmse", "fig31_bar_mse"],
        ["fig32_heatmap_r2", "fig33_heatmap_rmse", "fig34_residual_histogram_D"]],
    8: [["fig35_sensitivity_field", "fig36_rk4_trajectory", "fig37_observed_vs_predicted",
         "fig38_loglog_loss_comparison"],
        ["fig39_residual_timeseries", "fig40_grouped_error_metrics", "fig41_aic_bic",
         "fig42_bland_altman_plasma"],
        ["fig43_bland_altman_tumour", "fig44_ecdf_abs_error", "fig45_concordance_ccc",
         "fig46_auc_summary"]],
}

PANEL_TITLES: Dict[int, str] = {
    1: "Configuration D state reconstruction",
    2: "Multi-objective convergence",
    3: "Kinetic-parameter identification",
    4: "Self-adaptive loss weights",
    5: "Physics diagnostics and residual-based attention",
    6: "Ablation fits (Configurations A-D)",
    7: "Cross-model error statistics",
    8: "Digital-twin deployment and agreement analysis",
}


def _row(paths: Sequence[str], width: int, gap: int) -> Image.Image:
    n = len(paths)
    cell = (width - gap * (n - 1)) // n
    tiles = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            h = round(im.height * cell / im.width)
            tiles.append(im.resize((cell, h), Image.LANCZOS))
    height = max(t.height for t in tiles)
    row = Image.new("RGB", (width, height), "white")
    x = 0
    for t in tiles:
        row.paste(t, (x, (height - t.height) // 2))
        x += cell + gap
    return row


def compose_figure(number: int, panel_dir: str, output_path: str,
                   width: int = 3000, gap: int = 30) -> str:
    """Tile the panels of article figure ``number`` into ``output_path``."""
    rows = []
    for stems in PANEL_LAYOUT[number]:
        paths = [os.path.join(panel_dir, f"{s}.png") for s in stems]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"missing panels for figure {number}: {missing}; "
                                    f"run `folate-twin figures` first.")
        rows.append(_row(paths, width, gap))
    height = sum(r.height for r in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + gap
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, optimize=True)
    return output_path


def compose_all(panel_dir: str, output_dir: str, verbose: bool = True) -> List[str]:
    """Write ``figure_1.png`` ... ``figure_8.png`` into ``output_dir``."""
    written = []
    for n in sorted(PANEL_LAYOUT):
        path = compose_figure(n, panel_dir, os.path.join(output_dir, f"figure_{n}.png"))
        written.append(path)
        if verbose:
            print(f"  figure {n}: {PANEL_TITLES[n]} -> {path}")
    return written
