"""Tracking metrics (MSE, RMSE, R^2) evaluated on the live surrogate."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .data import Normaliser, TargetMatrix


def _metrics(pred_c: torch.Tensor, true_c: torch.Tensor) -> Dict[str, float]:
    err = pred_c - true_c
    mse = torch.mean(err ** 2)
    rmse = torch.sqrt(mse)
    ss_res = torch.sum(err ** 2)
    ss_tot = torch.sum((true_c - true_c.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return {"MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}


def compute_metrics(
    surrogate: nn.Module,
    target: TargetMatrix,
    normaliser: Normaliser,
) -> Dict[str, Dict[str, float]]:
    """MSE / RMSE / R^2 per compartment, in ug/mL, at the eight data nodes.

    Returns ``{"plasma": {...}, "tumour": {...}}``.  The surrogate is returned to
    training mode afterwards.
    """
    surrogate.eval()
    with torch.no_grad():
        pred = surrogate(target.t_bar)
        cp_hat = normaliser.denorm_conc(pred[:, 0:1])
        ct_hat = normaliser.denorm_conc(pred[:, 1:2])
    surrogate.train()
    return {
        "plasma": _metrics(cp_hat, target.cp_raw),
        "tumour": _metrics(ct_hat, target.ct_raw),
    }
