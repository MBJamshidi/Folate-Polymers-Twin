"""Physics residuals (autograd) and Residual-Based Attention (RBA)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import T_MAX
from .models import KineticParameters


def physics_residuals(
    surrogate: nn.Module,
    params: KineticParameters,
    t_bar: torch.Tensor,
    paradigm: str,
    t_max: float = T_MAX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(r_p, r_t, C_p, C_t)`` on collocation points.

    Derivatives dC/d(t_bar) come from :func:`torch.autograd.grad`.  Because
    ``t_bar = t / T_max`` the chain rule gives ``dC/dt = (1/T_max) dC/d(t_bar)``,
    and this factor is applied explicitly.

    Phenomenological (proposed), with systemic absorption ``C_a = C_0 e^{-k_a t}``::

        r_p = (1/T_max) dC_p/dt_bar - k_a C_a + (k10 + k_in) C_p
        r_t = (1/T_max) dC_t/dt_bar - k_in C_p + k_out C_t

    Passive two-compartment (ablation baseline)::

        r_p = (1/T_max) dC_p/dt_bar + (k10 + k12) C_p - k21 C_t
        r_t = (1/T_max) dC_t/dt_bar - k12 C_p + k21 C_t
    """
    t_bar = t_bar.clone().requires_grad_(True)
    C = surrogate(t_bar)
    Cp, Ct = C[:, 0:1], C[:, 1:2]

    ones = torch.ones_like(Cp)
    dCp = torch.autograd.grad(Cp, t_bar, grad_outputs=ones, create_graph=True)[0]
    dCt = torch.autograd.grad(Ct, t_bar, grad_outputs=ones, create_graph=True)[0]

    k = params.rates()
    inv_tmax = 1.0 / t_max                       # chain-rule scaling

    if paradigm == "phenomenological":
        # Absorption state in real time t = t_bar * T_max:  C_a = C_0 e^{-k_a t}.
        C_a = params.c0() * torch.exp(-k["k_a"] * t_bar * t_max)
        r_p = inv_tmax * dCp - k["k_a"] * C_a + (k["k10"] + k["k_in"]) * Cp
        r_t = inv_tmax * dCt - k["k_in"] * Cp + k["k_out"] * Ct
    elif paradigm == "passive":
        r_p = inv_tmax * dCp + (k["k10"] + k["k12"]) * Cp - k["k21"] * Ct
        r_t = inv_tmax * dCt - k["k12"] * Cp + k["k21"] * Ct
    else:
        raise ValueError(f"unknown paradigm {paradigm!r}")

    return r_p, r_t, Cp, Ct


def rba_weights(r_p: torch.Tensor, r_t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Residual-Based Attention weights.

    ``w_j = (|r_p_j| + |r_t_j|) / (sum_k(|r_p_k| + |r_t_k|) + eps)``

    Computed from the current residuals at every step, so collocation emphasis
    moves toward high-error transient intervals.  Detached: the weights act as
    multipliers, not as a second gradient path.
    """
    mag = (r_p.abs() + r_t.abs()).detach()
    return mag / (mag.sum() + eps)
