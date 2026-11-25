"""Inverse-PINN training engine (joint optimisation of network weights and
kinetic parameters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from . import config as C
from .config import DEVICE, DTYPE, AblationConfig
from .data import Normaliser, TargetMatrix
from .metrics import compute_metrics
from .models import KineticParameters, build_surrogate
from .physics import physics_residuals, rba_weights
from .weighting import AdaptiveLossWeights


@dataclass
class TrainResult:
    """Everything produced by one training run."""

    config: AblationConfig
    surrogate: nn.Module
    params: KineticParameters
    metrics: Dict[str, Dict[str, float]]
    final_rates: Dict[str, float]
    loss_history: List[float] = field(default_factory=list)
    param_history: List[Dict[str, float]] = field(default_factory=list)
    lambdas: Dict[str, float] = field(default_factory=dict)
    # Per-epoch diagnostics, filled only when history_every > 0.
    history: List[Dict[str, float]] = field(default_factory=list)


def train_configuration(
    cfg: AblationConfig,
    target: TargetMatrix,
    normaliser: Normaliser,
    epochs: int = C.EPOCHS,
    n_collocation: int = C.N_COLLOCATION,
    width: int = C.WIDTH,
    depth: int = C.DEPTH,
    lr: float = C.LEARNING_RATE,
    log_every: int = 0,
    seed: int = C.SEED,
    obs_mask: Optional[torch.Tensor] = None,
    bound_node: int = 0,
    bound_at_zero: bool = True,
    bound_mask: Tuple[bool, bool] = (True, True),
    physics_weight: float = 1.0,
    history_every: int = 0,
) -> TrainResult:
    """Train one ablation arm.

    With the default arguments this is the reference run behind the reported
    results.  The trailing arguments are used by the robustness protocols and
    leave the architecture, the residual law and the parameterisation untouched.

    Parameters
    ----------
    seed : RNG seed for weight initialisation.
    obs_mask : ``(N, 2)`` tensor selecting which of the 16 scalar observations
        enter the data loss.  ``None`` uses all of them.
    bound_node : index of the data node supplying the initial-condition anchor.
    bound_at_zero : impose the anchor at ``t_bar = 0`` (default) or at the
        anchor node's own normalised time.
    bound_mask : which compartments the boundary loss constrains.
    physics_weight : multiplier on the physics loss (0.0 = physics-free control).
    history_every : if > 0, log losses, multipliers and rates every
        ``history_every`` epochs into ``result.history``.  Logging is read-only
        and does not change the training trajectory.
    """
    torch.manual_seed(seed)

    surrogate = build_surrogate(cfg.topology, width, depth).to(DEVICE)
    params = KineticParameters(cfg.paradigm).to(DEVICE)
    adaptive = AdaptiveLossWeights(enabled=cfg.adaptive).to(DEVICE)

    optimiser = torch.optim.Adam(
        list(surrogate.parameters()) + list(params.parameters()) + list(adaptive.parameters()),
        lr=lr,
    )

    # N_r dense collocation grid over the scaled domain [0, 1].
    t_coll = torch.linspace(0.0, 1.0, n_collocation, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

    # Boundary anchor: C_p(0) = first-measurement plasma, C_t(0) = first-measurement
    # tumour loading.  The tumour already registers a strong EPR-mediated
    # concentration at the first sampling node (239.5 ug/mL at t = 0.083 h), so the
    # empirical loading is used instead of an idealised C_t(0) = 0.
    if bound_at_zero:
        t0 = torch.zeros((1, 1), dtype=DTYPE, device=DEVICE)
    else:
        t0 = target.t_bar[bound_node:bound_node + 1].detach().clone()
    cp0_bar = (target.cp_raw[bound_node] / normaliser.c_max).detach()
    ct0_bar = (target.ct_raw[bound_node] / normaliser.c_max).detach()

    result = TrainResult(cfg, surrogate, params, {}, {})

    for epoch in range(epochs):
        optimiser.zero_grad()

        # --- data loss L_d -------------------------------------------------- #
        pred_data = surrogate(target.t_bar)
        sq_err = torch.cat(
            ((pred_data[:, 0:1] - target.cp_bar) ** 2,
             (pred_data[:, 1:2] - target.ct_bar) ** 2), dim=1)
        if obs_mask is None:
            l_data = torch.mean(sq_err.sum(dim=1))
        else:
            # Same normalisation as the unmasked case (2 * sum / #retained).
            l_data = 2.0 * torch.sum(sq_err * obs_mask) / obs_mask.sum()

        # --- physics residuals + optional RBA ------------------------------- #
        r_p, r_t, _, _ = physics_residuals(surrogate, params, t_coll, cfg.paradigm, normaliser.t_max)
        sq = r_p ** 2 + r_t ** 2
        if cfg.attention:
            w = rba_weights(r_p, r_t)
            l_phys = torch.sum(w * sq)
        else:
            l_phys = torch.mean(sq)
        if physics_weight != 1.0:
            l_phys = physics_weight * l_phys

        # --- boundary / initial-condition loss L_b -------------------------- #
        c_at_0 = surrogate(t0)
        l_bound = 0.0 * c_at_0[0, 0]              # keep L_b on the autograd tape
        if bound_mask[0]:
            l_bound = l_bound + (c_at_0[0, 0] - cp0_bar) ** 2
        if bound_mask[1]:
            l_bound = l_bound + (c_at_0[0, 1] - ct0_bar) ** 2

        # --- self-adaptive combination -------------------------------------- #
        adaptive.maybe_update(l_data, l_phys, l_bound, list(surrogate.parameters()), epoch)
        loss = adaptive.combine(l_data, l_phys, l_bound)
        loss.backward()
        optimiser.step()

        result.loss_history.append(float(loss.detach()))
        if history_every and (epoch % history_every == 0 or epoch == epochs - 1):
            lam = adaptive.lambdas()
            row = {"epoch": epoch + 1, "total": float(loss.detach()),
                   "data": float(l_data.detach()), "physics": float(l_phys.detach()),
                   "boundary": float(l_bound.detach()), "lambda_d": lam["lambda_d"],
                   "lambda_p": lam["lambda_p"] if cfg.adaptive else 1.0,
                   "lambda_b": lam["lambda_b"] if cfg.adaptive else 1.0}
            row.update(params.detached())
            result.history.append(row)
        if epoch % max(1, epochs // 50) == 0:
            result.param_history.append(params.detached())
        if log_every and epoch % log_every == 0:
            print(f"  [{cfg.name}] epoch {epoch:5d}  loss={float(loss.detach()):.5e}", flush=True)

    result.metrics = compute_metrics(surrogate, target, normaliser)
    result.final_rates = params.detached()
    result.lambdas = adaptive.lambdas()
    return result


def run_ablation(
    target: TargetMatrix,
    normaliser: Normaliser,
    epochs: int = C.EPOCHS,
    n_collocation: int = C.N_COLLOCATION,
    log_every: int = 0,
    history_every: int = 0,
    verbose: bool = True,
) -> Dict[str, TrainResult]:
    """Train Configurations A-D (each seeded independently with seed 0)."""
    results: Dict[str, TrainResult] = {}
    for cfg in C.ABLATION:
        if verbose:
            print(f"\n=== Training {cfg.name}: paradigm={cfg.paradigm}, topology={cfg.topology}, "
                  f"RBA={cfg.attention}, self-adaptive={cfg.adaptive} ===", flush=True)
        results[cfg.name] = train_configuration(
            cfg, target, normaliser, epochs=epochs,
            n_collocation=n_collocation, log_every=log_every,
            history_every=history_every,
        )
    return results
