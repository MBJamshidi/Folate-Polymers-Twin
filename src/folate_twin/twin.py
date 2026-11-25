"""Differentiable fourth-order Runge-Kutta digital twin and dose sensitivity."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import C_MAX, DEVICE, DTYPE, RK4_STEPS, T_MAX
from .models import KineticParameters

# Empirical initial state (first sampling node), normalised by C_MAX.
CP_INIT = 267.42 / C_MAX
CT_INIT = 239.50 / C_MAX


class DifferentiableRK4(nn.Module):
    """Differentiable RK4 integrator of the absorption-driven influx-outflux
    field, in real time (hours)::

        C_a(t) = C_0 e^{-k_a t}
        f_p    = k_a C_a - (k10 + k_in) C_p
        f_t    = k_in C_p - k_out C_t

    Every RK4 stage stays on the autograd tape, so gradients flow to the kinetic
    parameters and to the dose amplitude ``C_0``.  Requires phenomenological
    :class:`~folate_twin.models.KineticParameters`.
    """

    def __init__(self, params: KineticParameters, t_max: float = T_MAX, steps: int = RK4_STEPS):
        super().__init__()
        if params.paradigm != "phenomenological":
            raise ValueError("DifferentiableRK4 needs phenomenological kinetic parameters")
        self.params = params
        self.t_max = t_max
        self.steps = steps

    def field(self, t: torch.Tensor, y: torch.Tensor, c0: torch.Tensor) -> torch.Tensor:
        k = self.params.rates()
        Cp, Ct = y[..., 0], y[..., 1]
        C_a = c0 * torch.exp(-k["k_a"] * t)               # systemic absorption input
        fp = k["k_a"] * C_a - (k["k10"] + k["k_in"]) * Cp
        ft = k["k_in"] * Cp - k["k_out"] * Ct
        return torch.stack((fp, ft), dim=-1)

    def rollout(
        self,
        c0: torch.Tensor,
        cp_init: float = CP_INIT,
        ct_init: float = CT_INIT,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate from ``[C_p(0), C_t(0)]`` over ``[0, T_max]`` with absorption
        amplitude ``c0`` (normalised).  Returns ``(t_grid, traj)`` where ``traj``
        has shape ``(steps + 1, 2)`` in normalised concentration units."""
        dt = self.t_max / self.steps
        y = torch.stack((
            torch.as_tensor(cp_init, dtype=DTYPE, device=c0.device) + 0.0 * c0,
            torch.as_tensor(ct_init, dtype=DTYPE, device=c0.device) + 0.0 * c0,
        ), dim=-1)
        traj = [y]
        t = torch.zeros((), dtype=DTYPE, device=c0.device)
        for _ in range(self.steps):
            k1 = self.field(t, y, c0)
            k2 = self.field(t + 0.5 * dt, y + 0.5 * dt * k1, c0)
            k3 = self.field(t + 0.5 * dt, y + 0.5 * dt * k2, c0)
            k4 = self.field(t + dt, y + dt * k3, c0)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
            traj.append(y)
        t_grid = torch.linspace(0.0, self.t_max, self.steps + 1, dtype=DTYPE, device=c0.device)
        return t_grid, torch.stack(traj, dim=0)

    def state_sensitivity(self, c0_value: float) -> Tuple[float, float]:
        """Return ``(C_t(T_max), dC_t(T_max)/dC_0)`` by autograd through the
        rollout, both in normalised units (the derivative is dimensionless)."""
        c0 = torch.tensor(c0_value, dtype=DTYPE, device=DEVICE, requires_grad=True)
        _, traj = self.rollout(c0)
        ct_final = traj[-1, 1]
        dct_dc0 = torch.autograd.grad(ct_final, c0)[0]
        return float(ct_final.detach()), float(dct_dc0)
