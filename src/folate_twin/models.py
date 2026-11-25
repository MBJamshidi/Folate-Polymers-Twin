"""Neural surrogates (DGM and MLP) and the learnable kinetic parameters."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .config import DTYPE


class DGMLayer(nn.Module):
    """One Deep Galerkin Method gated cell (Sirignano & Spiliopoulos, 2018).

    .. code-block:: text

        Z = tanh(Wz t + Uz S + bz)
        G = tanh(Wg t + Ug S + bg)
        R = tanh(Wr t + Ur S + br)
        H = tanh(Wh t + Uh (S . R) + bh)
        S_next = (1 - G) . S + Z . H

    The gated update connections replace the plain feed-forward MLP and keep the
    gradients healthy across the stiff multi-phase clearance window.
    """

    def __init__(self, width: int, in_dim: int = 1):
        super().__init__()
        self.Wz, self.Uz, self.bz = self._gate(width, in_dim)
        self.Wg, self.Ug, self.bg = self._gate(width, in_dim)
        self.Wr, self.Ur, self.br = self._gate(width, in_dim)
        self.Wh, self.Uh, self.bh = self._gate(width, in_dim)

    @staticmethod
    def _gate(width: int, in_dim: int):
        W = nn.Linear(in_dim, width, bias=False)
        U = nn.Linear(width, width, bias=False)
        b = nn.Parameter(torch.zeros(width))
        return W, U, b

    def forward(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        Z = torch.tanh(self.Wz(t) + self.Uz(S) + self.bz)
        G = torch.tanh(self.Wg(t) + self.Ug(S) + self.bg)
        R = torch.tanh(self.Wr(t) + self.Ur(S) + self.br)
        H = torch.tanh(self.Wh(t) + self.Uh(S * R) + self.bh)
        return (1.0 - G) * S + Z * H            # gated S^i update


class DGMSurrogate(nn.Module):
    """Forward surrogate ``N_theta : t_bar -> softplus([C_p, C_t])``.

    The terminal linear map is wrapped in a smooth softplus so every
    concentration prediction is non-negative.
    """

    def __init__(self, width: int = 64, depth: int = 3):
        super().__init__()
        self.input = nn.Linear(1, width)
        self.layers = nn.ModuleList(DGMLayer(width, in_dim=1) for _ in range(depth))
        self.output = nn.Linear(width, 2)

    def forward(self, t_bar: torch.Tensor) -> torch.Tensor:
        S = torch.tanh(self.input(t_bar))
        for layer in self.layers:
            S = layer(t_bar, S)                 # gated recurrence over depth
        raw = self.output(S)
        return torch.nn.functional.softplus(raw)


class MLPSurrogate(nn.Module):
    """Standard feed-forward tanh MLP surrogate (ablation Configuration A)."""

    def __init__(self, width: int = 64, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t_bar: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(t_bar))


def build_surrogate(topology: str, width: int = 64, depth: int = 3) -> nn.Module:
    """Create the surrogate for a topology name (``"dgm"`` or ``"mlp"``)."""
    if topology == "dgm":
        return DGMSurrogate(width=width, depth=depth)
    if topology == "mlp":
        return MLPSurrogate(width=width, depth=depth)
    raise ValueError(f"unknown topology {topology!r}; expected 'dgm' or 'mlp'")


class KineticParameters(nn.Module):
    """Learnable rate vector with smooth positivity.

    ``{k10, k_in, k_out, k_a}`` for the phenomenological paradigm or
    ``{k10, k12, k21}`` for the passive baseline, each constrained positive by
    ``k_m = softplus(kappa_m) = ln(1 + exp(kappa_m))``.

    The phenomenological paradigm also carries a learnable systemic-input
    amplitude ``C_0`` that parameterises the absorption state
    ``C_a(t) = C_0 exp(-k_a t)``.  This channel reproduces the observed rising
    plasma limb (267 -> 357 -> 400 ug/mL over t in [0.083, 1] h) that a
    decay-only law cannot.
    """

    NAMES_PHENO = ("k10", "k_in", "k_out", "k_a")
    NAMES_PASSIVE = ("k10", "k12", "k21")

    def __init__(self, paradigm: str, init: float = 0.3, c0_init: float = 0.5):
        super().__init__()
        if paradigm not in ("phenomenological", "passive"):
            raise ValueError(f"unknown paradigm {paradigm!r}")
        self.paradigm = paradigm
        self.names = self.NAMES_PHENO if paradigm == "phenomenological" else self.NAMES_PASSIVE
        # invert softplus so the initial *positive* rate equals `init`
        kappa0 = math.log(math.expm1(init))
        self.kappa = nn.Parameter(torch.full((len(self.names),), kappa0, dtype=DTYPE))
        if paradigm == "phenomenological":
            self.c0_raw = nn.Parameter(torch.tensor(math.log(math.expm1(c0_init)), dtype=DTYPE))
        else:
            self.register_parameter("c0_raw", None)

    def rates(self) -> Dict[str, torch.Tensor]:
        """Positive rate constants (1/h) as autograd tensors."""
        k = torch.nn.functional.softplus(self.kappa)
        return {name: k[i] for i, name in enumerate(self.names)}

    def c0(self) -> Optional[torch.Tensor]:
        """Normalised systemic-input amplitude C_0 (phenomenological only)."""
        if self.c0_raw is None:
            return None
        return torch.nn.functional.softplus(self.c0_raw)

    def detached(self) -> Dict[str, float]:
        """Current rates (and C_0) as plain floats."""
        out = {n: float(v.detach()) for n, v in self.rates().items()}
        if self.c0_raw is not None:
            out["C0"] = float(self.c0().detach())
        return out
