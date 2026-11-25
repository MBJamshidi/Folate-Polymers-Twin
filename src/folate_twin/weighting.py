"""Self-adaptive multi-objective loss weighting (gradient-norm balancing)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .config import DTYPE


class AdaptiveLossWeights(nn.Module):
    """Gradient-norm balancing of the physics and boundary terms
    (Wang, Teng & Perdikaris, 2021).

    The data term is the anchor (``lambda_d = 1``).  The physics and boundary
    multipliers are driven toward the ratio of mean back-propagated gradient
    magnitudes::

        lambda_hat_i = mean|grad_theta L_data| / mean|grad_theta L_i|

    clamped to ``clamp`` and updated as an exponential running average every
    ``update_every`` steps.  When ``enabled`` is False the three terms are summed
    with unit weights (Configurations A-C).
    """

    def __init__(
        self,
        enabled: bool = True,
        alpha: float = 0.9,
        update_every: int = 25,
        clamp: Tuple[float, float] = (0.1, 1.0),
    ):
        super().__init__()
        self.enabled = enabled
        self.alpha = alpha
        self.update_every = update_every
        self.clamp = clamp
        # Buffers (not Parameters): updated by the balancing rule, not by Adam.
        self.register_buffer("lambda_p", torch.tensor(1.0, dtype=DTYPE))
        self.register_buffer("lambda_b", torch.tensor(1.0, dtype=DTYPE))

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        total = torch.zeros((), dtype=DTYPE, device=loss.device)
        count = 0
        for g in grads:
            if g is not None:
                total = total + g.abs().sum()
                count += g.numel()
        return total / max(count, 1)

    def maybe_update(self, l_data, l_phys, l_bound, net_params, step: int) -> None:
        """Refresh the multipliers if enabled and ``step`` is an update step."""
        if not self.enabled or (step % self.update_every) != 0:
            return
        eps = 1e-12
        g_data = self._grad_norm(l_data, net_params)
        g_phys = self._grad_norm(l_phys, net_params)
        g_bound = self._grad_norm(l_bound, net_params)
        # Clamp to [floor, 1]: a term whose gradients would swamp the data anchor
        # may be down-weighted, but a near-satisfied constraint is never
        # up-weighted past parity with the data.
        lam_p_hat = (g_data / (g_phys + eps)).clamp(*self.clamp)
        lam_b_hat = (g_data / (g_bound + eps)).clamp(*self.clamp)
        self.lambda_p.mul_(self.alpha).add_((1.0 - self.alpha) * lam_p_hat)
        self.lambda_b.mul_(self.alpha).add_((1.0 - self.alpha) * lam_b_hat)

    def combine(self, l_data, l_phys, l_bound):
        """Weighted total loss."""
        if not self.enabled:
            return l_data + l_phys + l_bound
        return l_data + self.lambda_p.detach() * l_phys + self.lambda_b.detach() * l_bound

    def lambdas(self) -> Dict[str, float]:
        return {
            "lambda_d": 1.0,
            "lambda_p": float(self.lambda_p),
            "lambda_b": float(self.lambda_b),
        }
