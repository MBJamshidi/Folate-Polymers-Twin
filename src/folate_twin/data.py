"""Bioanalytical target matrix and stage-1 normalisation.

The dataset is the digitised plasma / tumour concentration-time profile of
folate-targeted PSMAC-PTX polymeric nanoparticles in a murine breast-cancer
model (8 sampling times, 2 compartments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import C_MAX, DEVICE, DTYPE, T_MAX

# Columns: t_i (h), C_p (ug/mL), C_t (ug/mL).  The normalised time t/T_MAX is
# recomputed in code, never hard-coded.
RAW_TARGET = (
    (0.083, 267.42, 239.50),
    (0.330, 357.45, 239.50),
    (1.000, 400.64, 239.50),
    (2.000, 338.15, 341.98),
    (4.000, 269.47, 341.98),
    (8.000, 167.23, 361.52),
    (12.000, 120.33, 320.74),
    (24.000, 82.92, 74.54),
)


def target_array() -> np.ndarray:
    """The target matrix as a float64 NumPy array of shape (8, 3)."""
    return np.array(RAW_TARGET, dtype=np.float64)


@dataclass
class TargetMatrix:
    """Ingested empirical PK target matrix and its normalised tensors."""

    t_hours: torch.Tensor          # (N,1) raw time in hours
    cp_raw: torch.Tensor           # (N,1) plasma  ug/mL
    ct_raw: torch.Tensor           # (N,1) tumour  ug/mL
    t_bar: torch.Tensor            # (N,1) normalised time  t/T_MAX
    cp_bar: torch.Tensor           # (N,1) normalised plasma C/C_MAX
    ct_bar: torch.Tensor           # (N,1) normalised tumour C/C_MAX

    @property
    def n(self) -> int:
        return self.t_hours.shape[0]


class Normaliser:
    """Stage-1 normalisation:  t_bar = t / T_max ,  C_bar = C / C_max.

    The forward pass and every loss term consume the *normalised* tensors, so the
    anchors T_max and C_max are genuine functional dependencies.
    """

    def __init__(self, t_max: float = T_MAX, c_max: float = C_MAX):
        self.t_max = float(t_max)
        self.c_max = float(c_max)

    def norm_time(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.t_max

    def norm_conc(self, c: torch.Tensor) -> torch.Tensor:
        return c / self.c_max

    def denorm_conc(self, c_bar: torch.Tensor) -> torch.Tensor:
        return c_bar * self.c_max

    def ingest(self) -> TargetMatrix:
        """Load :data:`RAW_TARGET` as normalised tensors on :data:`DEVICE`."""
        data = torch.tensor(RAW_TARGET, dtype=DTYPE, device=DEVICE)
        t_hours = data[:, 0:1]
        cp_raw = data[:, 1:2]
        ct_raw = data[:, 2:3]
        return TargetMatrix(
            t_hours=t_hours,
            cp_raw=cp_raw,
            ct_raw=ct_raw,
            t_bar=self.norm_time(t_hours),
            cp_bar=self.norm_conc(cp_raw),
            ct_bar=self.norm_conc(ct_raw),
        )


def load_target() -> tuple[Normaliser, TargetMatrix]:
    """Convenience: the default normaliser and the ingested target matrix."""
    norm = Normaliser(T_MAX, C_MAX)
    return norm, norm.ingest()
