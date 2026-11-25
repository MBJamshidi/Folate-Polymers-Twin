"""Global numerical settings, normalisation anchors and the ablation matrix.

Importing :mod:`folate_twin` sets ``torch.set_default_dtype(torch.float64)``.
Every network layer is created in double precision through this default, which
is what the reference results were computed with; it is therefore applied once,
here, before any model is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

# --------------------------------------------------------------------------- #
# Numerical precision and device
# --------------------------------------------------------------------------- #
torch.set_default_dtype(torch.float64)          # float64 for stiff-system stability
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Normalisation anchors
# --------------------------------------------------------------------------- #
T_MAX = 24.0          # hours
C_MAX = 422.5         # ug/mL

# --------------------------------------------------------------------------- #
# Reference training settings (the settings behind the reported results)
# --------------------------------------------------------------------------- #
EPOCHS = 6000
N_COLLOCATION = 300
WIDTH = 64
DEPTH = 3
LEARNING_RATE = 2e-3
RK4_STEPS = 240
SEED = 0
HISTORY_EVERY = 10    # epoch stride of the logged training record
DENSE_POINTS = 200    # points of the dense prediction grid


# --------------------------------------------------------------------------- #
# Ablation matrix
# --------------------------------------------------------------------------- #
@dataclass
class AblationConfig:
    """One arm of the A-D ablation study."""

    name: str
    paradigm: str          # "passive" | "phenomenological"
    topology: str          # "mlp" | "dgm"
    attention: bool        # Residual-Based Attention on/off
    adaptive: bool         # self-adaptive loss weighting on/off


ABLATION: List[AblationConfig] = [
    AblationConfig("Configuration A", "passive", "mlp", False, False),
    AblationConfig("Configuration B", "passive", "dgm", False, False),
    AblationConfig("Configuration C", "phenomenological", "dgm", False, False),
    AblationConfig("Configuration D", "phenomenological", "dgm", True, True),
]

CONFIG_ORDER = [cfg.name for cfg in ABLATION]
