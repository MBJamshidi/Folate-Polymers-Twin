"""folate_twin: physics-informed neural network digital twin of plasma-tumour
uptake and retention of folate-targeted polymeric nanoparticles in breast
cancer.

Importing the package sets ``torch.set_default_dtype(torch.float64)``; every
model in the package is built and trained in double precision.

Quick start::

    from folate_twin import load_target, ABLATION, train_configuration

    normaliser, target = load_target()
    result = train_configuration(ABLATION[3], target, normaliser)   # Configuration D
    print(result.metrics, result.final_rates)
"""

from .config import (
    ABLATION, C_MAX, CONFIG_ORDER, DEVICE, DTYPE, EPOCHS, N_COLLOCATION, RK4_STEPS, T_MAX,
    AblationConfig,
)
from .data import RAW_TARGET, Normaliser, TargetMatrix, load_target, target_array
from .metrics import compute_metrics
from .models import DGMLayer, DGMSurrogate, KineticParameters, MLPSurrogate, build_surrogate
from .physics import physics_residuals, rba_weights
from .pipeline import run_pipeline
from .training import TrainResult, run_ablation, train_configuration
from .twin import DifferentiableRK4
from .weighting import AdaptiveLossWeights

__version__ = "1.0.0"

__all__ = [
    "ABLATION", "C_MAX", "CONFIG_ORDER", "DEVICE", "DTYPE", "EPOCHS", "N_COLLOCATION",
    "RK4_STEPS", "T_MAX", "AblationConfig",
    "RAW_TARGET", "Normaliser", "TargetMatrix", "load_target", "target_array",
    "compute_metrics",
    "DGMLayer", "DGMSurrogate", "KineticParameters", "MLPSurrogate", "build_surrogate",
    "physics_residuals", "rba_weights",
    "run_pipeline",
    "TrainResult", "run_ablation", "train_configuration",
    "DifferentiableRK4",
    "AdaptiveLossWeights",
    "__version__",
]
