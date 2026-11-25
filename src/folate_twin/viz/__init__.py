"""Figure generation: figures 1-46, robustness figures 47-54 and the eight
multi-panel article figures."""

from .figures import render_figures
from .panels import compose_all
from .robustness_figures import render_robustness_figures

__all__ = ["render_figures", "render_robustness_figures", "compose_all"]
