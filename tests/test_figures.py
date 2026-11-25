import os

import pytest

from folate_twin.viz import compose_all, render_figures, render_robustness_figures


@pytest.mark.slow
def test_render_every_figure_from_reference_results(results_dir, tmp_path):
    out = str(tmp_path / "figures")
    assert render_figures(results_dir, out, verbose=False) == 46
    assert render_robustness_figures(results_dir, out, verbose=False) == []
    pngs = [f for f in os.listdir(out) if f.endswith(".png")]
    pdfs = [f for f in os.listdir(out) if f.endswith(".pdf")]
    assert len(pngs) == 54 and len(pdfs) == 54
    panels = compose_all(out, str(tmp_path / "panels"), verbose=False)
    assert len(panels) == 8 and all(os.path.getsize(p) > 0 for p in panels)


def test_house_style_restores_matplotlib_state():
    import matplotlib as mpl
    from folate_twin.viz.style import house_style
    before = dict(mpl.rcParams)
    with house_style():
        assert mpl.rcParams["savefig.dpi"] == 300
    assert dict(mpl.rcParams) == before
