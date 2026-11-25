import pytest
import torch

from folate_twin import KineticParameters, build_surrogate, physics_residuals, rba_weights


@pytest.mark.parametrize("paradigm", ["phenomenological", "passive"])
def test_residual_shapes(paradigm):
    torch.manual_seed(0)
    net = build_surrogate("dgm", 16, 2)
    params = KineticParameters(paradigm)
    t = torch.linspace(0, 1, 40).reshape(-1, 1)
    r_p, r_t, cp, ct = physics_residuals(net, params, t, paradigm)
    for x in (r_p, r_t, cp, ct):
        assert x.shape == (40, 1)
    assert r_p.requires_grad


def test_residual_matches_manual_derivative():
    """r_t must equal (1/T_max) dC_t/dt_bar - k_in C_p + k_out C_t."""
    torch.manual_seed(1)
    net = build_surrogate("mlp", 8, 2)
    params = KineticParameters("phenomenological")
    t = torch.tensor([[0.3]])
    _, r_t, cp, ct = physics_residuals(net, params, t, "phenomenological", t_max=24.0)
    h = 1e-6
    with torch.no_grad():
        dct = (net(t + h)[0, 1] - net(t - h)[0, 1]) / (2 * h)
        k = params.rates()
        manual = dct / 24.0 - k["k_in"] * cp[0, 0] + k["k_out"] * ct[0, 0]
    assert float(r_t[0, 0].detach()) == pytest.approx(float(manual), rel=1e-6, abs=1e-10)


def test_unknown_paradigm_raises():
    net = build_surrogate("mlp", 8, 1)
    with pytest.raises(ValueError):
        physics_residuals(net, KineticParameters("passive"), torch.zeros(3, 1), "other")


def test_rba_weights_are_a_distribution():
    r_p = torch.tensor([[1.0], [-2.0], [0.0]], requires_grad=True)
    r_t = torch.tensor([[0.5], [0.5], [1.0]])
    w = rba_weights(r_p, r_t)
    assert not w.requires_grad
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(w[1]) > float(w[0]) > float(w[2])
