import pytest
import torch

from folate_twin import DGMSurrogate, KineticParameters, MLPSurrogate, build_surrogate


@pytest.mark.parametrize("cls", [DGMSurrogate, MLPSurrogate])
def test_surrogate_output_is_nonnegative_float64(cls):
    torch.manual_seed(0)
    net = cls(width=16, depth=2)
    out = net(torch.linspace(0, 1, 50).reshape(-1, 1))
    assert out.shape == (50, 2)
    assert out.dtype == torch.float64
    assert bool((out >= 0).all())


def test_build_surrogate_rejects_unknown_topology():
    assert isinstance(build_surrogate("dgm"), DGMSurrogate)
    with pytest.raises(ValueError):
        build_surrogate("transformer")


def test_kinetic_parameter_initialisation():
    ph = KineticParameters("phenomenological")
    vals = ph.detached()
    assert set(vals) == {"k10", "k_in", "k_out", "k_a", "C0"}
    for k in ("k10", "k_in", "k_out", "k_a"):
        assert vals[k] == pytest.approx(0.3, rel=1e-12)
    assert vals["C0"] == pytest.approx(0.5, rel=1e-12)

    pa = KineticParameters("passive")
    assert set(pa.detached()) == {"k10", "k12", "k21"}
    assert pa.c0() is None
    with pytest.raises(ValueError):
        KineticParameters("other")
