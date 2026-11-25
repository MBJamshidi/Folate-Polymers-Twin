import pytest
import torch

from folate_twin import DifferentiableRK4, KineticParameters


def test_twin_requires_phenomenological_parameters():
    with pytest.raises(ValueError):
        DifferentiableRK4(KineticParameters("passive"))


def test_rollout_shape_and_initial_state():
    twin = DifferentiableRK4(KineticParameters("phenomenological"), steps=240)
    with torch.no_grad():
        t, traj = twin.rollout(torch.tensor(0.5))
    assert t.shape == (241,) and traj.shape == (241, 2)
    assert float(t[-1]) == pytest.approx(24.0)
    assert float(traj[0, 0]) == pytest.approx(267.42 / 422.5)
    assert float(traj[0, 1]) == pytest.approx(239.50 / 422.5)


def test_rk4_converges():
    params = KineticParameters("phenomenological")
    c0 = torch.tensor(0.5)
    with torch.no_grad():
        _, coarse = DifferentiableRK4(params, steps=240).rollout(c0)
        _, fine = DifferentiableRK4(params, steps=2400).rollout(c0)
    assert torch.allclose(coarse[-1], fine[-1], rtol=1e-8, atol=1e-10)


def test_autograd_sensitivity_matches_finite_difference():
    twin = DifferentiableRK4(KineticParameters("phenomenological"))
    c0, h = 0.5, 1e-6
    _, grad = twin.state_sensitivity(c0)
    up, _ = twin.state_sensitivity(c0 + h)
    dn, _ = twin.state_sensitivity(c0 - h)
    assert grad == pytest.approx((up - dn) / (2 * h), rel=1e-6)
