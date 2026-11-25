"""Training determinism and regression against the original implementation.

The golden values were produced by the original code that generated the
reported results (40 epochs, 32 collocation points, seed 0).
"""

import pytest
import torch

from folate_twin import ABLATION, train_configuration

GOLDEN = {
    "Configuration A": {"loss": 0.07073496369140633, "plasma_R2": 0.6762130950607363,
                        "tumour_R2": 0.19154027543672514,
                        "rates": {"k10": 0.28521765626561146, "k12": 0.28867360328284986,
                                  "k21": 0.30734050159128606}},
    "Configuration B": {"loss": 0.031397571161928935, "plasma_R2": 0.7423218338778907,
                        "tumour_R2": 0.9323911186028655,
                        "rates": {"k10": 0.28493074758042186, "k12": 0.29564251990274676,
                                  "k21": 0.29516066121642903}},
    "Configuration C": {"loss": 0.05156654683412279, "plasma_R2": 0.6805252285047887,
                        "tumour_R2": 0.9212949718440492,
                        "rates": {"k10": 0.2835371495517818, "k_in": 0.2861747001885899,
                                  "k_out": 0.28092927459580347, "k_a": 0.3185877898543604,
                                  "C0": 0.5321032350413173}},
    "Configuration D": {"loss": 0.06043603348723417, "plasma_R2": 0.6405331516535109,
                        "tumour_R2": 0.910895574758948,
                        "rates": {"k10": 0.2821669159068937, "k_in": 0.28506181343202297,
                                  "k_out": 0.28043889875880557, "k_a": 0.31684334644664647,
                                  "C0": 0.5327923781900088}},
}
RTOL = 1e-6   # bitwise on the reference platform; allows BLAS-level differences elsewhere


@pytest.mark.parametrize("cfg", ABLATION, ids=lambda c: c.name[-1])
def test_matches_original_implementation(cfg, target):
    norm, tm = target
    res = train_configuration(cfg, tm, norm, epochs=40, n_collocation=32)
    g = GOLDEN[cfg.name]
    assert res.loss_history[-1] == pytest.approx(g["loss"], rel=RTOL)
    assert res.metrics["plasma"]["R2"] == pytest.approx(g["plasma_R2"], rel=RTOL)
    assert res.metrics["tumour"]["R2"] == pytest.approx(g["tumour_R2"], rel=RTOL)
    assert res.final_rates.keys() == g["rates"].keys()
    for k, v in g["rates"].items():
        assert res.final_rates[k] == pytest.approx(v, rel=RTOL)


def test_training_is_deterministic(target):
    norm, tm = target
    a = train_configuration(ABLATION[3], tm, norm, epochs=30, n_collocation=32)
    b = train_configuration(ABLATION[3], tm, norm, epochs=30, n_collocation=32)
    assert a.loss_history == b.loss_history
    assert a.final_rates == b.final_rates


def test_history_logging_does_not_change_training(target):
    norm, tm = target
    a = train_configuration(ABLATION[3], tm, norm, epochs=30, n_collocation=32)
    b = train_configuration(ABLATION[3], tm, norm, epochs=30, n_collocation=32, history_every=10)
    assert a.loss_history == b.loss_history
    assert [h["epoch"] for h in b.history] == [1, 11, 21, 30]
    assert b.history[-1]["total"] == b.loss_history[-1]


def test_seed_changes_initialisation(target):
    norm, tm = target
    a = train_configuration(ABLATION[2], tm, norm, epochs=5, n_collocation=16, seed=0)
    b = train_configuration(ABLATION[2], tm, norm, epochs=5, n_collocation=16, seed=42)
    assert a.loss_history != b.loss_history


def test_full_mask_reproduces_unmasked_loss(target):
    norm, tm = target
    full = train_configuration(ABLATION[3], tm, norm, epochs=10, n_collocation=16)
    mask = torch.ones((tm.n, 2), dtype=torch.float64)
    masked = train_configuration(ABLATION[3], tm, norm, epochs=10, n_collocation=16, obs_mask=mask)
    assert masked.loss_history == pytest.approx(full.loss_history, rel=1e-12)
