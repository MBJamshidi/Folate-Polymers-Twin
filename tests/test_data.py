import numpy as np
import torch

from folate_twin import C_MAX, RAW_TARGET, T_MAX, Normaliser, target_array


def test_target_matrix_shape_and_values(target):
    _, tm = target
    assert tm.n == 8
    assert tm.t_hours.dtype == torch.float64
    assert float(tm.t_hours[0]) == 0.083 and float(tm.t_hours[-1]) == 24.0
    assert float(tm.cp_raw.max()) == 400.64
    assert float(tm.ct_raw.max()) == 361.52


def test_normalisation_is_live(target):
    norm, tm = target
    assert torch.equal(tm.t_bar, tm.t_hours / T_MAX)
    assert torch.equal(tm.cp_bar, tm.cp_raw / C_MAX)
    assert float(tm.t_bar[-1]) == 1.0
    assert torch.allclose(norm.denorm_conc(tm.ct_bar), tm.ct_raw)


def test_numpy_view_matches_tensor():
    arr = target_array()
    assert arr.shape == (8, 3)
    np.testing.assert_array_equal(arr, np.array(RAW_TARGET))
    assert Normaliser().t_max == 24.0 and Normaliser().c_max == 422.5
