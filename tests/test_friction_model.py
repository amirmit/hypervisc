import numpy as np

from hypervisc import get_friction_model, friction_force
from hypervisc.registry import _CACHE


class FakeCells:
    def __init__(self, n_params=3, n_cells=5, seed=0):
        rng = np.random.default_rng(seed)
        self.A = rng.uniform(0.1, 1.0, n_cells)
        self.A_int = np.sum(self.A)
        self.dAdp = rng.uniform(-1.0, 1.0, (n_params, n_cells))
        self.dAdp_int = np.sum(self.dAdp, axis=1)


class FakeFreestream:
    q = 45000.0


def test_constant_model_matches_old_hardcoded_formula():
    cells = FakeCells()
    freestream = FakeFreestream()

    cf_model = get_friction_model("constant", cf_value=0.001)
    Df, dDf_dp = friction_force(cells, freestream, cf_model)

    Df_old = freestream.q * 0.001 * cells.A_int
    dDf_dp_old = freestream.q * 0.001 * cells.dAdp_int

    assert np.isclose(Df, Df_old)
    assert np.allclose(dDf_dp, dDf_dp_old)


def test_registry_caches_by_name_and_params():
    m1 = get_friction_model("constant", cf_value=0.002)
    m2 = get_friction_model("constant", cf_value=0.002)
    m3 = get_friction_model("constant", cf_value=0.003)
    assert m1 is m2
    assert m1 is not m3


def test_unknown_model_raises():
    try:
        get_friction_model("does_not_exist")
    except ValueError as e:
        assert "Unknown friction model" in str(e)
    else:
        raise AssertionError("expected ValueError")
