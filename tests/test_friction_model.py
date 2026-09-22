import os
import pickle

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from hypervisc import get_friction_model, friction_drag, friction_drag_sens, MeadorSmartCfModel
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
    M = 8.0


def make_fake_pickle(tmp_path, mode, n_exp=0.139, constant=True):
    """A tiny synthetic Cf_x(mach, secondary) surrogate, so cf()/dcf_dp()
    results are hand-checkable without depending on the real
    BuildRefTempModel grid/fit. constant=True: Cf_x = 1e-3 everywhere.
    constant=False: Cf_x varies with Mach (decreasing, like the real
    model), for tests that need to tell two different query points apart."""
    mach_1d = np.array([4.0, 8.0, 12.0])
    sec_1d = np.array([20e3, 60e3, 120e3]) if mode == "q" else np.array([15e3, 30e3, 45e3])
    if constant:
        log_cfx = np.full((3, 3), np.log10(1e-3))
    else:
        log_cfx = np.tile(np.log10(1e-3) - 0.1 * (mach_1d - mach_1d[0]), (3, 1))
    spline = RectBivariateSpline(sec_1d, mach_1d, log_cfx, kx=2, ky=2, s=0)
    payload = dict(mode=mode, spline=spline, n_exp=n_exp,
                    mach_bounds=(mach_1d.min(), mach_1d.max()),
                    sec_bounds=(sec_1d.min(), sec_1d.max()))
    pkl_path = os.path.join(tmp_path, f"fake_{mode}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    return pkl_path


def test_meador_smart_q_mode_reads_freestream_q(tmp_path):
    pkl_path = make_fake_pickle(tmp_path, "q")
    model = MeadorSmartCfModel(pkl_path=pkl_path)
    freestream = FakeFreestream()  # q=45000 (in bounds), M=8 (in bounds)

    L = 10.0
    cf = model.cf(freestream=freestream, length=L)
    assert np.isclose(cf, 1e-3 * L ** (-0.139))


def test_meador_smart_rejects_alt_mode_pickle(tmp_path):
    """q-only now: an alt-mode pickle (from an older/different
    fit_cf_surrogate.py run) must fail loudly at construction, not
    silently misbehave."""
    pkl_path = make_fake_pickle(tmp_path, "alt")

    with pytest.raises(ValueError, match="q-mode"):
        MeadorSmartCfModel(pkl_path=pkl_path)


def test_meador_smart_recomputes_per_call_not_cached(tmp_path):
    """Trajectory use: successive calls can be different flight-path
    points, so Cf must be re-evaluated from `freestream` every call, not
    frozen from the first call like the earlier fixed-per-run design."""
    pkl_path = make_fake_pickle(tmp_path, "q", constant=False)  # Cf_x varies with Mach
    model = MeadorSmartCfModel(pkl_path=pkl_path)
    L = 10.0

    fs_a = FakeFreestream()  # M=8, q=45000

    class OtherFreestream:
        q = 45000.0
        M = 4.0  # different Mach -> different Cf_x on this non-constant fake

    cf_a = model.cf(freestream=fs_a, length=L)
    cf_b = model.cf(freestream=OtherFreestream(), length=L)

    # if _cf_x() were still caching from the first call, cf_b would
    # wrongly equal cf_a instead of reflecting the new Mach.
    assert not np.isclose(cf_a, cf_b)


def test_meador_smart_dcf_dp_matches_finite_difference(tmp_path):
    pkl_path = make_fake_pickle(tmp_path, "q")
    model = MeadorSmartCfModel(pkl_path=pkl_path)
    freestream = FakeFreestream()

    L0 = 8.0
    dL_dp = np.array([0.5, -1.2, 2.0])  # dL/dp for 3 fake design params

    analytic = model.dcf_dp(freestream=freestream, length=L0, dlength_dp=dL_dp)

    eps = 1e-6
    cf_plus = model.cf(freestream=freestream, length=L0 + eps)
    cf_minus = model.cf(freestream=freestream, length=L0 - eps)
    dcf_dL_fd = (cf_plus - cf_minus) / (2 * eps)
    fd = dcf_dL_fd * dL_dp

    assert np.allclose(analytic, fd, rtol=1e-4)


def test_meador_smart_out_of_bounds_clamps_to_nearest_bound(tmp_path):
    """Trajectory use: a transient excursion outside the fitted envelope
    should freeze at the nearest bound, not raise/abort the run."""
    pkl_path = make_fake_pickle(tmp_path, "q")  # fitted mach in [4,12], q in [20e3,120e3]
    model = MeadorSmartCfModel(pkl_path=pkl_path)

    class FarOutFreestream:
        q = 1e9  # way above the fitted upper bound
        M = 30.0  # way above the fitted upper bound

    class AtUpperBoundFreestream:
        q = 120e3
        M = 12.0

    cf_far = model.cf(freestream=FarOutFreestream(), length=10.0)
    cf_at_bound = model.cf(freestream=AtUpperBoundFreestream(), length=10.0)

    # clamped: querying way past the bound gives the same result as
    # querying exactly at the bound, not a crash or an extrapolated value.
    assert np.isclose(cf_far, cf_at_bound)


def test_meador_smart_registered_and_cached(tmp_path):
    pkl_path = make_fake_pickle(tmp_path, "q")
    m1 = get_friction_model("meador_smart", pkl_path=pkl_path)
    m2 = get_friction_model("meador_smart", pkl_path=pkl_path)
    assert m1 is m2
    assert isinstance(m1, MeadorSmartCfModel)


def test_friction_drag_threads_length_through(tmp_path):
    pkl_path = make_fake_pickle(tmp_path, "q")
    model = MeadorSmartCfModel(pkl_path=pkl_path)
    cells = FakeCells()
    freestream = FakeFreestream()
    L = 6.0

    Df = friction_drag(cells, freestream, model, length=L)
    expected_cf = 1e-3 * L ** (-0.139)
    assert np.isclose(Df, freestream.q * expected_cf * cells.A_int)


def test_constant_model_matches_old_hardcoded_formula():
    cells = FakeCells()
    freestream = FakeFreestream()

    cf_model = get_friction_model("constant", cf_value=0.001)
    Df = friction_drag(cells, freestream, cf_model)
    dDf_dp = friction_drag_sens(cells, freestream, cf_model)

    Df_old = freestream.q * 0.001 * cells.A_int
    dDf_dp_old = freestream.q * 0.001 * cells.dAdp_int

    assert np.isclose(Df, Df_old)
    assert np.allclose(dDf_dp, dDf_dp_old)


def test_value_and_sens_agree_with_each_other():
    """friction_drag and friction_drag_sens are now independent calls (no
    longer a single function returning both) -- guard against them ever
    resolving to inconsistent Cf models."""
    cells = FakeCells()
    freestream = FakeFreestream()
    cf_model = get_friction_model("constant", cf_value=0.0015)

    Df = friction_drag(cells, freestream, cf_model)
    dDf_dp = friction_drag_sens(cells, freestream, cf_model)

    assert np.isclose(Df, freestream.q * 0.0015 * cells.A_int)
    assert np.allclose(dDf_dp, freestream.q * 0.0015 * cells.dAdp_int)


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
