"""Tests for hypervisc's friction models and friction_drag().

New-stack conventions (matching what co_design's Force_Calculator passes in):
  * `freestream` is a real utilities.FlowState (`.mach`, `.q_dyn`).
  * `area` / `length` are autodiff.FloatWithSens (FWS) when a gradient is
    needed; plain floats also work.
  * friction_drag() returns an AeroResults; the drag magnitude Df is
    `-result.force.x`.
"""

import os
import pickle

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from autodiff import FloatWithSens
from utilities import FlowState

from hypervisc import (
    MeadorSmartCfModel,
    friction_drag,
    get_friction_model,
)

N_PARAMS = 3  # FloatWithSens.N is a process-wide global; keep it consistent


def make_freestream(mach=8.0, q_dyn=45000.0):
    """A real FlowState with the requested Mach and dynamic pressure.
    q_dyn = 0.5*gamma*p*M^2 = 0.7*p*M^2 for the default dry-air gamma=1.4."""
    return FlowState(mach=mach, pressure=q_dyn / (0.7 * mach**2), temperature=220.0)


def make_fake_pickle(tmp_path, mode, n_exp=0.139, constant=True):
    """A tiny synthetic Cf_x(mach, secondary) surrogate, so cf() results are
    hand-checkable without depending on the real BuildRefTempModel grid/fit.
    constant=True: Cf_x = 1e-3 everywhere. constant=False: Cf_x varies with
    Mach (decreasing, like the real model), for tests that need to tell two
    different query points apart."""
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


# ---------------------------------------------------------------------------
# ConstantCfModel through friction_drag
# ---------------------------------------------------------------------------

def test_constant_model_matches_old_hardcoded_formula():
    area = FloatWithSens(3.5, [0.1, -0.2, 0.05])
    freestream = make_freestream(q_dyn=45000.0)
    cf_model = get_friction_model("constant", cf_value=0.001)

    Df = -friction_drag(area, freestream, cf_model).force.x

    assert np.isclose(Df.number, 0.001 * freestream.q_dyn * 3.5)
    # Cf and q_dyn have no DV dependence -> only area's sens contributes
    assert np.allclose(Df.sens, 0.001 * freestream.q_dyn * area.sens)


def test_constant_model_ignores_length():
    area = FloatWithSens(3.5, [0.1, -0.2, 0.05])
    freestream = make_freestream()
    cf_model = get_friction_model("constant", cf_value=0.001)

    Df_a = -friction_drag(area, freestream, cf_model).force.x
    Df_b = -friction_drag(area, freestream, cf_model, length=FloatWithSens(7.0, [1, 1, 1])).force.x

    assert np.isclose(Df_a.number, Df_b.number)
    assert np.allclose(Df_a.sens, Df_b.sens)


def test_plain_floats_work_without_sens():
    cf_model = get_friction_model("constant", cf_value=0.0015)
    freestream = make_freestream(q_dyn=45000.0)

    Df = -friction_drag(3.5, freestream, cf_model).force.x

    assert np.isclose(Df, 0.0015 * freestream.q_dyn * 3.5)


def test_friction_drag_force_is_axial_only():
    freestream = make_freestream()
    result = friction_drag(3.5, freestream, get_friction_model("constant", cf_value=0.001))

    assert result.force.x < 0  # opposes +x
    assert result.force.y == 0 and result.force.z == 0
    assert result.flow_state is freestream


# ---------------------------------------------------------------------------
# MeadorSmartCfModel
# ---------------------------------------------------------------------------

def test_meador_smart_q_mode_reads_freestream(tmp_path):
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q"))
    freestream = make_freestream(mach=8.0, q_dyn=45000.0)  # both in bounds

    L = 10.0
    cf = model.cf(freestream=freestream, length=L)
    assert np.isclose(cf, 1e-3 * L ** (-0.139))


def test_meador_smart_rejects_alt_mode_pickle(tmp_path):
    """q-only: an alt-mode pickle (from a different fit_cf_surrogate.py run)
    must fail loudly at construction, not silently misbehave."""
    pkl_path = make_fake_pickle(tmp_path, "alt")

    with pytest.raises(ValueError, match="q-mode"):
        MeadorSmartCfModel(pkl_path=pkl_path)


def test_meador_smart_recomputes_per_call_not_cached(tmp_path):
    """Trajectory use: successive calls can be different flight-path points,
    so Cf must be re-evaluated from `freestream` every call."""
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q", constant=False))
    L = 10.0

    cf_a = model.cf(freestream=make_freestream(mach=8.0, q_dyn=45000.0), length=L)
    cf_b = model.cf(freestream=make_freestream(mach=4.0, q_dyn=45000.0), length=L)

    # a cached Cf_x would make cf_b wrongly equal cf_a
    assert not np.isclose(cf_a, cf_b)


def test_meador_smart_out_of_bounds_clamps_to_nearest_bound(tmp_path):
    """A transient excursion outside the fitted envelope freezes at the
    nearest bound rather than raising/aborting the run."""
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q"))  # M [4,12], q [20e3,120e3]

    far_out = make_freestream(mach=30.0, q_dyn=1e9)
    at_bound = make_freestream(mach=12.0, q_dyn=120e3)

    cf_far = model.cf(freestream=far_out, length=10.0)
    cf_at_bound = model.cf(freestream=at_bound, length=10.0)

    assert np.isclose(cf_far, cf_at_bound)


def test_meador_smart_length_gradient_matches_finite_difference(tmp_path):
    """The gradient comes from a FloatWithSens `length` propagating through
    L ** -n_exp -- no hand-written dcf_dp."""
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q"))
    freestream = make_freestream()

    dL_dp = np.array([0.5, -1.2, 2.0])
    L0 = 8.0
    cf = model.cf(freestream=freestream, length=FloatWithSens(L0, dL_dp))

    assert isinstance(cf, FloatWithSens)

    eps = 1e-6
    cf_plus = model.cf(freestream=freestream, length=L0 + eps)
    cf_minus = model.cf(freestream=freestream, length=L0 - eps)
    fd = (cf_plus - cf_minus) / (2 * eps) * dL_dp

    assert np.allclose(cf.sens, fd, rtol=1e-4)


def test_meador_smart_registered_and_cached(tmp_path):
    pkl_path = make_fake_pickle(tmp_path, "q")
    m1 = get_friction_model("meador_smart", pkl_path=pkl_path)
    m2 = get_friction_model("meador_smart", pkl_path=pkl_path)
    assert m1 is m2
    assert isinstance(m1, MeadorSmartCfModel)


# ---------------------------------------------------------------------------
# MeadorSmartCfModel through friction_drag (the co_design call path)
# ---------------------------------------------------------------------------

def test_friction_drag_threads_length_and_freestream_through(tmp_path):
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q"))
    freestream = make_freestream(mach=8.0, q_dyn=45000.0)
    L = 6.0

    Df = -friction_drag(3.5, freestream, model, length=L).force.x

    expected_cf = 1e-3 * L ** (-0.139)
    assert np.isclose(Df, freestream.q_dyn * expected_cf * 3.5)


def test_friction_drag_gradient_through_area_and_length_matches_finite_difference(tmp_path):
    """Df = Cf(L) * q * area: both area and L carry sens. Checks the full
    friction_drag gradient (product rule + Cf's L-dependence) end to end."""
    model = MeadorSmartCfModel(pkl_path=make_fake_pickle(tmp_path, "q"))
    freestream = make_freestream()

    area_sens = np.array([0.1, -0.2, 0.05])
    L_sens = np.array([0.5, -1.2, 2.0])
    area0, L0 = 3.5, 8.0

    Df = -friction_drag(
        FloatWithSens(area0, area_sens), freestream, model, length=FloatWithSens(L0, L_sens)
    ).force.x

    def Df_at(step):
        # perturb all three parameters' worth of (area, L) along direction i
        return -friction_drag(
            area0 + step * area_sens[i], freestream, model, length=L0 + step * L_sens[i]
        ).force.x

    eps = 1e-6
    for i in range(N_PARAMS):
        fd = (Df_at(eps) - Df_at(-eps)) / (2 * eps)
        assert np.isclose(Df.sens[i], fd, rtol=1e-4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_caches_by_name_and_params():
    m1 = get_friction_model("constant", cf_value=0.002)
    m2 = get_friction_model("constant", cf_value=0.002)
    m3 = get_friction_model("constant", cf_value=0.003)
    assert m1 is m2
    assert m1 is not m3


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown friction model"):
        get_friction_model("does_not_exist")
