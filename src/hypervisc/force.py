"""Friction-drag force and its design-parameter gradient, from a resolved
Cf model plus a pysagas CellArray/FlowState. Handles both a scalar Cf
(today's ConstantCfModel) and a per-cell Cf array matching cells.A (a
future spatially-varying model) without a different call signature.

Split into a value-only function (friction_drag, for Run_ObjFunc.py) and a
gradient-only function (friction_drag_sens, for Run_SensFunc.py) rather
than one function returning both -- so the objective-value path never pays
for the dual-number bookkeeping below when only Df is needed. Both share
_cf_weighted for the Cf * area reduction so that scalar-vs-per-cell branch
exists exactly once.

friction_drag_sens gets a model's Cf-vs-design-parameter dependence via
hyperVehicle's FloatWithSens dual numbers rather than a hand-written
dcf_dp(): callers pass `length` as a FloatWithSens (autodiff is already
active throughout calc_vehicle()), so cf_model.cf() returns a FloatWithSens
Cf for any model whose Cf actually depends on length (e.g.
MeadorSmartCfModel), or a plain float for one that doesn't (e.g.
ConstantCfModel) -- _get_number()/get_sens() extract the right thing from
either case uniformly, with no per-model derivative code required.
"""

import numpy as np
from hypervehicle.geometry import FloatWithSens
from hypervehicle.geometry.autodiff import get_sens


def _get_number(x):
    """Minimal get_number: hypervisc depends on hypervehicle (for the dual
    number type itself) but deliberately not on HyperPro (whose own
    get_number lives in a much heavier, scramjet-specific package)."""
    return x.number if isinstance(x, FloatWithSens) else float(x)


def _cf_weighted(Cf, A_reduced, A_full):
    """Cf (scalar, or a per-cell array matching A_full's last axis) times
    an area quantity, integrated over cells. A_reduced is the
    already-cell-summed quantity (cells.A_int / cells.dAdp_int) -- used
    directly when Cf is uniform. A_full is its per-cell counterpart
    (cells.A / cells.dAdp) -- used to weight each cell separately when Cf
    is not uniform."""
    if Cf.ndim == 0:
        return float(Cf) * A_reduced
    if A_full.ndim == 1:
        return np.sum(Cf * A_full)
    return np.sum(Cf[None, :] * A_full, axis=1)


def friction_drag(cells, freestream, cf_model, length=None):
    """Df: the friction-drag magnitude (opposes +x). `length` must be a
    plain float here (get_number() it first if it's a dual number) -- this
    is the cheap value-only path and never needs a gradient. `length` is
    only meaningful to a model that needs it (e.g. MeadorSmartCfModel);
    ConstantCfModel ignores it."""
    Cf = np.asarray(cf_model.cf(cells=cells, freestream=freestream, length=length))
    return freestream.q * _cf_weighted(Cf, cells.A_int, cells.A)


def friction_drag_sens(cells, freestream, cf_model, length=None):
    """dDf_dp: gradient of the friction drag, over the same parameter axis
    as cells.dAdp_int. `length` should be passed as the FloatWithSens dual
    number here (not get_number()'d) -- see module docstring."""
    Cf = cf_model.cf(cells=cells, freestream=freestream, length=length)
    Cf_num = np.asarray(_get_number(Cf))
    dCf_dp = np.asarray(get_sens(Cf))

    dDf_dp = freestream.q * _cf_weighted(Cf_num, cells.dAdp_int, cells.dAdp)

    if Cf_num.ndim == 0:
        dDf_dp = dDf_dp + freestream.q * cells.A_int * dCf_dp
    else:
        dDf_dp = dDf_dp + freestream.q * np.sum(dCf_dp * cells.A, axis=1)

    return dDf_dp
