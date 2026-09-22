"""Friction-drag force and its design-parameter gradient, from a resolved
Cf model plus a pysagas CellArray/FlowState. Handles both a scalar Cf
(today's ConstantCfModel) and a per-cell Cf array matching cells.A (a
future spatially-varying model) without a different call signature.

Split into a value-only function (friction_drag, for Run_ObjFunc.py) and a
gradient-only function (friction_drag_sens, for Run_SensFunc.py) rather
than one function returning both -- so the objective-value path never
pays for cf_model.dcf_dp() (a future fitted model's derivative call) when
only Df is needed. Both share _cf_weighted for the Cf * area reduction so
that scalar-vs-per-cell branch exists exactly once.
"""

import numpy as np


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
    """Df: the friction-drag magnitude (opposes +x). Calls only
    cf_model.cf() -- never .dcf_dp() -- since the objective-value path has
    no use for a gradient. `length` is only meaningful to a model that
    needs it (e.g. MeadorSmartCfModel); ConstantCfModel ignores it."""
    Cf = np.asarray(cf_model.cf(cells=cells, freestream=freestream, length=length))
    return freestream.q * _cf_weighted(Cf, cells.A_int, cells.A)


def friction_drag_sens(cells, freestream, cf_model, length=None, dlength_dp=None):
    """dDf_dp: gradient of the friction drag, over the same parameter
    axis as cells.dAdp_int."""
    Cf = np.asarray(cf_model.cf(cells=cells, freestream=freestream, length=length))
    dDf_dp = freestream.q * _cf_weighted(Cf, cells.dAdp_int, cells.dAdp)

    dcf_dp = cf_model.dcf_dp(cells=cells, freestream=freestream,
                              length=length, dlength_dp=dlength_dp)
    if dcf_dp is not None:
        dcf_dp = np.asarray(dcf_dp)
        if Cf.ndim == 0:
            dDf_dp = dDf_dp + freestream.q * cells.A_int * dcf_dp
        else:
            dDf_dp = dDf_dp + freestream.q * np.sum(dcf_dp * cells.A, axis=1)

    return dDf_dp
