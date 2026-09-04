"""Friction-drag force and its design-parameter gradient, from a resolved
Cf model plus a pysagas CellArray/FlowState. Handles both a scalar Cf
(today's ConstantCfModel) and a per-cell Cf array matching cells.A (a
future spatially-varying model) without a different call signature."""

import numpy as np


def friction_force(cells, freestream, cf_model):
    """Returns (Df, dDf_dp): Df is the friction-drag magnitude (opposes
    +x), dDf_dp is its gradient over the same parameter axis as
    cells.dAdp_int."""

    Cf = np.asarray(cf_model.cf(cells=cells, freestream=freestream))

    if Cf.ndim == 0:
        Df = freestream.q * float(Cf) * cells.A_int
        dDf_dp = freestream.q * float(Cf) * cells.dAdp_int
    else:
        Df = freestream.q * np.sum(Cf * cells.A)
        dDf_dp = freestream.q * np.sum(Cf[None, :] * cells.dAdp, axis=1)

    dcf_dp = cf_model.dcf_dp(cells=cells, freestream=freestream)
    if dcf_dp is not None:
        dcf_dp = np.asarray(dcf_dp)
        if Cf.ndim == 0:
            dDf_dp = dDf_dp + freestream.q * cells.A_int * dcf_dp
        else:
            dDf_dp = dDf_dp + freestream.q * np.sum(dcf_dp * cells.A, axis=1)

    return Df, dDf_dp
