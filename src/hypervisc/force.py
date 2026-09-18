"""Friction-drag force from a resolved Cf model, a wetted area, and a
dynamic pressure. `area` and `q` are typically `FloatWithSens` (autodiff's
dual-number scalar -- matching hypervehicle2's `mesh.surface_area` and the
new-stack `FlowState.q_dyn`), so Df's sensitivity to design parameters comes
for free through their own arithmetic; plain floats work too when
sensitivities aren't needed. hypervisc has no dependency on autodiff itself
-- Df's sens is updated by mutating the `.sens` attribute of whatever
`cf * q * area` already returns (duck-typed), not by constructing a new
FloatWithSens.
"""

import numpy as np


def _value(x):
    return x.number if hasattr(x, "number") else x


def friction_drag(area, q, cf_model):
    """Df = Cf * q * area (opposes +x). `cf_model.dcf_dp()` -- Cf's own
    direct dependence on a design parameter, on top of whatever sens area/q
    already carry -- is added onto Df's sens in place."""
    cf = cf_model.cf(area=area, q=q)
    Df = cf * q * area

    dcf_dp = cf_model.dcf_dp(area=area, q=q)
    if dcf_dp is not None:
        Df.sens = Df.sens + np.asarray(dcf_dp) * _value(q) * _value(area)

    return Df
