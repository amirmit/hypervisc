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

from utilities import AeroResults, Vector3
from autodiff import FloatWithSens

def friction_drag(area, freestream, cf_model):
    """Df = Cf * q * area (opposes +x). `cf_model.dcf_dp()` -- Cf's own
    direct dependence on a design parameter, on top of whatever sens area/q
    already carry -- is added onto Df's sens in place."""

    q = freestream.q_dyn
    cf_val = cf_model.cf(area=area, q=q)
    cf_sens = cf_model.dcf_dp(area=area, q=q) or [0]*FloatWithSens.N
    cf = FloatWithSens(cf_val, cf_sens)

    Df = cf * q * area

    net_force = Vector3(x=-Df, y=0, z=0)
    net_moment = Vector3(x=0, y=0, z=0)

    friction_force = AeroResults(
        flow_state=freestream, force=net_force, moment=net_moment
    )

    return friction_force
