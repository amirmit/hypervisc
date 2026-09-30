"""Friction-drag force from a resolved Cf model, a wetted area, and a
freestream FlowState.

`area` (and `length`, for models that use it) are typically autodiff
`FloatWithSens` dual numbers -- the wetted area summed from the mesh, and the
vehicle length from the geometry -- so Df's sensitivity to design parameters
propagates through ordinary arithmetic: `Df = cf * q_dyn * area` picks up
area's sens by the product rule, and cf's own sens (if the model's Cf depends
on a dual-number input such as `length`) rides along in `cf`. Plain floats
work too when sensitivities aren't needed. `freestream.q_dyn` is a plain
float (flight condition, not a design variable).

There is no per-model derivative hook: a model that wants a Cf gradient just
computes Cf from the dual-number inputs it is handed (see
MeadorSmartCfModel), and a model with no DV dependence (ConstantCfModel)
returns a plain float, which contributes nothing to the gradient.
hypervisc therefore depends on autodiff for the dual-number type.
"""

from utilities import AeroResults, Vector3
from autodiff import FloatWithSens, get_sens

def friction_drag(area, freestream, cf_model, length=None):
    """Df = Cf * q_dyn * area, returned as an AeroResults whose force is
    (-Df, 0, 0) -- friction opposes +x -- and moment is zero.

    `freestream` supplies q_dyn and is also passed to the model (Mach/q for
    MeadorSmartCfModel). `length` is the vehicle reference length; it is
    required by MeadorSmartCfModel and ignored by ConstantCfModel. Cf's own
    sensitivity (via a dual-number `length`) is taken from whatever the
    model's cf() returns."""

    q = freestream.q_dyn
    cf = cf_model.cf(freestream=freestream, length=length)

    Df = cf * q * area

    net_force = Vector3(x=-Df, y=0, z=0)
    net_moment = Vector3(x=0, y=0, z=0)

    friction_force = AeroResults(
        flow_state=freestream, force=net_force, moment=net_moment
    )

    return friction_force
