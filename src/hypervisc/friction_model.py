"""Skin-friction-coefficient models.

A model supplies a scalar Cf, uniform over the wetted area. `friction_drag()`
calls `cf(freestream=..., length=...)` with keyword context: `freestream` is
a utilities.FlowState (`.mach`, `.q_dyn`) and `length` is the vehicle
reference length -- needed by MeadorSmartCfModel (omitting it raises),
ignored by ConstantCfModel. Models accept `**kwargs` where they don't need
the context.

There is no separate hand-written d(Cf)/d(param) method. `length` is an
autodiff FloatWithSens dual number whenever a gradient is needed, so a model
that derives Cf from it via ordinary arithmetic -- as MeadorSmartCfModel does
-- returns a correctly-propagated FloatWithSens Cf for free. A model with no
DV dependence (ConstantCfModel) returns a plain float, i.e. no gradient
contribution. This makes hypervisc depend on autodiff, accepted deliberately
in favour of not hand-deriving chain rules per model.
"""

import pickle
from dataclasses import dataclass
import numpy as np

class FrictionModel:
    def cf(self, **kwargs):
        raise NotImplementedError


@dataclass
class ConstantCfModel(FrictionModel):
    """Uniform Cf over the whole wetted area. Reproduces the previous
    hardcoded `Cf = 0.001` exactly when cf_value=0.001."""

    cf_value: float = 0.001

    def cf(self, **kwargs):
        return self.cf_value



@dataclass
class MeadorSmartCfModel(FrictionModel):
    """Turbulent Meador-Smart average Cf, from a Cf_x(Mach, q) surrogate
    fit offline over dynamic pressure q (hypervisc/BuildRefTempModel/
    fit_cf_surrogate.py --mode q). q-only: q is already a freestream
    attribute (force.py's Cf*q_dyn*area term uses it), so this needs no extra
    baked-in flight-condition parameter -- just the pickle.

    Cf(L) = Cf_x(M, q) * L^-n_exp   (n_exp = 0.139, turbulent Meador-Smart;
    see fit_cf_surrogate.py's docstring for the derivation).

    Meant to be loaded once per optimisation run, via get_friction_model()
    (cached by pkl_path -- same load-once-at-setup pattern as HyperPro's
    nozzle surrogate get_surrogate()), NOT reloaded per call. Mach/q
    themselves, though, ARE read fresh from `freestream` (`.mach`,
    `.q_dyn`) on every cf() call rather than cached at construction -- intended for
    trajectory use, where successive calls can be different points along a
    flight path, not a single fixed flight condition for the whole run.
    Each call clamps (M, q) to the fitted training-data range rather than
    raising: a trajectory transiently stepping outside the fitted envelope
    should degrade gracefully (frozen at the nearest bound), not abort the
    optimisation.
    """

    pkl_path: str

    def __post_init__(self):
        with open(self.pkl_path, "rb") as f:
            payload = pickle.load(f)
        if payload["mode"] != "q":
            raise ValueError(
                f"MeadorSmartCfModel only supports q-mode pickles, got "
                f"mode={payload['mode']!r} ({self.pkl_path})"
            )
        self._spline = payload["spline"]
        self._n_exp = payload["n_exp"]
        self._mach_bounds = payload["mach_bounds"]
        self._q_bounds = payload["sec_bounds"]

    def _cf_x(self, freestream):
        """Cf_x(M, q) -- recomputed every call (not cached): M/q can
        differ call to call along a trajectory. Clamped to the fitted
        range rather than validated/raised."""
        mach = np.clip(freestream.mach, *self._mach_bounds)
        q = np.clip(freestream.q_dyn, *self._q_bounds)
        return float(10 ** self._spline(q, mach, grid=False))

    def cf(self, *, freestream, length, **kwargs):
        # length ** (-n_exp) works unchanged whether length is a plain
        # float (no gradient) or a FloatWithSens dual number (gradient via
        # ** overloading) -- Cf_x has no DV-dependence (Mach/q are
        # flight-condition state, not DVs), so the whole gradient comes
        # from this line alone.
        cfx = self._cf_x(freestream)
        return cfx * length ** (-self._n_exp)
