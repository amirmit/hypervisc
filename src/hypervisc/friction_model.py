"""Skin-friction-coefficient models.

A model supplies Cf (scalar, uniform over the wetted area -- or a per-cell
array matching CellArray.A, for a future spatially-varying model). `cf()`
accepts `cells`/`freestream`/`flow_state` as keyword context, plus `length`
(vehicle reference length -- needed by MeadorSmartCfModel, ignored by
ConstantCfModel), so a future model conditioned on local flow state doesn't
need a different call site.

No separate hand-written d(Cf)/d(param) method: `length` (and any future
DV-dependent context) is a hyperVehicle FloatWithSens dual number whenever
a gradient is needed (autodiff is already active throughout calc_vehicle()),
so a model that derives Cf from it via ordinary arithmetic -- as
MeadorSmartCfModel does -- gets a correctly-propagated FloatWithSens Cf back
for free. force.py's friction_drag_sens() extracts value/gradient from
whatever cf() returns with get_number()/get_sens(), which already handle a
plain-float Cf (e.g. ConstantCfModel, no DV dependence) as an all-zero
gradient with no per-model special-casing. This trades hypervisc's previous
numpy-only self-containment for one dependency on hypervehicle -- accepted
deliberately in favour of not hand-deriving chain rules per model.
"""

import pickle
from dataclasses import dataclass

import numpy as np


class FrictionModel:
    def cf(self, *, cells=None, freestream=None, flow_state=None, length=None):
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
    attribute (force.py's Cf*Swet*q term uses it), so this needs no extra
    baked-in flight-condition parameter -- just the pickle.

    Cf(L) = Cf_x(M, q) * L^-n_exp   (n_exp = 0.139, turbulent Meador-Smart;
    see fit_cf_surrogate.py's docstring for the derivation).

    Meant to be loaded once per optimisation run, via get_friction_model()
    (cached by pkl_path -- same load-once-at-setup pattern as HyperPro's
    nozzle surrogate get_surrogate()), NOT reloaded per call. Mach/q
    themselves, though, ARE read fresh from `freestream` on every cf()/
    dcf_dp() call rather than cached at construction -- intended for
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
        mach = np.clip(freestream.M, *self._mach_bounds)
        q = np.clip(freestream.q, *self._q_bounds)
        return float(10 ** self._spline(q, mach, grid=False))

    def cf(self, *, freestream, length, **kwargs):
        # length ** (-n_exp) works unchanged whether length is a plain
        # float (friction_drag's value-only path) or a FloatWithSens dual
        # number (friction_drag_sens's gradient path, via ** overloading)
        # -- Cf_x has no DV-dependence (Mach/q are flight-condition state,
        # not DVs), so the whole gradient comes from this line alone.
        cfx = self._cf_x(freestream)
        return cfx * length ** (-self._n_exp)
