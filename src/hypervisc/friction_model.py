"""Skin-friction-coefficient models.

A model supplies Cf (scalar, uniform over the wetted area -- or a per-cell
array matching CellArray.A, for a future spatially-varying model) and its
gradient w.r.t. design parameters (or None, if the model has none). Both
`cf()` and `dcf_dp()` accept `cells`/`freestream`/`flow_state` as keyword
context, plus `length`/`dlength_dp` (vehicle reference length and its
design-parameter sensitivity -- needed by MeadorSmartCfModel, ignored by
ConstantCfModel), so a future model conditioned on local flow state doesn't
need a different call site.
"""

import pickle
from dataclasses import dataclass

import numpy as np


class FrictionModel:
    def cf(self, *, cells=None, freestream=None, flow_state=None, length=None):
        raise NotImplementedError

    def dcf_dp(self, *, cells=None, freestream=None, flow_state=None,
               length=None, dlength_dp=None):
        """d(Cf)/d(param). Return None if Cf has no design-parameter
        dependence (the common case) -- callers treat None as an all-zero
        contribution rather than requiring every model to build a zero
        array of the right shape."""
        return None


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
        cfx = self._cf_x(freestream)
        return cfx * length ** (-self._n_exp)

    def dcf_dp(self, *, freestream, length, dlength_dp, **kwargs):
        # Cf_x has no DV-dependence (Mach/q are flight-condition state,
        # not DVs) -- the whole gradient comes from the L^-n_exp chain rule.
        cfx = self._cf_x(freestream)
        dcf_dL = -self._n_exp * cfx * length ** (-self._n_exp - 1)
        return dcf_dL * dlength_dp
