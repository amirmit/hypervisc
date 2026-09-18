"""Skin-friction-coefficient models.

A model supplies Cf (a scalar, uniform over the wetted area) and its
gradient w.r.t. design parameters (or None, if Cf has no direct dependence
on them -- separate from whatever sensitivity the area/q it's multiplied by
already carries). Both `cf()` and `dcf_dp()` accept `area`/`q` as keyword
context, even though today's ConstantCfModel ignores them, so a future
model conditioned on local flow conditions doesn't need a different call
site.
"""

from dataclasses import dataclass


class FrictionModel:
    def cf(self, **kwargs):
        raise NotImplementedError

    def dcf_dp(self, **kwargs):
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
