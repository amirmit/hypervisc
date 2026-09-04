"""Skin-friction-coefficient models.

A model supplies Cf (scalar, uniform over the wetted area -- or a per-cell
array matching CellArray.A, for a future spatially-varying model) and its
gradient w.r.t. design parameters (or None, if the model has none). Both
`cf()` and `dcf_dp()` accept `cells`/`freestream`/`flow_state` as keyword
context, even though today's ConstantCfModel ignores all of it, so a future
model conditioned on local flow state doesn't need a different call site.
"""

from dataclasses import dataclass


class FrictionModel:
    def cf(self, *, cells=None, freestream=None, flow_state=None):
        raise NotImplementedError

    def dcf_dp(self, *, cells=None, freestream=None, flow_state=None):
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
