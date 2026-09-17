# hypervisc

Skin-friction (Cf) drag models for the HyperMDAO external-body friction term.

Mirrors the nozzle-surrogate selection pattern used in `HyperPro`/`Surrogate_Opt_v3`:
a model is selected by name (`cf_model` in a case's `opt_input.py`), resolved
through a small cached registry, so switching to a fitted/spatially-varying
model later is a config change, not a call-site change.

## Usage

```python
from hypervisc import get_friction_model, friction_force

cf_model = get_friction_model("constant", cf_value=0.001)
Df, dDf_dp = friction_force(cells, freestream, cf_model)
```

`cells` is a `pysagas.CellArray` (needs `.A`, `.A_int`, `.dAdp`, `.dAdp_int`).
`freestream` is a `pysagas.flow.FlowState` (needs `.q`).

`friction_force` returns the drag magnitude `Df` (opposing +x) and its
gradient `dDf_dp` over the same parameter axis as `cells.dAdp`/`dAdp_int`.

## Models

- `"constant"` -> `ConstantCfModel(cf_value=...)`. Uniform Cf over the whole
  wetted area — today's only model, reproduces the previous hardcoded
  `Cf = 0.001` exactly when `cf_value=0.001`.

## Adding a model

Add a class to `friction_model.py` implementing `.cf(**kwargs)` (returns a
scalar or a per-cell array matching `cells.A`) and `.dcf_dp(**kwargs)`
(returns `None` if the model has no design-parameter dependence, else an
array matching `cells.dAdp`'s parameter axis), then register it in
`registry.py`'s `_MODEL_CLASSES`. `friction_force` already handles both the
scalar and per-cell-array cases, so no call-site changes are needed.
