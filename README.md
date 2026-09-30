# hypervisc

Skin-friction (Cf) drag models for the MDAO external-body friction term.

Mirrors the nozzle-surrogate selection pattern used in `hyperProp`:
a model is selected by name (`cf_model` in a case's `opt_input.py`), resolved
through a small cached registry, so switching to a fitted/spatially-varying
model later is a config change, not a call-site change.

## Usage

```python
from hypervisc import get_friction_model, friction_drag

cf_model = get_friction_model("constant", cf_value=0.001)
Df = friction_drag(area, q, cf_model)
```

`area` is the wetted reference area and `q` the freestream dynamic pressure.
Both are typically `FloatWithSens` (autodiff's dual-number scalar --
`area` matches hypervehicle2's `mesh.surface_area`, `q` matches the new-stack
`FlowState.q_dyn`), so `Df`'s sensitivity to design parameters comes for free
through their own arithmetic; plain floats work too when sensitivities
aren't needed. hypervisc has no dependency on autodiff itself -- it only
needs `area`/`q` to support multiplication (and, for a design-parameter-
dependent Cf model, a `.sens` attribute to fold `dcf_dp` into).

`friction_drag` returns the drag magnitude `Df` (opposing +x), carrying its
own gradient over whatever design-parameter axis `area`/`q` already carry.

## Models

- `"constant"` -> `ConstantCfModel(cf_value=...)`. Uniform Cf over the whole
  wetted area — today's only model, reproduces the previous hardcoded
  `Cf = 0.001` exactly when `cf_value=0.001`.

## Adding a model

Add a class to `friction_model.py` implementing `.cf(**kwargs)` (returns a
scalar) and `.dcf_dp(**kwargs)` (returns `None` if the model has no direct
design-parameter dependence, else an array matching `area`/`q`'s sens
axis), then register it in `registry.py`'s `_MODEL_CLASSES`. `friction_drag`
already handles the `dcf_dp is None` case, so no call-site changes are
needed.
