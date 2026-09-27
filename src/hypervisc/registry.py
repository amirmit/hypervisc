"""Cf-model registry: a name (e.g. "constant") resolves to a cached model
instance, same shape as HyperPro's nozzle-surrogate get_surrogate() --
switching models is a config change (opt_input.py's cf_model/cf_params),
not a call-site change. A future fitted model registers its own class here
(e.g. "v1_kriging" loading a pickle, mirroring SurrogateGen/nozzle_v3/)."""

from .friction_model import ConstantCfModel, MeadorSmartCfModel

_MODEL_CLASSES = {
    "constant": ConstantCfModel,
    "meador_smart": MeadorSmartCfModel,
}

_CACHE = {}


def get_friction_model(name, **params):
    if name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown friction model '{name}'. Available: {sorted(_MODEL_CLASSES)}"
        )
    key = (name, tuple(sorted(params.items())))
    if key not in _CACHE:
        _CACHE[key] = _MODEL_CLASSES[name](**params)
    return _CACHE[key]
