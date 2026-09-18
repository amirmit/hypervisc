import numpy as np

from hypervisc import get_friction_model, friction_drag


class FakeScalarWithSens:
    """Minimal duck-type of autodiff.FloatWithSens -- just enough
    multiplication (with the product rule) for friction_drag to exercise
    the same sens-propagation path FWS provides, without hypervisc
    depending on autodiff."""

    def __init__(self, number, sens):
        self.number = number
        self.sens = np.asarray(sens, dtype=float)

    def __mul__(self, other):
        if isinstance(other, FakeScalarWithSens):
            return FakeScalarWithSens(
                self.number * other.number,
                self.number * other.sens + self.sens * other.number,
            )
        return FakeScalarWithSens(self.number * other, self.sens * other)

    __rmul__ = __mul__


def test_constant_model_matches_old_hardcoded_formula():
    area = FakeScalarWithSens(3.5, [0.1, -0.2, 0.05])
    q = FakeScalarWithSens(45000.0, [0.0, 1.0, 0.0])
    cf_model = get_friction_model("constant", cf_value=0.001)

    Df = friction_drag(area, q, cf_model)

    assert np.isclose(Df.number, 0.001 * 45000.0 * 3.5)
    expected_sens = 0.001 * (q.number * area.sens + area.number * q.sens)
    assert np.allclose(Df.sens, expected_sens)


def test_plain_floats_work_without_sens():
    cf_model = get_friction_model("constant", cf_value=0.0015)
    Df = friction_drag(3.5, 45000.0, cf_model)
    assert np.isclose(Df, 0.0015 * 45000.0 * 3.5)


def test_registry_caches_by_name_and_params():
    m1 = get_friction_model("constant", cf_value=0.002)
    m2 = get_friction_model("constant", cf_value=0.002)
    m3 = get_friction_model("constant", cf_value=0.003)
    assert m1 is m2
    assert m1 is not m3


def test_unknown_model_raises():
    try:
        get_friction_model("does_not_exist")
    except ValueError as e:
        assert "Unknown friction model" in str(e)
    else:
        raise AssertionError("expected ValueError")
