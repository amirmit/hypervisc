"""Pipeline-fidelity check: same 3D Cf_x(Mach, q) plot as fit_cf_surrogate.py's
fig1, but the surface is produced by calling the ACTUAL hypervisc model --
get_friction_model("meador_smart", pkl_path=...) loading the pickle and
MeadorSmartCfModel.cf() evaluating each point -- the identical call path
Run_ObjFunc.py/Run_SensFunc.py use in the optimisation loop. This is not a
re-fit: it's a check that going through the real model registry + pickle
load reproduces the same surface as fit_cf_surrogate.py's direct
RectBivariateSpline query.

length=1.0 is passed to cf() so MeadorSmartCfModel's Cf(L) = Cf_x(M,q) *
L^-n_exp collapses to raw Cf_x -- the same quantity (and units) as the
training/validation dots below, so the two overlay meaningfully.

Training and validation points are generated exactly as in
fit_cf_surrogate.py: training = the grid's own (Mach, q) nodes (cf_grid.csv,
itself produced by make_cf_grid.cf_x_point()); validation =
fit_cf_surrogate.make_validation_points(), which calls cf_x_point() fresh
at off-grid midpoints. Neither goes through the surrogate -- both are raw
reference-temperature-method truth, same as the fit stage.

Run directly: `python3 plot_cf_pipeline_mpl.py`.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from hypervisc import get_friction_model

from make_cf_grid import grid_dir
from fit_cf_surrogate import SECONDARY_LABEL, SECONDARY_SCALE, load_grid, make_validation_points

MODE = "q"  # MeadorSmartCfModel is q-only (see friction_model.py)


class _Freestream:
    """Minimal (M, q) stand-in -- all MeadorSmartCfModel.cf() reads off
    `freestream`, same as the real pysagas FlowStateVec-derived object
    would supply at call sites in Run_ObjFunc.py."""

    def __init__(self, M, q):
        self.M = M
        self.q = q


def main():
    mach_1d, sec_1d, cfx_2d = load_grid(MODE)

    pkl_path = os.path.join(grid_dir(MODE), "cf_x_surrogate.pkl")
    model = get_friction_model("meador_smart", pkl_path=pkl_path)

    val = make_validation_points(MODE, mach_1d, sec_1d)

    # --- surface: query the real model instance point-by-point, the way a
    # single optimisation iteration would (one freestream at a time).
    mach_fine = np.linspace(mach_1d.min(), mach_1d.max(), 60)
    sec_fine = np.linspace(sec_1d.min(), sec_1d.max(), 60)
    Mach_f, Sec_f = np.meshgrid(mach_fine, sec_fine)
    cfx_fit = np.vectorize(
        lambda M, q: model.cf(freestream=_Freestream(M, q), length=1.0)
    )(Mach_f, Sec_f)

    scale = SECONDARY_SCALE[MODE]
    label = SECONDARY_LABEL[MODE]

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(projection="3d")
    ax1.plot_surface(Mach_f, Sec_f / scale, cfx_fit, cmap="viridis", alpha=0.7,
                      edgecolor="none", label="Cf model")
    Mach_grid, Sec_grid = np.meshgrid(mach_1d, sec_1d)
    ax1.scatter(Mach_grid, Sec_grid / scale, cfx_2d, c="k", s=25, label="training data")
    ax1.scatter(val[:, 0], val[:, 1] / scale, val[:, 2], c="red", s=25, label="validation data")
    ax1.set_xlabel("Mach")
    ax1.set_ylabel(label)
    ax1.set_zlabel(r"Cf(x) [m$^{0.139}$]", labelpad=12)
    ax1.tick_params(axis="z", pad=6)
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)
    ax1.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.95, 0.78),
               facecolor="white", framealpha=1.0)

    plt.show()


if __name__ == "__main__":
    main()
