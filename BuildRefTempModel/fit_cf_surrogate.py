"""Phase 2: fit a smooth Cf_x(Mach, secondary) model from the offline grid
(make_cf_grid.py) and pickle it for hypervisc to load. Works on either grid
mode (--mode alt/q); the alt-vs-q decision is still open, so both get
fit+validated here and the choice is made later at wiring time.

Method: tensor-product quadratic B-spline (scipy RectBivariateSpline,
kx=ky=2) over log10(Cf_x) -- appropriate for gridded (not scattered) data
where no gradient w.r.t. Mach/altitude/q is ever needed (those are fixed
per optimisation run, not DVs -- unlike the nozzle_v3 Kriging surrogate,
which needs predict_derivatives w.r.t. its DVs). log10 space keeps the fit
positive everywhere and handles Cf_x's ~10x dynamic range. Plain
interpolating spline (s=0): the grid was already decimated to remove
CEA-iteration noise, so there's nothing left to statistically smooth over.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from make_cf_grid import grid_dir, cf_x_point, altitude_from_pressure, GAMMA_FREESTREAM, N_EXP

SECONDARY_NAME = dict(alt="altitude", q="q_target")
SECONDARY_LABEL = dict(alt="Altitude [km]", q="q [kPa]")
SECONDARY_SCALE = dict(alt=1e3, q=1e3)  # native units -> plot units (km, kPa)


def load_grid(mode):
    df = pd.read_csv(os.path.join(grid_dir(mode), "cf_grid.csv"))
    mach_1d = np.sort(df["mach"].unique())
    sec_1d = np.sort(df[SECONDARY_NAME[mode]].unique())
    cfx_2d = df.pivot(index=SECONDARY_NAME[mode], columns="mach", values="Cf_x").to_numpy()
    return mach_1d, sec_1d, cfx_2d


def fit_spline(mach_1d, sec_1d, cfx_2d, s=1e-2):
    log_cfx_2d = np.log10(cfx_2d)
    # RectBivariateSpline wants (x, y, z) with z[i,j] = f(x[i], y[j]) --
    # our cfx_2d is (n_sec, n_mach), so x=sec, y=mach here.
    spline = RectBivariateSpline(sec_1d, mach_1d, log_cfx_2d, kx=2, ky=2, s=s)
    return spline


def predict(spline, mach, sec):
    return 10 ** spline(sec, mach, grid=False)


def make_validation_points(mode, mach_1d, sec_1d, n_per_axis=4):
    """Points strictly BETWEEN grid nodes -- a real accuracy check, not a
    trivial near-zero interpolation residual at the fitted nodes."""
    mach_mid = 0.5 * (mach_1d[:-1] + mach_1d[1:])
    sec_mid = 0.5 * (sec_1d[:-1] + sec_1d[1:])
    pts = []
    for M in mach_mid:
        for sec in sec_mid:
            if mode == "alt":
                altitude = sec
            else:
                P_target = 2 * sec / (GAMMA_FREESTREAM * M ** 2)
                altitude = altitude_from_pressure(P_target)
            row = cf_x_point(M, altitude)
            pts.append((M, sec, row["Cf_x"]))
    return np.array(pts)  # columns: mach, secondary, Cf_x_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alt", "q"], default="q")
    args = parser.parse_args()
    mode = args.mode

    mach_1d, sec_1d, cfx_2d = load_grid(mode)
    spline = fit_spline(mach_1d, sec_1d, cfx_2d)

    # Self-contained payload: hypervisc (installed, runtime) must never import
    # BuildRefTempModel/calc_RefTemp/CEA/ambiance, so everything cf() needs
    # (including the n_exp constant) has to live in the pickle itself.
    pkl_path = os.path.join(grid_dir(mode), "cf_x_surrogate.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(mode=mode, spline=spline, n_exp=N_EXP,
                          mach_bounds=(mach_1d.min(), mach_1d.max()),
                          sec_bounds=(sec_1d.min(), sec_1d.max())), f)
    print(f"wrote {pkl_path}")

    # --- validation against fresh off-grid points --------------------------
    val = make_validation_points(mode, mach_1d, sec_1d)
    val_pred = predict(spline, val[:, 0], val[:, 1])
    rel_err = np.abs(val_pred - val[:, 2]) / val[:, 2] * 100
    print(f"mode={mode}: {len(val)} off-grid validation points, "
          f"rel error median={np.median(rel_err):.2f}%  max={np.max(rel_err):.2f}%")

    # --- plots ---------------------------------------------------------------
    scale = SECONDARY_SCALE[mode]
    label = SECONDARY_LABEL[mode]

    # 3D: fit surface (fine mesh) + raw grid nodes + off-grid validation points
    mach_fine = np.linspace(mach_1d.min(), mach_1d.max(), 60)
    sec_fine = np.linspace(sec_1d.min(), sec_1d.max(), 60)
    Mach_f, Sec_f = np.meshgrid(mach_fine, sec_fine)
    cfx_fit = predict(spline, Mach_f.ravel(), Sec_f.ravel()).reshape(Mach_f.shape)

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(projection="3d")
    ax1.plot_surface(Mach_f, Sec_f / scale, cfx_fit, cmap="viridis", alpha=0.7, edgecolor="none")
    Mach_grid, Sec_grid = np.meshgrid(mach_1d, sec_1d)
    ax1.scatter(Mach_grid, Sec_grid / scale, cfx_2d, c="k", s=25, label="training data")
    ax1.scatter(val[:, 0], val[:, 1] / scale, val[:, 2], c="red", s=25, label="validation data")
    ax1.set_xlabel("Mach")
    ax1.set_ylabel(label)
    ax1.set_zlabel("Cf(x)", labelpad=12)
    ax1.tick_params(axis="z", pad=6)
    ax1.legend(fontsize=8, loc="upper left")

    # Cf_x vs Mach at fixed secondary, fit curve vs raw nodes vs validation
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    for sec in sec_1d:
        mask = np.isclose(val[:, 1], sec) if mode == "alt" else None
        ax2.plot(mach_fine, predict(spline, mach_fine, np.full_like(mach_fine, sec)),
                  lw=1.5, label=f"{label.split('[')[0].strip()}={sec/scale:.0f}")
        ax2.plot(mach_1d, cfx_2d[sec_1d == sec, :].flatten(), "ko", ms=4)
    # nearest-secondary validation points overlaid as x markers
    for M, sec, cfx_true in val:
        i = np.argmin(np.abs(sec_1d - sec))
        ax2.plot(M, cfx_true, "rx", ms=6)
    ax2.set_xlabel("Mach")
    ax2.set_ylabel("Cf_x")
    ax2.set_title(f"Fit vs grid nodes (o) vs off-grid validation (x), mode={mode}")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
