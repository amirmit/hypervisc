"""Verification plots for the Cf_x grid (make_cf_grid.py output) -- Phase 1
of the friction-model plan. Run after make_cf_grid.py."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from calc_RefTemp import calc_ref_temperature, calc_air_transport
from make_cf_grid import R_AIR, N_EXP, C0_AVG, OUT_DIR

df = pd.read_csv(os.path.join(OUT_DIR, "cf_grid.csv"))

n_mach = df["mach"].nunique()
n_alt = df["altitude"].nunique()
mach_grid = df["mach"].to_numpy().reshape(n_mach, n_alt)
alt_grid = df["altitude"].to_numpy().reshape(n_mach, n_alt) / 1e3
cfx_grid = df["Cf_x"].to_numpy().reshape(n_mach, n_alt)
niter_grid = df["n_iter"].to_numpy().reshape(n_mach, n_alt)
tstar_grid = df["T_star"].to_numpy().reshape(n_mach, n_alt)

# --- 1. Cf_x contour over (Mach, altitude) ---------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(mach_grid, alt_grid, cfx_grid, levels=25, cmap="viridis")
fig.colorbar(cs, ax=ax, label="Cf_x")
ax.set_xlabel("Mach")
ax.set_ylabel("Altitude [km]")
ax.set_title("Meador-Smart Cf_x(Mach, altitude) -- turbulent, length-averaged")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cf_x_contour.png"), dpi=150)
plt.close(fig)

# --- 2. T* contour (diagnostic: where is the reference temperature hottest) -
fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(mach_grid, alt_grid, tstar_grid, levels=25, cmap="inferno")
fig.colorbar(cs, ax=ax, label="T* [K]")
ax.set_xlabel("Mach")
ax.set_ylabel("Altitude [km]")
ax.set_title("Reference temperature T*(Mach, altitude)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "t_star_contour.png"), dpi=150)
plt.close(fig)

# --- 3. Convergence diagnostic: iteration count ------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
cs = ax.contourf(mach_grid, alt_grid, niter_grid, levels=20, cmap="magma")
fig.colorbar(cs, ax=ax, label="n_iter")
ax.set_xlabel("Mach")
ax.set_ylabel("Altitude [km]")
ax.set_title("Reference-temperature loop iteration count (convergence diagnostic)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "n_iter_contour.png"), dpi=150)
plt.close(fig)

# --- 4. Cf(L) vs Mach at a representative altitude/length, compared to a ---
#        fresh direct (non-grid, non-interpolated) calc_ref_temperature call
#        at each point -- sanity-checks that reading Cf_x off the grid and
#        applying L^-n reproduces the direct formula exactly.
alt_check = 30e3
L_check = 10.0  # m, representative vehicle length
mach_line = np.linspace(4, 12, 33)

cf_direct = []
for M in mach_line:
    ref = calc_ref_temperature(altitude=alt_check, mach=M, lam_flg=False, constGamma_flg=True)
    rho_star, mu_star = ref.transport.rho, ref.transport.mu
    edge = calc_air_transport(p=ref.Pe, T=ref.Te, transport=False)
    rho_e = edge.rho
    u_e = M * np.sqrt(ref.gamma * R_AIR * ref.Te)
    Re_L = rho_e * u_e * L_check / calc_air_transport(p=ref.Pe, T=ref.Te).mu
    cf_direct.append(C0_AVG * Re_L ** (-N_EXP) * (rho_star / rho_e) ** (1 - N_EXP) *
                      (mu_star / calc_air_transport(p=ref.Pe, T=ref.Te).mu) ** N_EXP)
cf_direct = np.array(cf_direct)

# from the grid: interpolate Cf_x at (mach_line, alt_check), then apply L^-n
from scipy.interpolate import RectBivariateSpline
mach_1d = df["mach"].unique()
alt_1d = df["altitude"].unique()
cfx_2d = df.pivot(index="mach", columns="altitude", values="Cf_x").to_numpy()
spline = RectBivariateSpline(mach_1d, alt_1d, cfx_2d, kx=3, ky=3)
cfx_from_grid = spline(mach_line, alt_check).flatten()
cf_from_grid = cfx_from_grid * L_check ** (-N_EXP)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(mach_line, cf_direct, "k-", lw=2, label="direct calc_ref_temperature (no grid/fit)")
ax.plot(mach_line, cf_from_grid, "r--", lw=2, label=f"grid Cf_x * L^-n (L={L_check}m)")
ax.set_xlabel("Mach")
ax.set_ylabel(f"Cf (average, L={L_check}m)")
ax.set_title(f"Cf vs Mach at alt={alt_check/1e3:.0f}km -- grid-derived vs direct formula")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cf_vs_mach_check.png"), dpi=150)
plt.close(fig)

max_rel_err = np.max(np.abs(cf_from_grid - cf_direct) / cf_direct)
print(f"max relative error (grid-interp vs direct formula) = {max_rel_err:.2e}")
print("plots written to", OUT_DIR)
