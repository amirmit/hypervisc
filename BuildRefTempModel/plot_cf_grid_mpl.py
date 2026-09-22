"""Interactive matplotlib verification plots for the Cf_x grid (Phase 1) --
run this directly (`python3 plot_cf_grid_mpl.py`), figures pop up in
matplotlib's own interactive windows (zoom/pan/rotate with the mouse), no
HTML/browser involved.

Plots the grid's native quantity, Cf_x(Mach, altitude) -- not a physical Cf
(that needs a vehicle length L via Cf = Cf_x * L^-n, not decided/wired up
yet -- see the friction-model plan).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_cf_grid import OUT_DIR

df = pd.read_csv(os.path.join(OUT_DIR, "cf_grid.csv"))

mach_1d = np.sort(df["mach"].unique())
alt_1d = np.sort(df["altitude"].unique())
cfx_2d = df.pivot(index="altitude", columns="mach", values="Cf_x").to_numpy()  # (n_alt, n_mach)

# --- 1. 3D surface: Cf_x vs (Mach, Altitude) --------------------------------
Mach_grid, Alt_grid_km = np.meshgrid(mach_1d, alt_1d / 1e3)
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(projection="3d")
surf = ax1.plot_surface(Mach_grid, Alt_grid_km, cfx_2d, cmap="viridis", edgecolor="none")
ax1.set_xlabel("Mach")
ax1.set_ylabel("Altitude [km]")
ax1.set_zlabel("Cf_x")
ax1.set_title("Cf_x(Mach, Altitude) -- 3D surface")
fig1.colorbar(surf, shrink=0.6, label="Cf_x")

# --- 2. Cf_x vs Mach, one line per altitude ---------------------------------
n_lines = 8
alt_idx = np.linspace(0, len(alt_1d) - 1, n_lines).round().astype(int)
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
for i in alt_idx:
    ax2.plot(mach_1d, cfx_2d[i, :], marker="o", ms=3, label=f"alt={alt_1d[i]/1e3:.0f} km")
ax2.set_xlabel("Mach")
ax2.set_ylabel("Cf_x")
ax2.set_title("Cf_x vs Mach, at fixed altitudes")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# --- 3. Cf_x vs Altitude, one line per Mach ---------------------------------
n_lines = 9
mach_idx = np.linspace(0, len(mach_1d) - 1, n_lines).round().astype(int)
fig3, ax3 = plt.subplots(figsize=(7.5, 5.5))
for j in mach_idx:
    ax3.plot(alt_1d / 1e3, cfx_2d[:, j], marker="o", ms=3, label=f"M={mach_1d[j]:.1f}")
ax3.set_xlabel("Altitude [km]")
ax3.set_ylabel("Cf_x")
ax3.set_title("Cf_x vs Altitude, at fixed Mach")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.show()
