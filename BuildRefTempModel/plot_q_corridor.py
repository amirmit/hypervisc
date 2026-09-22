"""Visualize the dynamic-pressure-bounded (q=20-100 kPa) region in the
(Mach, altitude) plane, against the current Cf_x grid box (M4-12,
alt15-45km) -- run directly, matplotlib interactive window, no HTML.

q = 0.5 * gamma * P(alt) * M^2 (ideal gas, gamma=1.4, same assumption as the
freestream u_e term in make_cf_grid.py). P(alt) is monotonic decreasing in
the standard atmosphere (unlike T, which has a stratospheric inversion), so
each (M, q) pair maps to a unique altitude -- q is a valid alternative grid
axis to alt if that turns out to be more useful than (Mach, altitude).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ambiance import Atmosphere as Atm

from make_cf_grid import OUT_DIR, MACH_RANGE, ALT_RANGE

GAMMA = 1.4
Q_LO, Q_HI = 20e3, 100e3  # Pa

# --- q(Mach, altitude) field over a context box wider than the current grid
mach_ctx = np.linspace(1.0, 14.0, 200)
alt_ctx = np.linspace(0.0, 50e3, 200)
P_ctx = np.array([float(Atm(a).pressure[0]) for a in alt_ctx])
Mach_grid, Alt_grid = np.meshgrid(mach_ctx, alt_ctx)
P_grid = np.tile(P_ctx.reshape(-1, 1), (1, len(mach_ctx)))
q_grid = 0.5 * GAMMA * P_grid * Mach_grid ** 2

fig, ax = plt.subplots(figsize=(8, 6.5))

# shaded corridor
mask = np.ma.masked_outside(q_grid, Q_LO, Q_HI)
ax.contourf(Mach_grid, Alt_grid / 1e3, np.ma.filled(mask, np.nan),
            levels=50, cmap="Blues", alpha=0.6)

# bounding isolines, bold
cs = ax.contour(Mach_grid, Alt_grid / 1e3, q_grid / 1e3, levels=[Q_LO / 1e3, Q_HI / 1e3],
                 colors=["tab:blue", "tab:red"], linewidths=2.5)
ax.clabel(cs, fmt=lambda v: f"q={v:.0f} kPa", fontsize=9)

# current grid box
box_mach = [MACH_RANGE[0], MACH_RANGE[1], MACH_RANGE[1], MACH_RANGE[0], MACH_RANGE[0]]
box_alt = [ALT_RANGE[0], ALT_RANGE[0], ALT_RANGE[1], ALT_RANGE[1], ALT_RANGE[0]]
ax.plot(box_mach, np.array(box_alt) / 1e3, "k--", lw=2,
         label=f"current grid box (M{MACH_RANGE[0]:.0f}-{MACH_RANGE[1]:.0f}, "
               f"alt{ALT_RANGE[0]/1e3:.0f}-{ALT_RANGE[1]/1e3:.0f}km)")

# existing 527 grid points, colored by inside/outside the corridor
df = pd.read_csv(os.path.join(OUT_DIR, "cf_grid.csv"))
P_pts = np.array([float(Atm(a).pressure[0]) for a in df["altitude"]])
q_pts = 0.5 * GAMMA * P_pts * df["mach"] ** 2
inside = (q_pts >= Q_LO) & (q_pts <= Q_HI)
ax.scatter(df["mach"][inside], df["altitude"][inside] / 1e3, s=8, c="green",
           label=f"grid points inside corridor ({inside.sum()}/{len(df)})")
ax.scatter(df["mach"][~inside], df["altitude"][~inside] / 1e3, s=8, c="gray", alpha=0.4,
           label="grid points outside corridor")

ax.set_xlabel("Mach")
ax.set_ylabel("Altitude [km]")
ax.set_title("q=20-100 kPa corridor vs. current Cf_x grid box")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(mach_ctx.min(), mach_ctx.max())
ax.set_ylim(alt_ctx.min() / 1e3, alt_ctx.max() / 1e3)
ax.grid(alpha=0.2)

plt.show()
