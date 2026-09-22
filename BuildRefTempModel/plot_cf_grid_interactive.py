"""Interactive (Plotly) verification plots for the Cf_x grid -- companion to
plot_cf_grid.py's static PNGs. Run after make_cf_grid.py. Produces a single
self-contained HTML file (plotly.js inlined, works offline / over RDP with
no server needed -- just open it in a browser)."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RectBivariateSpline

from calc_RefTemp import calc_ref_temperature, calc_air_transport
from make_cf_grid import R_AIR, N_EXP, C0_AVG, OUT_DIR

df = pd.read_csv(os.path.join(OUT_DIR, "cf_grid.csv"))

n_mach = df["mach"].nunique()
n_alt = df["altitude"].nunique()
mach_1d = np.sort(df["mach"].unique())
alt_1d = np.sort(df["altitude"].unique())

fields = ["Cf_x", "T_star", "n_iter", "Pr", "gamma"]
grids = {
    f: df.pivot(index="altitude", columns="mach", values=f).to_numpy() for f in fields
}
alt_km = alt_1d / 1e3

# customdata stack for hover: [Cf_x, T_star, Pr, gamma, n_iter] at every point,
# regardless of which field is the displayed z -- so hovering any panel shows
# the full state at that grid point.
customdata = np.stack([grids[f] for f in fields], axis=-1)
hover_tmpl = (
    "Mach=%{x:.2f}  Alt=%{y:.1f} km<br>"
    "Cf_x=%{customdata[0]:.5f}<br>"
    "T*=%{customdata[1]:.1f} K<br>"
    "Pr=%{customdata[2]:.4f}  gamma=%{customdata[3]:.4f}<br>"
    "n_iter=%{customdata[4]:.0f}<extra></extra>"
)

# --- Panel 1: three linked contour maps (Cf_x, T_star, n_iter) -------------
fig1 = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Cf_x", "T* [K]", "n_iter (convergence)"),
    horizontal_spacing=0.08,
)
panel_specs = [("Cf_x", "Viridis"), ("T_star", "Inferno"), ("n_iter", "Magma")]
for i, (field, cscale) in enumerate(panel_specs, start=1):
    fig1.add_trace(
        go.Contour(
            x=mach_1d, y=alt_km, z=grids[field],
            colorscale=cscale, ncontours=25,
            customdata=customdata,
            hovertemplate=hover_tmpl,
            colorbar=dict(len=0.9, x=i / 3 - 0.02, title=field),
            contours=dict(coloring="heatmap"),
        ),
        row=1, col=i,
    )
    fig1.update_xaxes(title_text="Mach", row=1, col=i)
fig1.update_yaxes(title_text="Altitude [km]", row=1, col=1)
fig1.update_layout(
    title="Meador-Smart Cf_x grid -- hover for full state at any point, zoom/pan freely",
    height=520, width=1500,
)

# --- Panel 2: Cf vs Mach, altitude slider, direct recompute vs grid spline -
spline = RectBivariateSpline(alt_1d, mach_1d, grids["Cf_x"], kx=3, ky=3)
mach_line = np.linspace(mach_1d.min(), mach_1d.max(), 41)
L_check = 10.0  # m, representative vehicle length

frames = []
for alt in alt_1d:
    direct_cf = []
    direct_niter = []
    for M in mach_line:
        ref = calc_ref_temperature(altitude=alt, mach=M, lam_flg=False, constGamma_flg=True)
        rho_star, mu_star = ref.transport.rho, ref.transport.mu
        edge = calc_air_transport(p=ref.Pe, T=ref.Te, transport=False)
        rho_e = edge.rho
        u_e = M * np.sqrt(ref.gamma * R_AIR * ref.Te)
        cfx = C0_AVG * rho_star ** (1 - N_EXP) * mu_star ** N_EXP / rho_e * u_e ** (-N_EXP)
        direct_cf.append(cfx * L_check ** (-N_EXP))
        direct_niter.append(ref.n_iter)
    direct_cf = np.array(direct_cf)
    direct_niter = np.array(direct_niter)

    cfx_grid_line = spline(alt, mach_line).flatten()
    cf_grid_line = cfx_grid_line * L_check ** (-N_EXP)

    frames.append(
        go.Frame(
            name=f"{alt/1e3:.1f}",
            data=[
                go.Scatter(x=mach_line, y=direct_cf, mode="lines", line=dict(color="black", width=2),
                           name="direct calc_ref_temperature"),
                go.Scatter(x=mach_line, y=cf_grid_line, mode="lines", line=dict(color="red", width=2, dash="dash"),
                           name="grid spline (Cf_x * L^-n)"),
                go.Scatter(x=mach_line, y=direct_cf, mode="markers",
                           marker=dict(size=7, color=direct_niter, colorscale="Magma",
                                       colorbar=dict(title="n_iter", x=1.02), cmin=3, cmax=35),
                           customdata=direct_niter, hovertemplate="Mach=%{x:.2f}<br>Cf=%{y:.5f}<br>n_iter=%{customdata:.0f}<extra></extra>",
                           name="n_iter (color)"),
            ],
        )
    )

fig2 = go.Figure(
    data=frames[0].data,
    frames=frames,
    layout=go.Layout(
        title=f"Cf vs Mach (L={L_check} m) -- drag the altitude slider; marker color = convergence iteration count",
        xaxis_title="Mach",
        yaxis_title=f"Cf (average, L={L_check}m)",
        height=550, width=1000,
        sliders=[dict(
            active=len(frames) // 2,
            currentvalue=dict(prefix="Altitude = ", suffix=" km"),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for f in frames],
        )],
    ),
)
fig2.update_traces(selector=dict(name=None))

# --- combine into one HTML file --------------------------------------------
out_path = os.path.join(OUT_DIR, "cf_grid_interactive.html")
with open(out_path, "w") as f:
    f.write("<html><head><title>Cf_x grid -- interactive</title></head><body>\n")
    f.write(fig1.to_html(full_html=False, include_plotlyjs="inline"))
    f.write("<hr>\n")
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
    f.write("</body></html>\n")

print(f"wrote {out_path}")
