"""Offline Cf_x grid generation: turbulent Meador-Smart reference-temperature
skin-friction coefficient over (Mach, altitude) OR (Mach, dynamic pressure),
for the hypervisc Cf surrogate (Phase 1 of the friction-model plan). Grid
values only -- no design-variable dependence, since Mach/altitude are fixed
per optimisation run, not DVs.

Two grid modes, selected with --mode:
  alt  -- grid directly over (Mach, altitude), MACH_RANGE x ALT_RANGE.
  q    -- grid over (Mach, dynamic pressure q), MACH_RANGE x Q_RANGE. Each
          (M, q) point implies a static pressure P = 2*q/(gamma*M^2) (ideal
          gas, same gamma=1.4 assumed for u_e below); altitude is then
          recovered via ambiance.Atmosphere.from_pressure(P), an exact
          analytic inverse of the ICAO standard atmosphere's P(altitude)
          (verified machine-precision round-trip -- P(altitude) is
          monotonic decreasing, unlike T, which has a stratospheric
          inversion and so cannot be inverted this way). Once altitude is
          known, the point is evaluated exactly like alt-mode.
q-mode exists because a rectangular (Mach, altitude) box is a poor match to
a real flight envelope: at fixed Mach, q changes a lot with altitude, so a
box drawn in (M, alt) either wastes most of its points on physically
unreachable (M, alt) combinations for a q-bounded vehicle, or -- if drawn
tight enough to avoid that -- can't cover the (M, alt) pairs the trajectory
actually needs at a different q. See plot_q_corridor.py for the geometry.

Formula (length-averaged turbulent Meador-Smart, i.e. one Cf for the whole
vehicle, not a local Cf(x)):

    Cf(L) = 0.02667 * Re_L^-n * (rho*/rho_e)^(1-n) * (mu*/mu_e)^n,   n=0.139
    Re_L  = rho_e * u_e * L / mu_e

0.02667 = 0.02296/(1-n): the standard length-average of the local turbulent
correlation Cf_x(x) = 0.02296 * Re_x^-n * (...) used in Validate_Smart.py.

mu_e cancels algebraically (it appears as (mu_e)^-n from the ratio term and
(mu_e)^+n from Re_L^-n), leaving a piece that depends only on (Mach,
altitude) -- never on L -- which is what's tabulated/surrogated here:

    Cf_x(M, alt) = 0.02667 * rho*^(1-n) * mu*^n / rho_e * u_e^-n
    Cf(L)        = Cf_x(M, alt) * L^-n

rho*, mu* are equilibrium-air CEA properties at the Meador-Smart reference
temperature T* (hot -- dissociation matters). rho_e, u_e are freestream
(boundary-layer-edge) quantities: rho_e via CEA at (Pe, Te) for consistency
with rho* (matches Calc_Cf_6.5.py); u_e = M * sqrt(gamma * R_AIR * Te) using
ideal-gas gamma=1.4 (freestream static T is ~200-300 K in this envelope --
negligible real-gas effect on sos, and consistent with constGamma_flg=True
below since ref.gamma is then always 1.4 anyway).

constGamma_flg=True (fixed gamma=1.4 in the T* iteration) is required, not
optional, over the requested envelope: constGamma_flg=False (T*-dependent
dissociation gamma) fails to converge at the high-Mach corners (checked at
M=12, alt=15/45 km -- see session notes). This trades some accuracy at the
hottest corner (M12/45km -> T*~4560 K) for a method that actually converges
everywhere in [4,12] x [15,45] km.
"""

import argparse
import csv
import os
import time

import numpy as np
from ambiance import Atmosphere as Atm

from calc_RefTemp import calc_ref_temperature, calc_air_transport

R_AIR = 287.0
N_EXP = 0.139
C0_AVG = 0.02667  # length-averaged turbulent Meador-Smart constant
GAMMA_FREESTREAM = 1.4  # ideal-gas freestream assumption (u_e, and q<->P below);
                         # matches constGamma_flg=True's fixed gamma in cf_x_point

MACH_RANGE = (4.0, 12.0)
N_MACH = 5  # step 0.5

ALT_RANGE = (15e3, 45e3)  # m, alt-mode
N_ALT = 5  # step 1 km

Q_RANGE = (20e3, 120e3)  # Pa, q-mode
N_Q = 5  # step 5 kPa


def grid_dir(mode):
    return os.path.join(os.path.dirname(__file__), f"cf_grid_v1_{mode}")


# Backward-compat default for the plotting scripts already written against
# the alt-mode grid (plot_cf_grid.py, plot_cf_grid_interactive.py,
# plot_q_corridor.py, plot_cf_grid_mpl.py) -- unaffected by --mode q runs.
OUT_DIR = grid_dir("alt")


def altitude_from_pressure(P_target):
    """Exact analytic inverse of the ICAO standard atmosphere's
    P(altitude) (ambiance.Atmosphere.from_pressure) -- verified
    machine-precision round-trip against Atmosphere(h).pressure. Valid
    because P(altitude) is monotonic decreasing (unlike T, which has a
    stratospheric inversion and so has no unique inverse)."""
    return float(Atm.from_pressure(P_target).h[0])


def cf_x_point(mach, altitude):
    ref = calc_ref_temperature(
        altitude=altitude, mach=mach, lam_flg=False, constGamma_flg=True
    )
    rho_star = ref.transport.rho
    mu_star = ref.transport.mu

    edge = calc_air_transport(p=ref.Pe, T=ref.Te, transport=True)
    rho_e = edge.rho
    mu_e = edge.mu
    u_e = mach * np.sqrt(ref.gamma * R_AIR * ref.Te)
    Re_x = rho_e * u_e / mu_e

    cf_x = C0_AVG / Re_x ** N_EXP * (rho_star / rho_e) ** (1 - N_EXP) * (mu_star / mu_e) ** N_EXP
    # cf_x = C0_AVG * rho_star ** (1 - N_EXP) * mu_star ** N_EXP / rho_e * u_e ** (-N_EXP)

    q = 0.5 * GAMMA_FREESTREAM * ref.Pe * mach ** 2

    return dict(
        mach=mach,
        altitude=altitude,
        q=q,
        T_star=ref.T_star,
        Pr=ref.Pr,
        gamma=ref.gamma,
        rho_star=rho_star,
        mu_star=mu_star,
        Te=ref.Te,
        Pe=ref.Pe,
        rho_e=rho_e,
        u_e=u_e,
        n_iter=ref.n_iter,
        Cf_x=cf_x,
    )


def build_grid(mode):
    mach_vec = np.linspace(*MACH_RANGE, N_MACH)

    if mode == "alt":
        secondary_vec = np.linspace(*ALT_RANGE, N_ALT)
    elif mode == "q":
        secondary_vec = np.linspace(*Q_RANGE, N_Q)
    else:
        raise ValueError(f"unknown mode {mode!r}, expected 'alt' or 'q'")

    rows = []
    t0 = time.time()
    n_fail = 0
    for M in mach_vec:
        for sec in secondary_vec:
            try:
                if mode == "alt":
                    altitude = sec
                else:
                    P_target = 2 * sec / (GAMMA_FREESTREAM * M ** 2)
                    altitude = altitude_from_pressure(P_target)
                row = cf_x_point(M, altitude)
                if mode == "q":
                    row["q_target"] = sec
                    # consistency check: q reconstructed from the recovered
                    # altitude's actual Pe should match the target q --
                    # confirms altitude_from_pressure() round-trips exactly.
                    row["q_err"] = row["q"] - sec
                rows.append(row)
            except Exception as e:
                n_fail += 1
                print(f"FAILED at M={M}, secondary={sec}: {e}")
    t1 = time.time()

    print(f"mode={mode}: {len(rows)} points ok, {n_fail} failed, {t1 - t0:.1f}s total")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["alt", "q"], default="q")
    args = parser.parse_args()

    out_dir = grid_dir(args.mode)
    os.makedirs(out_dir, exist_ok=True)

    rows = build_grid(args.mode)

    csv_path = os.path.join(out_dir, "cf_grid.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
