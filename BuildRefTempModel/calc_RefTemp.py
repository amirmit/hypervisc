from types import SimpleNamespace

import cea
import cea.matlab
from ambiance import Atmosphere as Atm

R_AIR = 287.0  # J/(kg.K), specific gas constant for air (matches the gamma check below)


def calc_air_transport(p, T, transport=True):
    """Equilibrium ideal-gas air properties at (p, T), for the reference
    temperature method (Cf calculation).

    Uses the NASA CEA Python package (`pip install cea`,
    https://github.com/nasa/cea) in place of gdtk's CEAGas model, which
    requires a compiled `cea2` executable not available on this machine.

    Parameters
    ----------
    p : float
        Static pressure [Pa].
    T : float
        Static temperature [K].

    Returns
    -------
    SimpleNamespace (SI units):
        mu   -- dynamic viscosity [Pa.s]
        k    -- thermal conductivity [W/(m.K)]
        Pr   -- Prandtl number (equilibrium)
        cp   -- specific heat [J/(kg.K)]
        rho  -- density [kg/m^3]
        raw  -- the underlying cea.matlab.eq_solve() SimpleNamespace
    """
    p_bar = p * 1e-5  # Pa -> bar (CEA's native pressure unit)

    soln = cea.matlab.eq_solve(
        cea.TP,
        ["Air"],
        T=T,
        P=p_bar,
        fuel_amounts=[0.0],
        oxid_amounts=[1.0],
        moles=False,
        transport=transport,
    )

    if not soln.converged:
        raise RuntimeError(
            f"CEA equilibrium solve did not converge (p={p}, T={T}, "
            f"last_error={soln.last_error})"
        )

    # Unit conversions verified against Pr = cp*mu/k and ideal-gas rho = p/(R T):
    # viscosity: millipoise -> Pa.s (x1e-4); conductivity: mW/(cm.K) -> W/(m.K) (x0.1);
    # cp: kJ/(kg.K) -> J/(kg.K) (x1e3); density is already kg/m^3.
    return SimpleNamespace(
        mu=soln.viscosity * 1e-4,
        k=soln.conductivity_eq * 0.1,
        Pr=soln.Pr_eq,
        cp=soln.cp_eq * 1e3,
        rho=soln.density,
        raw=soln,
    )


def calc_ref_temperature(altitude, mach, lam_flg=True, constGamma_flg=True, tol=1e-6, max_iter=100):
    """Meader-Smart reference temperature for a flat-plate/boundary-layer
    edge state at (altitude, mach), via the equilibrium-air Prandtl number.

        T* = Te * (1 + r * C * (gamma - 1)/2 * Me^2),   r = Pr^Pow

    For laminar flow: C=0.71, Pow=0.5
    For turbulent flow: C=0.66, Pow=1/3

    Pr (and gamma, via cp) depend on temperature, so this is solved by
    successive substitution: Pr/gamma are evaluated via CEA at the current
    T* guess, a new T* is computed, and the loop repeats until T* stops
    moving. Edge conditions (Te, Pe) are taken as the freestream static
    state at `altitude` (ICAO standard atmosphere via `ambiance`) — i.e.
    `mach` is the edge/flight Mach number, with no shock/compression
    applied upstream of the boundary layer.

    Note: this loop can fail to converge (oscillate) for high Mach numbers
    where T* lands near the O2/N2 dissociation onset (~2000-3000 K) — see
    session notes / ask before relying on it above ~M8-9.

    Parameters
    ----------
    altitude : float
        Geometric altitude [m].
    mach : float
        Edge (flight) Mach number [-].
    lam_flg:
        Defines C and Pow
        True - laminar flow
        False - turbulent flow
    cosntGamma_flg:
        True - gamma=1.4
        False - gamma=gamma(T*)
    tol : float
        Convergence tolerance on T* [K].
    max_iter : int
        Iteration cap.

    Returns
    -------
    SimpleNamespace:
        T_star  -- converged reference temperature [K]
        Pr      -- equilibrium Prandtl number at T_star
        gamma   -- cp/(cp - R_AIR) at T_star
        r       -- recovery factor, Pr^(1/3)
        Te, Pe  -- freestream static temperature [K] / pressure [Pa] at altitude
        n_iter  -- iterations taken to converge
        transport -- calc_air_transport() result evaluated at T_star
    """

    if lam_flg:
        C = 0.71
        Pow = 1.0/2.0
    else:
        C = 0.66
        Pow = 1.0/3.0

    atm = Atm(altitude)
    Te = float(atm.temperature[0])
    Pe = float(atm.pressure[0])

    T_guess = Te
    for n_iter in range(1, max_iter + 1):
        props = calc_air_transport(p=Pe, T=T_guess)
        if constGamma_flg:
            gamma = 1.4
        else:
            gamma = props.cp / (props.cp - R_AIR)
        r = props.Pr ** Pow
        T_star = Te * (1.0 + r * C * (gamma - 1.0) / 2.0 * mach**2)

        if abs(T_star - T_guess) < tol:
            T_guess = T_star
            break
        T_guess = T_star
    else:
        raise RuntimeError(
            f"Reference temperature loop did not converge in {max_iter} "
            f"iterations (altitude={altitude}, mach={mach})"
        )

    return SimpleNamespace(
        T_star=T_guess,
        Pr=props.Pr,
        gamma=gamma,
        r=r,
        Te=Te,
        Pe=Pe,
        n_iter=n_iter,
        transport=props,
    )


if __name__ == '__main__':

    result = calc_air_transport(p=1000, T=2000)

    print('Pr  = ', result.Pr)
    print('mu  = ', result.mu, 'Pa.s')
    print('k   = ', result.k, 'W/(m.K)')
    print('cp  = ', result.cp, 'J/(kg.K)')
    print('rho = ', result.rho, 'kg/m^3')
    print('gamma = ', result.cp / (result.cp - 287))

    print()
    ref = calc_ref_temperature(altitude=32.5e3, mach=10
                               , max_iter=200)
    print(f"T*  = {ref.T_star:.2f} K  (Te = {ref.Te:.2f} K, Pe = {ref.Pe:.2f} Pa)")
    print(f"Pr  = {ref.Pr:.4f}, r = {ref.r:.4f}, gamma = {ref.gamma:.4f}")
    print(f"converged in {ref.n_iter} iterations")