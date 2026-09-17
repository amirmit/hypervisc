from calc_RefTemp import calc_ref_temperature, calc_air_transport
from ambiance import Atmosphere as Atm
import numpy as np

h = 0e3
M_vec = np.arange(0.1, 10.0 + 1e-9, 0.1)
# M_vec = np.array([1,3, 5, 7, 9])
Rex = 1e7
atm = Atm(h)
Te = float(atm.temperature[0])
Pe = float(atm.pressure[0])
props = calc_air_transport(p=Pe, T=Te)
rhoe = props.rho
mue = props.mu


Pr = np.zeros(len(M_vec))
Ts = np.zeros(len(M_vec))
mu = np.zeros(len(M_vec))
rho = np.zeros(len(M_vec))
Cf = np.zeros(len(M_vec))
Cf_ratio = np.zeros(len(M_vec))
for i, M in enumerate(M_vec):
    out = calc_ref_temperature(altitude=h, mach=M, lam_flg=False)
    Pr[i] = out.Pr
    Ts[i] = out.T_star
    mu[i] = out.transport.mu
    rho[i] = out.transport.rho

    Cf[i] = 0.02296 / Rex**0.139 * (rho[i] / rhoe)**(1-0.139) * (mu[i] / mue)**0.139
    # Cf[i] = 0.664 * np.sqrt(rho[i] * mu[i] / mue / rhoe)
    Cf_ratio[i] = (rho[i] / rhoe)**(1-0.139) * (mu[i] / mue)**0.139


# print(f"Cf = {Cf}")
print(f"Cf_ratio = {Cf_ratio}")


if __name__=="__main__":
    pass

