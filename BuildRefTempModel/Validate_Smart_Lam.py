from calc_RefTemp import calc_ref_temperature, calc_air_transport
from ambiance import Atmosphere as Atm
import numpy as np

h = 30e3
M_vec = np.arange(0.1, 6.0 + 1e-9, 0.1)
M_vec = np.array([1,2,3,4,5,6])

mue = 1.789e-5
Te = 226
Pr = 0.725
gamma = 1.4
S = 110

Cf = np.zeros(len(M_vec))

for i, M in enumerate(M_vec):
    r = np.sqrt(Pr)
    theta = 1.0 + 0.71 * r * (gamma-1)/2 * M**2
    rho_ratio = 1 / theta
    mu_ratio = theta**1.5 * (Te + S) / (theta * Te + S)
    Cf[i] = 0.664 * np.sqrt(rho_ratio * mu_ratio)

print(f"Cf = {Cf}")

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
Cf2 = np.zeros(len(M_vec))
for i, M in enumerate(M_vec):
    out = calc_ref_temperature(altitude=h, mach=M, lam_flg=True)
    Pr[i] = out.Pr
    Ts[i] = out.T_star
    mu[i] = out.transport.mu
    rho[i] = out.transport.rho

    Cf2[i] = 0.664 * np.sqrt(rho[i] * mu[i] / mue / rhoe)

print(f"Cf2 = {Cf2}")

if __name__=="__main__":
    pass

