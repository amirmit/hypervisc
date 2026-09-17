from calc_RefTemp import calc_ref_temperature, calc_air_transport
from ambiance import Atmosphere as Atm
import numpy as np

h = 35e3
M = 10
L = 4

atm = Atm(h)
Te = float(atm.temperature[0])
Pe = float(atm.pressure[0])
props = calc_air_transport(p=Pe, T=Te)
rhoe = props.rho
mue = props.mu

R = 287
gamma = 1.4
sos_e = np.sqrt(gamma * R * Te)
ue = sos_e * M
ReL = rhoe * ue * L / mue

# out = calc_ref_temperature(altitude=h, mach=M, lam_flg=False)
out = calc_ref_temperature(altitude=h, mach=M, lam_flg=False)
Pr_s = out.Pr
T_s = out.T_star
mu_s = out.transport.mu
rho_s = out.transport.rho
Re_s = rho_s * ue * L / mu_s
Cf_t = 0.02667 / ReL**0.139 * (rho_s / rhoe)**(1-0.139) * (mu_s / mue)**0.139

print(f"Re_s = {Re_s}")
print(f"Re_L = {ReL}")

Re_T = 10**(6.421*np.exp(1.209e-4*M**2.641))
print(f"Re_T = {Re_T}")

r = np.sqrt(0.725)
theta = 1.0 + 0.71 * r * (1.4-1)/2 * M**2
rho_ratio = 1 / theta
mu_ratio = theta**1.5 * (Te + 110) / (theta * Te + 110)
rho_s_lam = rho_ratio * rhoe
mu_s_lam = mu_ratio * mue
Re_l = rho_s_lam * ue * L / mu_s_lam
Cf_l = 1.328 / np.sqrt(ReL) * np.sqrt(rho_ratio * mu_ratio)
print(f"Re_lam = {Re_l}")

print(f"Cf_tub = {Cf_t}")
print(f"Cf_lam = {Cf_l}")

print(f"Ts_turb = {T_s}")
print(f"Ts_lam = {theta*Te}")


if __name__=="__main__":
    pass

