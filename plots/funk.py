import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy import constants
from naima.radiative import Synchrotron, InverseCompton
from naima.models import BrokenPowerLaw

e = np.logspace(-4, 14.5, 1000) * u.eV

electrons = BrokenPowerLaw(
    1e43 / u.eV, e_0=1 * u.GeV, e_break=1e12 * u.eV, alpha_1=2, alpha_2=3
)
syn_model = Synchrotron(electrons, B=30 * u.uG)

#     SYN = Synchrotron(T, B=B * u.uG, Eemax=50 * u.PeV, Eemin=0.01 * u.GeV, nEed=precision)

# Compute photon density spectrum from synchrotron emission assuming R=2.1 pc
Rpwn = 2.1 * u.pc
Esy = np.logspace(-10, 10, 50) * u.MeV
Lsy = syn_model.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * constants.c) * 2.25

ic_model = InverseCompton(electrons, seed_photon_fields=["CMB", ["SSC", Esy, phn_sy]])
ic_model_cmb = InverseCompton(electrons, seed_photon_fields=["CMB"])
ic_model_fir = InverseCompton(electrons, seed_photon_fields=["FIR"])
ic_model_ssc = InverseCompton(
    electrons, seed_photon_fields=[["SSC", Esy, phn_sy]], nEed=120
)

plt.plot(e, syn_model.sed(e), color="gray")
plt.plot(e, ic_model.sed(e), color="gray")
plt.plot(e, ic_model_cmb.sed(e), color="silver", linestyle="dotted", label="IC CMB")

plt.plot(e, ic_model_ssc.sed(e), color="darkgray", linestyle="--", label="IC SSC")
plt.plot(
    e,
    syn_model.sed(e) + ic_model.sed(e),
    color="crimson",
    lw=2,
    label="IC + Synchrotron",
)
plt.yscale("log")
plt.xscale("log")
plt.ylabel("$\\mathrm{E}^2 \\frac{\\mathrm{dN}}{\\mathrm{dE}}$ / arbitrary units")
plt.xlabel("Energy / arbitrary units")

plt.ylim(3e-11, 6e-7)
plt.xlim(e.min().value, e.max().value)

plt.text(3.5e-4, 1.3e-8, "$s=\\frac{p_1 + 1}{2}$", rotation=50, fontsize=8.5)
plt.text(0.5e2, 2.5e-7, "$s=\\frac{p_2 + 1}{2}$", fontsize=8.5)

plt.text(11e6, 0.8e-9, "$s=\\frac{p_1 + 1}{2}$", rotation=50, fontsize=8.5)
plt.text(2.3e12, 1.5e-9, "$s=p_1 + 1$", rotation=-42, fontsize=8.5)

plt.text(5e0, 2e-8, "Synchrotron", fontsize=13, alpha=0.7)
plt.text(5e8, 2e-8, "Inverse Compton", fontsize=13, alpha=0.7)

plt.text(8e10, 1e-9, "CMB", fontsize=8.5, color="silver")

plt.text(0.5e13, 2.5e-10, "SSC", fontsize=8.5, color="darkgray", rotation=-60)

plt.tight_layout(pad=0)
params = dict(right=1)
plt.gcf().subplots_adjust(**params)
plt.savefig("build/funk.pdf")
