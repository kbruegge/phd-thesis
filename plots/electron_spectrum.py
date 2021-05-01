import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from meyer_model import meyer_model

energy = np.logspace(-5.2, 4, 300) * u.TeV

T = meyer_model()
plt.plot(energy, energy**2 * T(energy), color='gray', lw=2)

# radio, wind = meyer_model_components()
# plt.plot(energy, energy**2 * radio(energy))
# plt.plot(energy, energy**2 * wind(energy))

plt.xscale('log')
plt.yscale('log')
plt.tight_layout(pad=0)
plt.savefig('./build/electron_spectrum.pdf')
