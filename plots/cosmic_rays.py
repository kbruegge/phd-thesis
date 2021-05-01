import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u

# filename: (name, reference)
data = {
    'tibet.txt': ('Tibet', 'tibet-data', 'o'),
    'ice_top.txt': ('IceTop', 'icetop-data', 'v'),
    'kascade_grande.txt': ('Kascade Grande', 'kascade-data', 'h'),
    'hires_i.txt': ('HiRes I', 'hires-data', 's'),
    'hires_ii.txt': ('HiRes II', 'hires-data', 'D'),
    'auger.txt': ('Auger', 'auger-data', 'P'),
#     'yakutsk.txt': ('Yakutsk', 'yakutsk_data'),
}


cols = ['E', 'Flux', 'UncertLow', 'UncertHigh']
power = 2.7
# plt.figure(figsize=(14, 9))
for file, (name, reference, marker) in data.items():
    m = np.genfromtxt(f'./plots/data/cosmic_ray/{file}', skip_header=7, delimiter=';')
    df = pd.DataFrame(data=m, columns=cols)

    flux = df.Flux.values * u.Unit('m-2 s-1 sr-1 eV-1')
    flux_hi = df.UncertHigh.values * u.Unit('m-2 s-1 sr-1 eV-1')
    flux_lo = df.UncertLow.values * u.Unit('m-2 s-1 sr-1 eV-1')
    energy = df.E.values * u.eV

    target_unit = u.Unit('m-2 s-1 sr-1 TeV-1')

    energy = energy.to('TeV')
    flux = flux.to(target_unit)
    flux_lo = flux_lo.to(target_unit)
    flux_hi = flux_hi.to(target_unit)

    y = energy**power * flux
    y_up = energy**power * flux_lo
    y_low = energy**power * flux_hi
    plt.errorbar(energy.value, y.value, yerr=[y_low.value, y_up.value], label=f'{name}~\cite{{{reference}}}', fmt="o", lw=1, marker=marker)

plt.xlabel(f'Energy / \si{{{energy.unit}}}')

if power > 0:
    label = f'$\mathrm{{E}}^{{{power}}} \\frac{{\mathrm{{dN}}}}{{\mathrm{{dE}}}}$ / \si{{\\raiseto{{{(power - 1):.1f}}}\TeV\per\square\meter \per\second \per\steradian}}'
else:
    label = '$\\frac{\\mathrm{dN}}{\\mathrm{dE}}$ / \si{\TeV\per\square\meter \per\second \per\steradian}'
plt.ylabel(label)

plt.xscale('log')
plt.yscale('log')
plt.tight_layout(pad=0)
plt.legend()
plt.savefig('build/cosmic_rays.pgf')