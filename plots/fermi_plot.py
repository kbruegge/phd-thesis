from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm

import matplotlib
print("USING CONFIG FILE FOR MPL -->: ", matplotlib.matplotlib_fname())
print("USING BACKEND FOR MPL --->", matplotlib.get_backend())

t = Table.read('./plots/data/4fgl.fits', hdu=1)

t.sort('Flux1000')
coords = SkyCoord(t['GLON'], t['GLAT'], frame='galactic')

size = plt.gcf().get_size_inches()
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'mollweide'}, figsize=(size[0], size[1]*0.76))

x = coords.l.wrap_at('180 deg').rad
y = coords.b.rad

c = t['Energy_Flux100'].to('TeV/(cm2 s)').to_value('TeV/(cm2 s)') # wtf matplotlib 3.1 fuckup with masked columns

scatter = ax.scatter(x, y, c=c, cmap='YlGnBu_r', norm=LogNorm(), alpha=0.9, s=8, edgecolors='none')

ax.set_facecolor('black')

ax.set_yticklabels([])
ax.set_xticklabels([])
plt.grid(False)

plt.colorbar(scatter, fraction=0.0235, pad=0.04, label='Flux / \si{\TeV\per\square\centi\meter \per\second}')
plt.tight_layout(pad=0,  rect=(0, 0, 0.998, 1))

plt.savefig('build/4fgl.pdf', )  # left, bottom, right and top.
