import matplotlib.pyplot as plt

import numpy as np
from gammapy.maps import Map
from mpl_toolkits.axes_grid1 import make_axes_locatable


def setfont(font, text):
    s = f'{{\\fontspec{{{font}}}{text}}}'
    return s


path = './plots/data/hgps_map_flux_0.1deg_v1.fits.gz'
survey_map = Map.read(path)
factor = 1E12
survey_map.data *= factor

kw = {}
size = plt.gcf().get_size_inches()
fig, axs = plt.subplots(4, 1, figsize=(size[0], 3.4))
# fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=-1.3,)

width = 40
x0 = 260

cmap = 'YlGnBu_r'
stretch = 'sqrt'
vmax = 2

lat_range = [-2.5, 2.3]
offsets = [0, 35, 70, 105]
axs = axs.ravel()
for ax, offset in zip(axs, offsets):
    survey_map.plot(ax=ax, fig=fig, stretch=stretch, cmap=cmap, vmin=0.0, vmax=vmax)
    x_cutout = np.array([x0, x0 + width]) + offset
    xlim_pix, ylim_pix = survey_map.geom.wcs.wcs_world2pix(x_cutout, lat_range, 1)

    ax.set_xlim(xlim_pix)
    ax.set_ylim(ylim_pix)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


arrowprops = dict(facecolor='white', arrowstyle='-')
# position of sagittarius a in galactic coordinates
sgr_a = (359.94423568, -0.04616002)
vela_junior = (266.28, -1.24)

xy = np.array(survey_map.geom.wcs.wcs_world2pix(vela_junior[0], vela_junior[1], 1))
axs[0].annotate(xy=xy, s=setfont('Minion Pro', 'RX J0852'), fontsize=10, color='white', arrowprops=arrowprops, xytext=xy + [-250, 10])

xy = np.array(survey_map.geom.wcs.wcs_world2pix(sgr_a[0], sgr_a[1], 1))
axs[2].annotate(xy=xy, s=setfont('Minion Pro', 'Sagittarius A'), fontsize=10, color='white', arrowprops=arrowprops, xytext=xy + [500, 30])


im = axs[2].get_images()[0]
# factor = 1E-12
label = f'Flux \\num{{E-12}}/ \si[per-mode=reciprocal]{{\per\square\centi\metre \per\second}}'

# divider = make_axes_locatable(axs[-1])
# cax = divider.append_axes("bottom", "5%", pad="5%")
# fig.colorbar(im, ax=cax, orientation='horizontal')


# plt.draw()

plt.tight_layout(pad=0)
plt.subplots_adjust(left=0.000, right=1, bottom=0.18)
p = axs[-1].get_position().get_points().flatten()
print(p)
height = 0.04
offset = 0.001
ax_cbar = fig.add_axes([p[0], p[1] - 0.05, p[2] - p[0], height])
cb = fig.colorbar(im, cax=ax_cbar, orientation='horizontal', label=label)

# ticks = cb.get_ticks()
# ticklabs = cb.ax.get_xticklabels()
# from IPython import embed; embed()
# print(ticks, [t for t in ticklabs])
# ticks[0] += 0.0035**2
# ticks[-1] -= 0.1**2
# ticks = ticks[::2]
# ticklabs = ticklabs[::2]
# print(ticks, [t for t in ticklabs])
# from IPython import embed; embed()
# cb.ax.set_xticklabels(ticklabs)

cb.set_ticks(np.linspace(0, 2, 5))
ticks = cb.get_ticks()
ticks[-1] -= 0.1**2
ticks[0] += 0.003**2
cb.set_ticks(ticks)
ticklabs = cb.ax.get_xticklabels()
print([t for t in ticklabs])
cb.ax.set_xticklabels([f'${t:.1f}$' for t in  np.linspace(0, 2, 5)])

cb.ax.get_xticklabels()[0].set_ha('left')
cb.ax.get_xticklabels()[-1].set_ha('right')
# cb.ax.tick_params(direction='out')

# print(p)


# import matplotlib
# # You input the POSITION AND DIMENSIONS RELATIVE TO THE AXES
# x0, y0, width, height = [0, -0.1, 1, 0.1]

# # and transform them after to get the ABSOLUTE POSITION AND DIMENSIONS
# Bbox = matplotlib.transforms.Bbox.from_bounds(x0, y0, width, height)
# trans = axs[3].transAxes + fig.transFigure.inverted()
# l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds

# # Now just create the axes and the colorbar
# cbaxes = fig.add_axes([l, b, w, h])
# cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
# cbar.ax.tick_params(labelsize=9)
# fig.subplots_adjust(bottom=0.2)

with open("build/hgps_lon_range.txt", "w") as text_file:
    min_lon, max_lon = x0, x0 + width + max(offsets)
    max_lon = max_lon % 360
    result = f'\\SIrange{{{min_lon}}}{{{max_lon}}}{{\degree}}'
    text_file.write(result)


with open("build/hgps_lat_range.txt", "w") as text_file:
    result = f'\\SIrange{{{lat_range[0]}}}{{{lat_range[1]}}}{{\degree}}'
    text_file.write(result)

plt.savefig('build/hgps.pdf') # bbox_inches=Bbox.from_extents(0.06, 1, 7, 3.5)  # left, bottom, right and top.
