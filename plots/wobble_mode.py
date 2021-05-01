import matplotlib.pyplot as plt

import numpy as np
from gammapy.maps import Map
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.coordinates import SkyCoord
from scipy.stats import multivariate_normal
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


def setfont(font, text):
    s = f'{{\\fontspec{{{font}}}{text}}}'
    return s

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])



path = './plots/data/hgps_map_flux_0.1deg_v1.fits.gz'
survey_map = Map.read(path)

# factor = 1E12
# survey_map.data *= factor

lw = 1
alpha = 0.7


shape = (2000, 2000)



psr = SkyCoord(ra='133.60075 deg', dec='-46.23705556 deg')
psr_cutout = survey_map.cutout(psr, width='3 deg')
print(psr_cutout.data.shape)
im = psr_cutout.data * 1E14

h = 2.0 * np.random.poisson(10, size=shape)
x_offset = h.shape[0] // 2 - im.shape[0] // 2, h.shape[0] // 2 + im.shape[0] // 2
y_offset = h.shape[1] // 2 - im.shape[1] // 2, h.shape[1] // 2 + im.shape[1] // 2
h[x_offset[0]:x_offset[1], y_offset[0]:y_offset[1]] += 0.1 * im**2


psr = SkyCoord(l='17.5 deg', b='0 deg', frame='galactic')
psr_cutout = survey_map.cutout(psr, width='3 deg')
print(psr_cutout.data.shape)
im = psr_cutout.data * 1E14

x_offset = np.array([h.shape[0] // 2 - im.shape[0] // 2, h.shape[0] // 2 + im.shape[0] // 2])
x_offset -= 380

y_offset = np.array([h.shape[1] // 2 - im.shape[1] // 2, h.shape[1] // 2 + im.shape[1] // 2])
y_offset += 270
h[x_offset[0]:x_offset[1], y_offset[0]:y_offset[1]] += 0.02 * im**2


psr = SkyCoord(l='335 deg', b='0 deg', frame='galactic')
psr_cutout = survey_map.cutout(psr, width=['10 deg', '3 deg'])
print(psr_cutout.data.shape)
im = psr_cutout.data * 1E14
im = np.flip(im)
x_offset = np.array([h.shape[0] // 2 - im.shape[0] // 2, h.shape[0] // 2 + im.shape[0] // 2])
x_offset += 550

y_offset = np.array([h.shape[1] // 2 - im.shape[1] // 2, h.shape[1] // 2 + im.shape[1] // 2])
y_offset -= 50
h[x_offset[0]:x_offset[1], y_offset[0]:y_offset[1]] += 0.019 * im**2


psr = SkyCoord(l='347 deg', b='0 deg', frame='galactic')
psr_cutout = survey_map.cutout(psr, width=['5 deg', '3 deg'])
print(psr_cutout.data.shape)
im = psr_cutout.data * 1E14
im = np.flip(im)
x_offset = np.array([h.shape[0] // 2 - im.shape[0] // 2, h.shape[0] // 2 + im.shape[0] // 2])
x_offset -= 700

y_offset = np.array([h.shape[1] // 2 - im.shape[1] // 2, h.shape[1] // 2 + im.shape[1] // 2])
y_offset -= 270
h[x_offset[0]:x_offset[1], y_offset[0]:y_offset[1]] += 0.019 * im**2




XS, YS = np.meshgrid(range(shape[0]), range(shape[0]))
coords = np.array([XS.ravel(), YS.ravel()])  
n = multivariate_normal.pdf(x=coords.T, mean=[1000, 1000], cov=[[1000000, 0], [0, 1000000]])
n = n.reshape(shape)
n = n / n.max()

h *= n
size = plt.gcf().get_size_inches()
fig, ax = plt.subplots(1, 1, figsize=(size[0], 2.85))
im = ax.imshow(h.reshape(shape).T, norm=PowerNorm(0.55), cmap='magma')
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylim(380, 1550)
# ax.set_xlim(200, 1800)


r = 300
p1 = [1000 + r, 1000]
for i, phi in enumerate(np.arange(90, 360, 30)):
    if i in [5, 4, 3]:
        continue
    phi -= 120 + 90
    coords = pol2cart(r, np.deg2rad(phi)) + p1
    c = Circle(xy=coords, radius=60, fill=False, lw=lw, alpha=alpha)
    patch = ax.add_artist(c)
    # ax.text(coords[0], coords[1], s=i)

p2 = [1000 - r, 1000]
for i, phi in enumerate(np.arange(90, 360, 30)):
    if i == 2:
        continue
    phi -= 120 + 180 + 90
    coords = pol2cart(r, np.deg2rad(phi)) + p2
    c = Circle(xy=coords, radius=60, fill=False, lw=lw, alpha=alpha)
    ax.add_artist(c, )
    # ax.text(coords[0], coords[1], s=i)

ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], marker='P', s=12, color='white', label='Pointing Position')


legend_elements = [
    Line2D([0], [0], color='white', marker='P', label='Pointing Position', ms=6, ls=''), 
    Line2D([0], [0], marker='o', markerfacecolor='none', color='white', label='Off Region', markersize=8, ls='', lw=0.9, alpha=0.8),
]
ax.legend(handles=legend_elements, loc='lower left',  ncol=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=.05)

cb = fig.colorbar(im, label='Counts', cax=cax)
cb.ax.get_yticklabels()[0].set_va('bottom')

plt.tight_layout(pad=0)
plt.subplots_adjust(left=0.0, right=0.913, bottom=-0.07, top=1.05)
# r = np.random.multivariate_normal(mean=[0, 0], cov=[[20, 0], [0, 20]], size=1000000)
# h, _, _ = np.histogram2d(r[:, 0], r[:, 1], bins=shape, density=True)

# # im += (np.random.poisson(10, size=im.shape,) + np.random.normal(loc=1, scale=2, size=im.shape))
# # print(x.mean())
# # im[50:150, 50:150] += x
# # h = (h + 10)**2
# plt.imshow(h, interpolation='nearest')
# plt.colorbar(label='Counts')
# # from IPython import embed; embed()
# cmap = 'YlGnBu_r'
# stretch = 'sqrt'
# vmax = 2

# lat_range = [-2.5, 2.3]
# offsets = [0, 35, 70, 105]
# axs = axs.ravel()
# for ax, offset in zip(axs, offsets):
#     survey_map.plot(ax=ax, fig=fig, stretch=stretch, cmap=cmap, vmin=0.0, vmax=vmax)
#     x_cutout = np.array([x0, x0 + width]) + offset
#     xlim_pix, ylim_pix = survey_map.geom.wcs.wcs_world2pix(x_cutout, lat_range, 1)

#     ax.set_xlim(xlim_pix)
#     ax.set_ylim(ylim_pix)
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.grid(False)


# arrowprops = dict(facecolor='white', arrowstyle='-')
# # position of sagittarius a in galactic coordinates
# sgr_a = (359.94423568, -0.04616002)
# vela_junior = (266.28, -1.24)

# xy = np.array(survey_map.geom.wcs.wcs_world2pix(vela_junior[0], vela_junior[1], 1))
# axs[0].annotate(xy=xy, s=setfont('Minion Pro', 'RX J0852'), fontsize=10, color='white', arrowprops=arrowprops, xytext=xy + [-250, 10])

# xy = np.array(survey_map.geom.wcs.wcs_world2pix(sgr_a[0], sgr_a[1], 1))
# axs[2].annotate(xy=xy, s=setfont('Minion Pro', 'Sagittarius A'), fontsize=10, color='white', arrowprops=arrowprops, xytext=xy + [500, 30])


# im = axs[2].get_images()[0]
# # factor = 1E-12
# label = f'Flux \\num{{E-12}}/ \si[per-mode=reciprocal]{{\per\square\centi\metre \per\second}}'

# # divider = make_axes_locatable(axs[-1])
# # cax = divider.append_axes("bottom", "5%", pad="5%")
# # fig.colorbar(im, ax=cax, orientation='horizontal')


# # plt.draw()

# plt.tight_layout(pad=0)
# plt.subplots_adjust(left=0.000, right=1, bottom=0.18)
# p = axs[-1].get_position().get_points().flatten()
# print(p)
# height = 0.04
# offset = 0.001
# ax_cbar = fig.add_axes([p[0], p[1] - 0.05, p[2] - p[0], height])
# cb = fig.colorbar(im, cax=ax_cbar, orientation='horizontal', label=label)

# # ticks = cb.get_ticks()
# # ticklabs = cb.ax.get_xticklabels()
# # from IPython import embed; embed()
# # print(ticks, [t for t in ticklabs])
# # ticks[0] += 0.0035**2
# # ticks[-1] -= 0.1**2
# # ticks = ticks[::2]
# # ticklabs = ticklabs[::2]
# # print(ticks, [t for t in ticklabs])
# # from IPython import embed; embed()
# # cb.ax.set_xticklabels(ticklabs)

# cb.set_ticks(np.linspace(0, 2, 5))
# ticks = cb.get_ticks()
# ticks[-1] -= 0.1**2
# ticks[0] += 0.003**2
# cb.set_ticks(ticks)
# ticklabs = cb.ax.get_xticklabels()
# print([t for t in ticklabs])
# cb.ax.set_xticklabels([f'${t:.1f}$' for t in  np.linspace(0, 2, 5)])

# cb.ax.get_xticklabels()[0].set_ha('left')
# cb.ax.get_xticklabels()[-1].set_ha('right')
# # cb.ax.tick_params(direction='out')

# # print(p)


# # import matplotlib
# # # You input the POSITION AND DIMENSIONS RELATIVE TO THE AXES
# # x0, y0, width, height = [0, -0.1, 1, 0.1]

# # # and transform them after to get the ABSOLUTE POSITION AND DIMENSIONS
# # Bbox = matplotlib.transforms.Bbox.from_bounds(x0, y0, width, height)
# # trans = axs[3].transAxes + fig.transFigure.inverted()
# # l, b, w, h = matplotlib.transforms.TransformedBbox(Bbox, trans).bounds

# # # Now just create the axes and the colorbar
# # cbaxes = fig.add_axes([l, b, w, h])
# # cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
# # cbar.ax.tick_params(labelsize=9)
# # fig.subplots_adjust(bottom=0.2)

# with open("build/hgps_lon_range.txt", "w") as text_file:
#     min_lon, max_lon = x0, x0 + width + max(offsets)
#     max_lon = max_lon % 360
#     result = f'\\SIrange{{{min_lon}}}{{{max_lon}}}{{\degree}}'
#     text_file.write(result)


# with open("build/hgps_lat_range.txt", "w") as text_file:
#     result = f'\\SIrange{{{lat_range[0]}}}{{{lat_range[1]}}}{{\degree}}'
#     text_file.write(result)

plt.savefig('build/wobble_mode.pdf') # bbox_inches=Bbox.from_extents(0.06, 1, 7, 3.5)  # left, bottom, right and top.
