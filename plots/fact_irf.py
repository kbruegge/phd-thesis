from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker


def plot_aeff(path, ax, cmap='viridis', hdu=1, name=None):
    aeff = Table.read(path, hdu=hdu)
    values = aeff['EFFAREA'][0]*1E-6
    e_bins = list(aeff['ENERG_LO'][0]) + [aeff['ENERG_HI'][0][-1]]
    theta_bins = list(aeff['THETA_LO'][0]) + [aeff['THETA_HI'][0][-1]]
    # print(theta_bins)
    im = ax.pcolormesh(e_bins, theta_bins, values, cmap=cmap, vmin=0)

    ax.set_xscale('log')
    # ax.set_xlabel('Energy / \\si{TeV}')
    ax.set_ylabel('$\\theta / \\si{\degree}$')

    ticks = [0, values.max() / 2, values.max()]
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=ticker.StrMethodFormatter("{x:.2f}"), ticks=ticks)
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')

    ticklabs = cb.ax.get_xticklabels()
    ticklabs[0].set_ha('left')
    ticklabs[2].set_ha('right')

    if name:
        ax.text(
            0.12,
            0.9,
            name,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
        )
 

    return im, cb, aeff.meta['RAD_MAX'] if 'RAD_MAX' in aeff.meta else None


def plot_e_migra(path, ax, cmap='viridis', hdu=2, vmax=None, name=None):
    migra = Table.read(path, hdu=hdu)
    values = migra['MATRIX'][0][0]
    e_bins = list(migra['ENERG_LO'][0]) + [migra['ENERG_HI'][0][-1]]
    migra_bins = list(migra['MIGRA_LO'][0]) + [migra['MIGRA_HI'][0][-1]]
    theta_bins = list(migra['THETA_LO'][0]) + [migra['THETA_HI'][0][-1]]
    print(theta_bins)
    im = ax.pcolormesh(e_bins, migra_bins, values, cmap=cmap, vmax=vmax)

    if vmax:
        ticks = [0, vmax / 2, vmax]
    else:
        ticks = [0, values.max() / 2, values.max()]


    ax.set_xscale('log')
    ax.set_ylabel('$\mu$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=ticker.StrMethodFormatter("{x:.1f}"), ticks=ticks)
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_ticks_position('top')

    ticklabs = cb.ax.get_xticklabels()
    ticklabs[0].set_ha('left')
    ticklabs[2].set_ha('right')

    if name:
        ax.text(
            0.12,
            0.9,
            name,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
        )
 
    return im, cb


path_fact = 'plots/data/joint_crab/dl3/fact/fact_irf.fits'

size = plt.gcf().get_size_inches()
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(size[0], 4.7), sharex=True, )
cmap = 'magma_r'

im, cb, rad_max = plot_aeff(path_fact, ax1, cmap=cmap, name='\\fact')
cb.set_label('Effective Area / \\si{\square\kilo\metre}')

im, cb = plot_e_migra(path_fact, ax2, cmap=cmap, vmax=0.4, name='\\fact')
cb.set_label('Energy Migration $\mu$ ')

ax1.set_xlim([0.3, 50])

ax2.set_ylim([0.25, 3.5])
ax2.set_xlim([0.3, 50])

with open("build/fact_irf_selection_radius.txt", "w") as text_file:
    result = f'\SI{{{rad_max:.2f}}}{{\\degree}}'
    text_file.write(result)

# path_magic = 'plots/data/joint_crab/dl3/magic/run_05029747_DL3.fits'
path_magic = 'plots/data/joint_crab/dl3/hess/data/hess_dl3_dr1_obs_id_023523.fits.gz'
im, cb, _ = plot_aeff(path_magic, ax3, cmap=cmap, hdu=3, name='\\hess')
im, cb = plot_e_migra(path_magic, ax4, cmap=cmap, hdu=4, vmax=6, name='\\hess')


ax3.set_xlabel('Energy / \\si{TeV}')

ax4.set_ylim([0.25, 3.5])
ax4.set_xlabel('Energy / \\si{TeV}')


# with open("build/fact_irf_selection_radius.txt", "w") as text_file:
#     result = f'\SI{{{rad_max:.2f}}}{{\\degree}}'
#     text_file.write(result)


plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.25, wspace=0.2)
# fig.subplots_adjust(right=1)
plt.savefig('build/fact_irf.pdf')