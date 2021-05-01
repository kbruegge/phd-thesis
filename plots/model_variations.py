import matplotlib as mpl
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from data_utils import read_crab_mwl_data
from meyer_model import ssc_model_components
from mpl_toolkits.axes_grid1 import make_axes_locatable
import click

CRAB_DATA = read_crab_mwl_data(component='nebula', e_min=10 * u.keV)


def init_colors(values, cmap_name='viridis'):
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    cmap.set_array([])
    return cmap


def plot_data(data, ax):
    yerr = [data['energy']**2 * data['flux_error_lo'], data['energy']**2 * data['flux_error_hi']]
    ax.errorbar(x=data['energy'], y=data['energy']**2 * data['flux'], yerr=yerr, linestyle=None, fmt='o', color='gray')


def plot_spectrum(ax, parameters, energy, color):
    SYN, IC = ssc_model_components(parameters, precision=35)
    f = IC.flux(energy, distance=2 * u.kpc)
    f = f.to_value('cm-2 TeV-1 s-1')
    ax.plot(energy, energy**2 * f, color=color, lw=1, alpha=0.6)

    sf = SYN.flux(energy, distance=2 * u.kpc)
    sf = sf.to_value('cm-2 TeV-1 s-1')
    ax.plot(energy, energy**2 * sf, color=color, lw=1, alpha=0.6)

    ax.plot(energy, energy**2 * (sf + f), lw=1.5, color=color)
    


def plot_variations(param_index, param_set, cbar_label, ax, cmap_name='viridis'):
    # set 'best' parameters as fixed for the plots
    log_e_min = -7.8
    log_e_max = 2.9
    energy = np.logspace(log_e_min, log_e_max, 300) * u.TeV

    params = [47.5, 2.9, 0.055, 15.5, 11, 94]
    cmap = init_colors(param_set, cmap_name=cmap_name)
    for p in param_set:
        params[param_index] = p
        color = cmap.to_rgba(p)
        plot_spectrum(ax, params, energy, color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([2E-13, 5E-8])
        ax.set_xlim([10**log_e_min, 10**log_e_max])
        ax.set_xticks([])
        ax.set_yticks([])

    ax.plot(CRAB_DATA['energy'], CRAB_DATA['energy']**2 * CRAB_DATA['flux'], 'o', color='gray', ms=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    plt.gcf().colorbar(cmap, cax=cax, ticks=param_set[1:-1][::2], label=cbar_label, orientation='horizontal')
    # ax.set_yticks([])
    # ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


@click.command()
@click.option('-c', '--cmap', default='viridis')
def main(cmap):
    size = plt.gcf().get_size_inches()
    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(size[0], 6.53))

    ax = axs.ravel()[0]
    betas = np.arange(0.02, 0.12, 0.015)
    plot_variations(2, betas, '$\\beta$', ax, cmap_name=cmap)

    ax = axs.ravel()[1]
    alphas = np.arange(2.4, 3.4, 0.1)
    plot_variations(1, alphas, '$\\alpha$', ax, cmap_name=cmap)

    ax = axs.ravel()[2]
    e_min = np.arange(10.5, 11.5, 0.2)
    label = '$\\log_{10}\\left(E_{\\text{min}} /\\, \\text{TeV}\\right)$'
    plot_variations(4, e_min, label, ax, cmap_name=cmap)

    ax = axs.ravel()[3]
    e_max = np.arange(15, 16, 0.1)
    label = '$\\log_{10}\\left(E_{\\text{max}} /\\, \\text{TeV}\\right)$'
    plot_variations(3, e_max, label, ax, cmap_name=cmap)

    ax = axs.ravel()[4]
    e_max = np.arange(46.8, 48.5, 0.25)
    plot_variations(0, e_max, '$N$', ax, cmap_name=cmap)


    ax = axs.ravel()[5]
    field_strength = np.arange(40, 250, 25)
    plot_variations(5, field_strength, 'B / $\mu G$', ax, cmap_name=cmap)




    plt.tight_layout(pad=0)
    params = dict(bottom=0.07, left=0, right=1, hspace=0.3)
    fig.subplots_adjust(**params, )
    # from matplotlib.transforms import Bbox
    # bbox_inches=Bbox.from_extents(0.0, 5.35, 5.53, 0)  # left, bottom, right and top.
    plt.savefig('./build/model_variations.pdf')


if __name__ == "__main__":
    main()