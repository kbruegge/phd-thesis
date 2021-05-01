import h5py
import numpy as np
import matplotlib.pyplot as plt
import click
# from matplotlib.gridspec import GridSpec
from itertools import product
from tqdm import tqdm


main_color = 'crimson'


def asym_number(data, name, precision=2):
    l, m, u = np.percentile(data, q=[16, 50, 84]).T
    l = l - m
    u = u - m
    s = f'{name} $= \\num{{{m:.{precision}f}}}\substack{{ {u:+.{precision}f} \\\\ {l:+.{precision}f} }}$'
    return s


def asym_number_raw(data, precision=2):
    l, m, u = np.percentile(data, q=[16, 50, 84]).T
    l = l - m
    u = u - m
    s = f'\\num{{{m:.{precision}f}}}\substack{{ {u:+.{precision}f} \\\\ {l:+.{precision}f} }}'
    return s


def walker_plot(ax, chains, color='crimson'):
    ax.plot(chains.T, color='gray', alpha=0.1, lw=0.1, rasterized=True)
    m = np.median(chains, axis=0)
    ax.plot(m, lw=1, color=color)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([0, len(m) // 2, len(m)])
    ax.set_xticklabels([0, len(m) // 2, len(m)])
    ax.set_xlim([0, len(m)])
    ax.get_xticklabels()[0].set_ha('left')
    ax.get_xticklabels()[2].set_ha('right')



def density_plot(ax, chains, color='crimson', precision=2):
    data = chains.ravel()
    xlim_l, lower, median, uppper, xlim_u = np.percentile(data, q=[0.01, 16, 50, 84, 99.99])
    ax.hist(data, bins=100, density=True, histtype='stepfilled', alpha=0.3, color='gray')
    ax.hist(data, bins=100, density=True, histtype='step', lw=1, color=color)
    ax.axvspan(lower, uppper, color='gray', alpha=0.3)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([xlim_l, median, xlim_u])
    l = [f'{l:.{precision}f}' for l in [xlim_l, median, xlim_u]]
    ax.set_xticklabels(l)
    ax.set_xlim([xlim_l, xlim_u])
    ax.get_xticklabels()[0].set_ha('left')
    ax.get_xticklabels()[2].set_ha('right')


@click.command()
@click.option('-s', '--start', default=0)
def main(start):
    f = h5py.File('./data/naima/crab_chain.h5', mode='r')
    chain = f['sampler/chain'][()]

    param_names = ['$\log_{10}\\left(\\frac{N}{\\si{erg}}\\right)$', '$\\alpha$', '$\\beta$', '$\log_{10}\\left(\\frac{E_\\text{max}}{\\si{TeV}} \\right)$', '$\log_{10}\\left(\\frac{E_\\text{min}}{\\si{TeV}} \\right)$', '$\\frac{B}{\\si{\micro G}}$']


    print('plotting chains')
    figsize = (1.6, 1.1)
    for i, p in tqdm(enumerate(param_names), total=6):
        precision = 2 
        if p == '$\\beta$':
            precision = 3
        if p == '$\\frac{B}{\\si{\micro G}}$':
            precision = 1
        
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        walker_plot(ax, chain[:, :, i], color=main_color)
        # ax.set_ylabel(p)
        plt.tight_layout(pad=0)
        plt.savefig(f'./build/naima_results/chain_{i}.pdf', dpi=300)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        density_plot(ax, chain[:, :, i], precision=precision, color=main_color)
        # ax.set_xlabel(p)
        plt.tight_layout(pad=0)
        plt.subplots_adjust(right=0.995)
        plt.savefig(f'./build/naima_results/density_{i}.pdf', dpi=300)

        with open(f"build/naima_results/param_{i}.txt", "w") as text_file:
            result = asym_number(chain[:, :, i].ravel(), p, precision=precision)
            text_file.write(result)

        with open(f"build/naima_results/param_{i}_raw.txt", "w") as text_file:
            result = asym_number_raw(chain[:, :, i].ravel(), precision=precision)
            text_file.write(result)

        with open(f"build/naima_results/name_{i}.txt", "w") as text_file:
            text_file.write(f'{p}')


    with open(f"build/naima_results/num_samples.txt", "w") as text_file:
        text_file.write(f'\\num{{{chain.shape[0] * chain.shape[1]}}}')

    with open(f"build/naima_results/num_chains.txt", "w") as text_file:
        text_file.write(f'\\num{{{chain.shape[0]}}}')

    n_par = chain.shape[2]
    flat_chain = chain.reshape(-1, n_par)
    plt.figure()
    size = plt.gcf().get_size_inches()
    fig, axs = plt.subplots(n_par, n_par, figsize=(size[0], 6), dpi=2000)

    for ax, (i, j) in zip(axs.T.ravel(), product(range(n_par), range(n_par))):
    #     print(ax, i, j )

        if i > j:
            ax.set_visible(False)
            continue
    #     ax.hist2d(flat_chain[:, i], flat_chain[:, j], cmap='gray_r', vmin=2, bins=30,)
        if i == j:
            ax.hist(flat_chain[:, i], bins=150, density=True, histtype='stepfilled', alpha=0.3, color='gray')
            ax.hist(flat_chain[:, i], bins=150, density=True, histtype='step', lw=1, color=main_color)
        else:
            ax.scatter(flat_chain[::40, i], flat_chain[::40, j], color='k', s=1, alpha=0.01, rasterized=True, marker='.')    
            xm, ym = np.median(flat_chain[:, i]), np.median(flat_chain[:, j])
            ax.scatter(xm, ym, s=3, c=main_color)

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
    
        xlim_l, median, xlim_u = np.percentile(flat_chain[:, i], q=[0.01, 50, 99.99])
        ax.set_xticks([xlim_l, median, xlim_u])

        precision = 3
        if median > 1:
            precision = 2
        if median > 100:
            precision = 0
        l = [f'{l:.{precision}f}' for l in [xlim_l, median, xlim_u]]
        ax.set_xticklabels(l, rotation=90)

        if i != j:
            ylim_l, median, ylim_u = np.percentile(flat_chain[:, j], q=[1, 50, 99])
            ax.set_yticks([ylim_l, median, ylim_u])
            l = [f'{l:.{precision}f}' for l in [ylim_l, median, ylim_u]]
            ax.set_yticklabels(l)
        # ax.get_xticklabels()[0].set_ha('left')
        # ax.get_xticklabels()[2].set_ha('right')
        
        if (i == 0) and (j == 0):
            ax.set_yticks([])
            ax.set_ylabel('')
        if j < (n_par - 1):
            ax.set_xticks([])
        else:
            ax.set_xlabel(param_names[i], rotation=90, fontsize=7)
        if i > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel(param_names[j], va='center', ha='right', rotation=0, fontsize=7)
        
        if (i == 0) and (j == 0):
            ax.set_yticks([])
            ax.set_ylabel('')

        ax.grid(False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.2, wspace=0.025, hspace=0.025, bottom=0.19, right=1.01)
    plt.savefig('build/naima_corner.pdf')


if __name__ == "__main__":
    main()