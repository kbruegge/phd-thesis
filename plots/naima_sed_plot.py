import h5py
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from data_utils import read_crab_mwl_data
from latex_utils import latex_table
from meyer_model import ssc_model_lut
import click

plot_dict = {
    'spi': ('INTEGRAL', 'integral-crab-data', 'o'),
    'magic': ('MAGIC', 'magic-crab-data', 'v'),
    'fermi_33months': ('FERMI LAT', 'fermi-crab-data', 'h'),
    'hess': ('H.E.S.S', 'hess-crab-data', 's'),
    'hegra': ('HEGRA', 'hegra-crab-data', 'D'),
    'comptel': ('Comptel', 'comptel-crab-data', 'P'),
}


def asym_number(data):
    l, m, u = np.percentile(data, q=[5, 50, 95]).T
    l = l - m
    u = u - m
    s = f'$\\num{{{m:.2f}}}\substack{{ {u:.2f} \\\ {l:.2f} }}$'
    return s

@click.command()
@click.option('-s', '--start', default=0)
def main(start):
    f = h5py.File('./data/naima/crab_chain.h5', mode='r')
    chain = f['sampler/chain'][()]

    short_chain = chain[:, start:, :]
    param_names = ['$N$', '$\\alpha$', '$\\beta$', '$\log_{10}\\left(\\nicefrac{E_\\text{max}}{\\si{TeV}} \\right)$', '$\log_{10}\\left(\\nicefrac{E_\\text{min}}{\\si{TeV}} \\right)$', '$\\nicefrac{B}{\\si{\micro G}}$']
  
    # plot the sed
    print('plotting sed')
    data = read_crab_mwl_data(component='nebula', e_min=10 * u.keV)
    energy = np.logspace(-8, 3, 1000) * u.TeV
    ssc_model = ssc_model_lut(path='./data/naima/lut.fits')

    plt.figure()

    samples = short_chain.reshape((-1, 6))
    for i in np.random.randint(len(samples), size=100):
        params = samples[i]
        # print(params)
        plt.plot(energy, energy**2 * ssc_model(params, energy,), color='gray', alpha=0.2, lw=1)

    params = np.median(short_chain, axis=[0, 1])
    # print(param_names)
    print(params)

    groups = data.group_by('paper').groups
    for group, paper in zip(groups, groups.keys):
        paper = paper[0]
        label, reference, marker = plot_dict[paper]
        yerr = [group['energy']**2 * group['flux_error_lo'], group['energy']**2 * group['flux_error_hi']]
        plt.errorbar(x=group['energy'], y=group['energy']**2 * group['flux'], yerr=yerr, linestyle='', marker=marker, label=f'{label} \\cite{{{reference}}}', lw=1, zorder=20)

    plt.plot(energy, energy**2 * ssc_model(params, energy), color='black', lw=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([2E-13, 2E-8])
    plt.xlabel('Energy / \si{TeV}')
    plt.ylabel('$\mathrm{E}^2 \\frac{\mathrm{dN}}{\mathrm{dE}}$ / \si{\TeV\per\square\centi\meter \per\second}')
    plt.legend(loc='lower left')
    plt.tight_layout(pad=0)
    plt.savefig('./build/ssc_fit.pgf')

    print('Creating result table')
    with open("build/naima_result_table.txt", "w") as text_file:
        rows = [(name, asym_number(r)) for (name, r) in zip(param_names, short_chain.T)]
        table = latex_table(rows, column_names=['Parameter', 'Fit Result'])
        text_file.write(table)


if __name__ == "__main__":
    main()