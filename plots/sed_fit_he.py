from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
import pandas as pd
from latex_utils import matrix_to_latex
from data_utils import read_crab_mwl_data



def logpar(E, N, alpha, beta):
    return N * (E / u.TeV) ** (-alpha - beta * np.log10(E / u.TeV))


def fit_logpar(df):
    energy, flux, _, _ = (df.energy, df.flux, df.flux_error_lo, df.flux_error_hi)

    def func(e, N, a, b):
        return e ** 2 * logpar(e * u.TeV, N, a, b)

    p0 = [5e-11, 2.7, 0.3]
    #     sigma = (d_flux_high - d_flux_low)/flux
    r, cov = curve_fit(func, energy, energy ** 2 * flux, p0=p0)
    return r, cov


def error_band(mu, cov, energy, N=10000):
    samples = np.random.multivariate_normal(mu, cov, size=N) 
    fluxes = [logpar(energy, *s).value for s in samples]
    fluxes = np.vstack(fluxes)
    lower, median, upper = np.percentile(fluxes, q=[5, 50, 95], axis=0)
    return lower, median, upper


plot_dict = {
    'magic': ('MAGIC', 'magic-crab-data', 'v'),
    'fermi_33months': ('FERMI LAT', 'fermi-crab-data', 'h'),
    'hess': ('H.E.S.S', 'hess-crab-data', 's'),
    'hegra': ('HEGRA', 'hegra-crab-data', 'D'),
}


def main():
    df_magic = read_magic_data()
    df_hess = read_crab_mwl_data(paper="hess").to_pandas()
    df_fermi = read_crab_mwl_data(paper="fermi_33months").to_pandas()
    df_hegra = read_crab_mwl_data(paper="hegra").to_pandas()

    # size = plt.gcf().get_size_inches()
    # fig = plt.figure(figsize=(size[0], size[1]))

    label, reference, marker = plot_dict['magic']
    plot_flux_points(df_magic, power=2, label=f"{label} \\cite{{{reference}}}", marker=marker)
    label, reference, marker = plot_dict['hess']
    plot_flux_points(df_hess, power=2, label=f"{label} \\cite{{{reference}}}", marker=marker)
    label, reference, marker = plot_dict['fermi_33months']
    plot_flux_points(df_fermi, power=2, label=f"{label} \\cite{{{reference}}}", marker=marker)
    label, reference, marker = plot_dict['hegra']
    plot_flux_points(df_hegra, power=2, label=f"{label} \\cite{{{reference}}}", marker=marker)

    # dont fit first few fermi points.
    df_fermi_he = df_fermi.loc[12:]
    df = pd.concat([df_fermi_he, df_magic, df_hess, df_hegra], sort=False)
    r, cov = fit_logpar(df)

    amp, alpha, beta = r
    d_amp, d_alpha, d_beta = np.sqrt(cov.diagonal())


    ex = np.logspace(-3.0, 2, 100) * u.TeV
    plt.plot(ex, ex ** 2 * logpar(ex, *r), color="k", lw=2.5, label='LSQR Fit')
    
    lower, median, upper = error_band(r, cov, ex)

    color = (.7, .7, .7, 1)
    plt.fill_between(ex, ex ** 2 * lower, ex ** 2 * upper, facecolor=color, lw=1, edgecolor=color,)
    
    plt.xlabel('Energy / \si{TeV}')
    plt.ylabel('$\\mathrm{E}^2 \\frac{\\mathrm{dN}}{\\mathrm{dE}}$ / \si{\TeV\per\square\centi\meter \per\second}')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout(pad=0)
    
    plt.savefig("build/sed_fit_he.pgf")

    with open("build/sed_fit_he.txt", "w") as text_file:
        amp *= 1E11
        d_amp *= 1E11
        result = f'$A = ({amp:.2f} \\pm {d_amp:.2f}) \cdot 10^{{-11}} \si{{\TeV\per\square\centi\meter \per\second}}$, $\\alpha = {alpha:.2f} \\pm {d_alpha:.2f}$ and $\\beta={beta:.2f} \\pm {d_beta:.2f}$'
        text_file.write(result)

    with open("build/sed_fit_he_matrix.txt", "w") as text_file:
        result = bordermatrix(cov, precision=3, force_scientific=True, titles=['A', '\\alpha', '\\beta'])
        print(result)
        text_file.write(result)


def bordermatrix(m, kind='pmatrix', precision=None, force_scientific=False, titles=None):
    '''
    convert numpy array to latex matrx to be used with siunitx
    '''
    if precision:
        num_options = f'round-mode=places,round-precision={precision}'
    else:
        num_options = ''

    if force_scientific:
        num_options += ',scientific-notation=true'
    
    m_string = '\\bordermatrix{ ~ ' + ''.join([f'& {t}' for t in titles]) + '  \\cr  '

    for row, t in zip(m, titles):
        number_strings = [f'\\num[{{{num_options}}}]{{{a}}}' for a in row]
        row_string = ' & '.join(number_strings)
        row_string = f'{t} & ' + row_string + '\\cr'
        m_string += row_string

    m_string += ' }'
    return m_string


def read_magic_data():
    t = Table.read("plots/data/MAGIC_2015_Crab_Nebula.fits", hdu=2)
    energy = t["energy"].quantity.to_value("TeV")
    flux = t["flux"]
    d_flux_low = t["Dflux"] / 2
    d_flux_high = t["Dflux"] / 2

    df = pd.DataFrame(
        {
            "energy": energy,
            "flux": flux,
            "flux_error_lo": d_flux_low,
            "flux_error_hi": d_flux_high,
        }
    )

    return df




def plot_flux_points(df, ax=None, power=None, label=None, marker='o'):
    if not ax:
        ax = plt.gca()

    factor = df['energy'] ** power if power else 1

    yerr = [factor * df['flux_error_lo'], factor * df['flux_error_hi']]
    ax.errorbar(df['energy'], factor * df['flux'], yerr=yerr, fmt='o', label=label, zorder=20, marker=marker)



if __name__ == "__main__":
    main()
