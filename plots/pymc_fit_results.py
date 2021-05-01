import matplotlib.pyplot as plt
import numpy as np
# from gammapy.spectrum import CrabSpectrum
from meyer_model import ssc_model_lut, Log10Parabola
import h5py
import astropy.units as u
import pymc3 as pm
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm
from latex_utils import asym_number

from ruamel.yaml import YAML
yaml = YAML(typ='safe')


def load_config(telescope, config_file):
    with open(config_file) as f:
        d = yaml.load(f)
        tel_config = d['datasets'][telescope]
        d = {
            'telescope': telescope,
            'on_radius': tel_config['on_radius'] * u.deg,
            'containment_correction': tel_config['containment_correction'],
            'stack': tel_config.get('stack', False),
            'fit_range': tel_config['fit_range'] * u.TeV,
        }

        return d


    
def plot_spectra(trace, fit_range=[0.03, 30] * u.TeV, min_sample=50, ax=None, label=None, percentiles=[5, 50, 95]):
    if not ax:
        ax = plt.gca()

    energy = np.logspace(*np.log10(fit_range.to_value('TeV')), 200) * u.TeV

    model_values = []
    N = len(trace.get_values('alpha'))
    ids = np.random.choice(N, size=min(5000, N))
    for index in tqdm(ids):
        alpha = trace.get_values('alpha')[index] * u.Unit('')
        beta = trace.get_values('beta')[index] * u.Unit('')
        amplitude = trace.get_values('amplitude')[index] * 1e-11 * u.Unit('cm-2 s-1 TeV-1')
        reference = 1 * u.TeV
        y = Log10Parabola.evaluate(energy, alpha=alpha, amplitude=amplitude, beta=beta, reference=reference)
        model_values.append(y)
    
    model_values = np.array(model_values)
    lower, median, upper = np.percentile(model_values, axis=0, q=percentiles)
    line, = ax.plot(energy, energy**2 * median, label=label)
    
    color = line.get_color()
    ax.fill_between(energy, energy**2 * lower, energy**2 * upper, color=color, alpha=0.5)
    # fitted_model.plot(energy_range=fit_range, energy_power=2, ax=ax, label=label)


def write_parameter_values(trace, telescope):
    with open(f'./build/pymc_results/fit/{telescope}/alpha.txt', 'w') as out:
        out.write(asym_number(trace.get_values('alpha')))
    with open(f'./build/pymc_results/fit/{telescope}/amplitude.txt', 'w') as out:
        out.write(asym_number(trace.get_values('amplitude')))
    with open(f'./build/pymc_results/fit/{telescope}/beta.txt', 'w') as out:
        out.write(asym_number(trace.get_values('beta')))



def plot_naima_model(path='./data/naima/crab_chain.h5', lut_path='./data/naima/lut.fits', ax=None, color='black', label='SSC Model'):
    f = h5py.File(path, mode='r')
    chain = f['sampler/chain'][()]

    start = 0
    short_chain = chain[:, start:, :]
    ssc_model = ssc_model_lut(path=lut_path)

    params = np.median(short_chain, axis=[0, 1])

    if not ax:
        ax = plt.gca()

    energy = np.logspace(-2.4, 2.2, 1000) * u.TeV
    y = energy**2 * ssc_model(params, energy)

    # ax.plot(energy, y, label=label, lw=0.5, color='orange')
    ax.plot(energy, gaussian_filter1d(y, sigma=12, mode='nearest'), color=color, label=label, ls='--')

# Plot pymc fit spectral fit results together with the fitted SED spectrum from the naima model
# build a dummy model. this is weird. i know. wtf pymc?
model = pm.Model(theano_config={'compute_test_value': 'ignore'})
with model:
    amplitude = pm.TruncatedNormal('amplitude', mu=4, sd=1, lower=0.05, testval=0.21)
    alpha = pm.TruncatedNormal('alpha', mu=2.5, sd=1, lower=0.05, testval=0.21)
    beta = pm.TruncatedNormal('beta', mu=0.5, sd=0.5, lower=0.001, testval=0.1)

    mu_s = pm.Deterministic('mu_s', alpha + beta)  # dummy function to load traces.

    mu_b = pm.TruncatedNormal('mu_b', lower=0, shape=2, mu=[1, 2], sd=5)

    # model has to be on stack to load traces. why? who the f knows
    telescopes = ['magic', 'fact', 'veritas', 'hess']
    for tel in telescopes:
        config = load_config(tel, config_file='./configs/pymc/data_conf.yaml')
        trace = pm.load_trace(f'./build/pymc_results/fit/{tel}/traces')
        plot_spectra(trace, label=f'\\{tel}', fit_range=config['fit_range'], percentiles=[16, 50, 84])
        write_parameter_values(trace, tel)

    plot_naima_model(label='SSC model from chapter~\\ref{ch:crab-sed}')
    plt.xlabel('Energy / \si{TeV}')
    plt.ylabel('$\\text{E}^2 \\frac{\\text{dN}}{\\text{dE}}$ / \si{\TeV\per\square\centi\meter \per\second}')
    plt.xlim([10**(-2.1), 10**(2.1)])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout(pad=0, rect=(0, 0, 1.005, 0.985))
    plt.savefig('build/pymc_results/pymc_fit_result.pgf')

