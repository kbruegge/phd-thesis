import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import pymc3 as pm
from tqdm import tqdm
from meyer_model import Log10Parabola
import pandas as pd
from functools import reduce
from decimal import Decimal
from pymc_utils import load_config, dummy_model_unfold, dummy_model_fit
from plot_utils import LIST_OF_TEL_NAMES, LIST_OF_TELS


def plot_spectra(trace, fit_range=[0.03, 30] * u.TeV, min_sample=50, ax=None, label=None, color=None, percentiles=[5, 50, 95]):
    if not ax:
        ax = plt.gca()

    scale = 2

    energy = np.logspace(*np.log10(fit_range.to_value('TeV')), 200) * u.TeV
    model_values = []
    N = len(trace.get_values('alpha'))
    ids = np.random.choice(N, size=min(500, N))
    for index in tqdm(ids):
        alpha = trace.get_values('alpha')[index] * u.Unit('')
        beta = trace.get_values('beta')[index] * u.Unit('')
        amplitude = trace.get_values('amplitude')[index] * 1e-11 * u.Unit('cm-2 s-1 TeV-1')
        reference = 1 * u.TeV
        y = Log10Parabola.evaluate(energy, alpha=alpha, amplitude=amplitude, beta=beta, reference=reference)
        model_values.append(y)
    
    model_values = np.array(model_values)
    lower, median, upper = np.percentile(model_values, axis=0, q=percentiles)
    line, = ax.plot(energy, energy**scale * median, label=label, color=color)
    
    last_color = line.get_color()
    ax.fill_between(energy, energy**scale * lower, energy**scale * upper, color=last_color, alpha=0.5)
    # fitted_model.plot(energy_range=fit_range, energy_power=2, ax=ax, label=label)


def plot_unfold_result(trace, bins, fit_range, percentiles=[16, 50, 84], ax=None,):
    if not ax:
        ax = plt.gca()

    scale = 2

    norm = 1 * u.Unit('km-2 s-1 TeV-1')
    flux = (trace['expected_counts'][:, :] * norm).to_value(1 / (u.TeV * u.s * u.cm**2)) 

    fit_range = fit_range.to_value('TeV')
    

    lower, median, upper = np.percentile(flux, q=percentiles, axis=0)
    # discard overflow bins
    bins = bins.to_value('TeV')[1:-1]
    lower, median, upper = lower[1:-1], median[1:-1], upper[1:-1]
    bin_center = np.sqrt(bins[0:-1] * bins[1:])


    xl = bin_center - bins[:-1]
    xu = bins[1:] - bin_center

    dl = median - lower
    du = upper - median
    ax.errorbar(bin_center, bin_center**scale * median, yerr=[bin_center**scale * dl, bin_center**scale * du], xerr=[xl, xu], linestyle='', color='black')


def calculate_flux(trace, percentiles=[16, 50, 84]):
    norm = 1 * u.Unit('km-2 s-1 TeV-1')
    flux = (trace['expected_counts'][:, :] * norm).to_value(1 / (u.TeV * u.s * u.cm**2)) 

    return np.percentile(flux, q=percentiles, axis=0)


def create_table(trace, bins, telescope, ignore_overflow=True):
    lower, median, upper = calculate_flux(trace)
    bins = bins.to_value('TeV')
    index = [pd.Interval(left=a, right=b, closed='left') for a, b in zip(bins[:-1], bins[1:])]
    d = {f'{telescope}_lower': lower, f'{telescope}_median': median, f'{telescope}_upper': upper} 
    df = pd.DataFrame(index=index, data=d)
    if ignore_overflow:
        return df.iloc[1:-1]
    return df


def write_table(frames):
    df_final = reduce(lambda left, right: pd.merge(left=left, right=right, left_index=True, right_index=True, how='outer'), frames)

    def fexp(number):
        (sign, digits, exponent) = Decimal(number).as_tuple()
        return len(digits) + exponent - 1

    def fman(number, common_exponent):
        return Decimal(number).scaleb(common_exponent).normalize()

    bs = ''
    for e_bin, row in df_final.iterrows():
        bs += f'{e_bin.left:.2f} & {e_bin.right:.2f}'
        # print(bs)
        for tel in LIST_OF_TELS:
            l, m, u = row[f'{tel}_lower'], row[f'{tel}_median'], row[f'{tel}_upper']
            l = l - m
            u = u - m
            
            if np.isnan(l):
                bs += '&  '
            else:
                t = (f'{fman(m, common_exponent=-fexp(m)):.2f}', f'{fman(l, common_exponent=-fexp(m)):+.2f}', f'{fman(u, common_exponent=-fexp(m)):+.2f}')
                bs += f'& $\\left(\\num{{{t[0]}}}\substack{{ {t[2]} \\\ {t[1]} }}\\right) \; 10^{{{fexp(m)}}}$'
        bs += '\\\\'

    with open('build/pymc_results/unfold/result_table.txt', 'w') as f:
        f.write(bs)


# build a dummy model. this is weird. i know. wtf pymc?
size = plt.gcf().get_size_inches()
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(size[0], 5.5))
axs = axs.ravel()
ax_top_left = axs[0]
ax_top_right = axs[1]
ax_bottom_left = axs[2]
ax_bottom_right = axs[3]

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

telescopes = LIST_OF_TELS
names = LIST_OF_TEL_NAMES

with dummy_model_fit():
    # model has to be on stack to load traces. why? who the f knows
    for telescope, name, ax, color in zip(telescopes, names, axs.ravel(), cycle):
        config = load_config(config_file='./configs/pymc/data_conf.yaml', telescope=telescope)
        trace = pm.load_trace(f'./build/pymc_results/fit/{telescope}/traces')
        plot_spectra(trace, fit_range=config['fit_range'], percentiles=[16, 50, 84], ax=ax, color=color)

frames = []
with dummy_model_unfold():
    for telescope, name, ax in zip(telescopes, names, axs.ravel()):
        config = load_config(config_file='./configs/pymc/data_conf_unfold.yaml', telescope=telescope)
        trace_unfold = pm.load_trace(f'./build/pymc_results/unfold/{telescope}/traces')
        plot_unfold_result(trace_unfold, bins=config['e_true_bins'], fit_range=config['fit_range'], ax=ax)
        df = create_table(trace_unfold, config['e_true_bins'], telescope, ignore_overflow=True)
        frames.append(df)
        # write_parameter_values(trace_unfold, tel)
        ax.text(0.1, 2E-12, f'{name}', alpha=0.4)


write_table(frames)
axs = axs.ravel()
ax_bottom_right.set_xscale('log')
ax_bottom_right.set_yscale('log')

# ax_bottom_left.set_xlabel('Energy / \si{TeV}')
# ax_bottom_right.set_xlabel()


ylabel = '$\\text{E}^2 \\frac{\\text{dN}}{\\text{dE}}$ / \si{\TeV\per\square\centi\meter \per\second}'
xlabel = 'Energy / \si{TeV}'
fig.text(0.57, 0.01, xlabel, ha='center')
fig.text(0.00001, 0.55, ylabel, va='center', rotation='vertical')
# ax_top_left.set_ylabel(ylabel)
# ax_bottom_left.set_ylabel(ylabel)
ax_bottom_left.set_xlim([10**(-1.6), 10**(1.4)])
ax_bottom_left.set_ylim([1E-12, 5E-10])

plt.tight_layout(pad=0, rect=(0.07, 0.05, 1.009, 1))
plt.savefig('build/pymc_results/pymc_unfold_result.pdf')


conf = load_config(config_file='./configs/pymc/data_conf_unfold.yaml', telescope='magic')
with open(f'./build/pymc_results/unfold/bins_per_decade_e_true.txt', 'w') as out:
    out.write(f"{conf['bins_per_decade']}")
with open(f'./build/pymc_results/unfold/bins_per_decade_e_est.txt', 'w') as out:
    out.write(f"{conf['bins_per_decade_e_reco']}")