import matplotlib.pyplot as plt
import numpy as np
from gammapy.spectrum import SpectrumObservation
import glob


def plot_counts(ax, observations, name):
    '''
    Plot the signal counts i the extraced data.
    '''
    signal_counts = [obs.on_vector.data.data for obs in observations]
    signal_counts = np.sum(signal_counts, axis=0)

    bkg_counts = [obs.background_vector.data.data for obs in observations]
    bkg_counts = np.sum(bkg_counts, axis=0)

    x = observations[0].e_reco.lower_bounds.to_value('TeV')

    start = np.argwhere(signal_counts)[0][0] - 1
    stop = np.argwhere(signal_counts)[-1][0] + 1
    line, = ax.step(x[start:stop], signal_counts[start:stop], where='post', lw=2, label=name)
    ax.step(x[start:stop], bkg_counts[start:stop], where='post', lw=1, color=line.get_color(), alpha=0.5)
    return ax


fig, ax = plt.subplots(1, 1)

# filenames = glob.glob('./plots/data/joint_crab/spectra/magic/pha_*.fits')
filenames = glob.glob('./build/pymc_results/spectra/magic/pha_*.fits')
obs = [SpectrumObservation.read(f) for f in filenames]
plot_counts(ax, obs, 'MAGIC')

filenames = glob.glob('./build/pymc_results/spectra/fact/pha_*.fits')
obs = [SpectrumObservation.read(f) for f in filenames]
plot_counts(ax, obs, 'FACT')

filenames = glob.glob('./build/pymc_results/spectra/veritas/pha_*.fits')
obs = [SpectrumObservation.read(f) for f in filenames]
plot_counts(ax, obs, 'VERITAS')

filenames = glob.glob('./build/pymc_results/spectra/hess/pha_*.fits')
obs = [SpectrumObservation.read(f) for f in filenames]
plot_counts(ax, obs, 'H.E.S.S')

ax.legend()
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('Estimated Energy / \\si{TeV}')
ax.set_ylabel('Counts')
ax.set_xlim([0.02, 35])
ax.set_ylim([0, 85])
plt.tight_layout(pad=0, rect=(0, 0, 1.0082, 1))
plt.savefig('build/iact_counts.pdf')
# from IPython import embed; embed()


