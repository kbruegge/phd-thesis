import numpy as np
import click
import matplotlib.pyplot as plt
import os
from gammapy.spectrum import CrabSpectrum
import astropy.units as u


def model(amplitude, alpha, beta):
    amplitude *= 1E-11
    return lambda xx: amplitude * np.power(xx, (-alpha - beta * np.log10(xx)))


magic_model = CrabSpectrum('magic_lp').model
meyer_model = CrabSpectrum('meyer').model

fact_model = model(3.49, 2.54, 0.42)
magic_joint_model = model(4.15, 2.6, 0.44)
hess_joint_model = model(4.47, 2.39, 0.37)


@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--color', default='crimson')
def main(input_dir, output, color):

    alphas = []
    betas = []
    amplitudes = []
    for dir in os.listdir(input_dir):
        d = np.load(os.path.join(input_dir, dir, 'samples.npz'))
        alphas.append(d['alpha'])
        betas.append(d['beta'])
        amplitudes.append(d['amplitude'])

    alphas = np.hstack(alphas)
    betas = np.hstack(betas)
    amplitudes = np.hstack(amplitudes)
    samples = np.vstack([amplitudes, alphas, betas]).T

    energies = np.logspace(-1, 1.5, 100)
    for i in np.random.uniform(0, len(samples), size=100).astype(np.int):
        f = model(*samples[i])
        plt.plot(energies, energies**2 *  f(energies), color='gray', alpha=0.1)

    median_model = model(*np.median(samples, axis=0))
    # from IPython import embed; embed()
    plt.plot(energies, energies**2 * median_model(energies), color='black', label='MCMC Median')

    plt.plot(energies, energies**2 * fact_model(energies), color='xkcd:green', label='FACT (joint)')
    plt.plot(energies, energies**2 * hess_joint_model(energies), color='red', label='HESS (joint)')

    plt.plot(energies, energies**2 * magic_joint_model(energies), color='crimson', ls='--', label='MAGIC (joint)')
    plt.plot(energies, energies**2 * magic_model(energies * u.TeV).to('TeV/(TeV2 cm2 s)'), color='crimson', label='MAGIC (2014)')
    plt.plot(energies, energies**2 * meyer_model(energies * u.TeV).to('TeV/(TeV2 cm2 s)'), color='xkcd:orange', label='Meyer (2010)')
    plt.xlabel('$Energy \\, / \\, \\mathrm{TeV} $')
    plt.ylabel('$E^2 \\cdot \\mathrm{d\\Phi} / \\mathrm{dE}  \\;  $')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    main()
