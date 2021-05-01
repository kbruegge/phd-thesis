import numpy as np
import corner
import click
import matplotlib.pyplot as plt
import os
import matplotlib


font = {'size': 16}

matplotlib.rc('font', **font)

@click.command()
@click.argument('input_dir', type=click.Path(dir_okay=True, file_okay=False))
@click.argument('output_file', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--truths', nargs=3, type=float)
@click.option('--color',  default='crimson')
@click.option('--reverse', default=False )
def main(input_dir, output_file, truths, color, reverse):

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

    corner.corner(samples,
                    labels=['$\\Phi_{0}$', '$\\Gamma$' , '$\\beta$'],
                    reverse=reverse,
                    truth_color=color,
                    truths=truths if truths else None
                  )

    plt.savefig(output_file)


if __name__ == '__main__':
    main()
