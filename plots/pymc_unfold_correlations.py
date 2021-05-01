import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import transforms
import numpy as np
import pymc3 as pm
from pymc_utils import load_config, dummy_model_unfold
from plot_utils import LIST_OF_TEL_NAMES, LIST_OF_TELS


def plot_correlations(names, matrices):
    size = plt.gcf().get_size_inches()
    fig, axs = plt.subplots(2, 2, figsize=(size[0], 4.7))
    norm = colors.DivergingNorm(vmin=-0.7, vcenter=0, vmax=0.3)
    for ax, name, matrix in zip(axs.ravel(), names, matrices):
        mc = np.ma.masked_where(matrix > 0.99, matrix)
        
        # RdYlBu
        im = ax.imshow(mc.T, cmap="RdBu_r", norm=norm)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_aspect("equal")
        ax.text(
            0.2,
            0.07,
            name,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
        )


    plt.tight_layout(pad=0, rect=(-0.015, 0, 0.985, 1))
    plt.subplots_adjust(wspace=0.04, hspace=0.04, top=1, bottom=0, )
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), label="Spearman Correlation Coefficient", use_gridspec=True, pad=0, anchor=(0.2, 0.5) )

    dx = 0 / 72.
    dy = 1 / 72. 
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    for label in cb.ax.get_yticklabels():
        label.set_transform(label.get_transform() + offset)


telescopes = LIST_OF_TELS
names = LIST_OF_TEL_NAMES

# plot covariance matrix
matrices = []
with dummy_model_unfold():
    for telescope, name in zip(telescopes, names):
        config = load_config(
            config_file="./configs/pymc/data_conf_unfold.yaml", telescope=telescope
        )
        trace_unfold = pm.load_trace(f"./build/pymc_results/unfold/{telescope}/traces")
        m = trace_unfold.get_values("expected_counts")
        c = np.corrcoef(m.T[1:-1])
        matrices.append(c)

plot_correlations(names, matrices)

plt.savefig("build/pymc_results/pymc_unfold_correlations.pdf")
