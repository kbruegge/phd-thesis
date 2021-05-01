import numpy as np
import yaml
import astropy.units as u
import pymc3 as pm


def create_energy_bins(n_bins_per_decade=10, fit_range=None, overflow=False):
    bins = np.logspace(-2, 2, (4 * n_bins_per_decade) + 1)
    if fit_range is not None:
        bins = apply_range(bins, fit_range=fit_range, bins=bins * u.TeV)[0]
        bins = np.append(0.01, bins)
        bins = np.append(bins, 100)
    return bins * u.TeV


def apply_range(*arr, fit_range, bins):
    """
    Takes one or more array-like things and returns only those entries
    whose bins lie within the fit_range.
    """
    idx = np.searchsorted(bins.to_value(u.TeV), fit_range.to_value(u.TeV))
    lo, up = idx[0], idx[1] + 1

    return [a[lo:min(up, len(a) - 1)] for a in arr]


def load_config(config_file, telescope):
    with open(config_file) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
        tel_config = d["datasets"][telescope]
        fit_range = tel_config["fit_range"] * u.TeV
        d = {
            "bins_per_decade": tel_config["bins_per_decade"],
            "bins_per_decade_e_reco": tel_config.get("bins_per_decade_e_reco", None),
            "telescope": telescope,
            "on_radius": tel_config["on_radius"] * u.deg,
            "containment_correction": tel_config["containment_correction"],
            "stack": tel_config.get("stack", False),
            "fit_range": fit_range,
            "e_reco_bins": create_energy_bins(
                tel_config.get("bins_per_decade_e_reco", 20)
            ),
            "e_true_bins": create_energy_bins(
                tel_config["bins_per_decade"], fit_range=fit_range
            ),
        }

        return d


def dummy_model_unfold():
    model = pm.Model(theano_config={"compute_test_value": "ignore"})
    with model:
        pm.HalfFlat("mu_b", shape=2)
        pm.Lognormal("expected_counts", shape=2)
    return model


def dummy_model_fit():
    model = pm.Model(theano_config={'compute_test_value': 'ignore'})
    with model:
        pm.TruncatedNormal('amplitude', )
        alpha = pm.TruncatedNormal('alpha',)
        beta = pm.TruncatedNormal('beta', )

        pm.Deterministic('mu_s', alpha + beta)  # dummy function to load traces.

        pm.TruncatedNormal('mu_b', lower=0, shape=2, mu=[1, 2], sd=5)
    return model
