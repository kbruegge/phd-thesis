import h5py
import numpy as np
import pandas as pd


def h5py_get_n_rows(file_path, key="array_events"):
    """
    Stolen from IACTtools and pyfact I think
    """
    with h5py.File(file_path, "r") as f:
        group = f.get(key)

        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))

        return group[next(iter(group.keys()))].shape[0]


def mean_multiplicity(file_path, key="array_events"):
    with h5py.File(file_path, "r") as f:
        group = f.get(key)

        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))
        return group["num_triggered_telescopes"][()].mean()


def round_angle(v):
    return np.round(np.rad2deg(v))


def pointing(group):
    az = round_angle(group["mc_max_az"][0]).astype(np.int)
    alt = round_angle(group["mc_max_alt"][0]).astype(np.int)
    return f"{alt}, {az}"


def energy_ranges(group):
    e_max = np.round(group["mc_energy_range_max"][0])
    e_min = np.round(group["mc_energy_range_min"][0], decimals=4)

    return f"{e_min} -- {e_max}"


def num_showers(group):
    n = group["mc_num_showers"][()].sum()
    return f"\\num{{{n}}}"


def spectral_index(group):
    gamma = group["mc_spectral_index"][0]
    return f"{gamma}"


def viewcone_radius(group):
    r = group["mc_max_viewcone_radius"][0].astype(np.int)
    return f"{r}"


def scatter_radius(group):
    r = group["mc_max_scatter_range"][0].astype(np.int)
    return f"{r}"


def info_table(file_path):
    with h5py.File(file_path, "r") as f:
        group = f.get("runs")

        if group is None:
            raise IOError('File does not contain group "{}"'.format("runs"))
        d = {
            "Pointing Alt/Az (\\si{\\degree})": pointing(group),
            "Energy (\\si{TeV})": energy_ranges(group),
            "Simulated Showers": num_showers(group),
            "Spectral Index": spectral_index(group),
            "View Cone (\\si{\\degree}) ": viewcone_radius(group),
            "Scatter Radius (\\si{\\metre})": scatter_radius(group),
        }

    d["Processed Events"] = f"\\num{{{h5py_get_n_rows(file_path)}}}"
    d["\\corsika Runs"] = f"\\num{{{h5py_get_n_rows(file_path, key='runs')}}}"
    d["Mean Multiplicity"] = np.round(mean_multiplicity(file_path), decimals=1)

    return d


def build_dataset_table(files):
    d = [info_table(f) for f in files]
    df = pd.DataFrame(d).T

    table = ""
    for n, series in df.iterrows():
        v = [str(s) for s in series.values]
        s = f"{n} &" + " & ".join(v) + "\\\\ "
        table += s

    return table


infile = "data/cta_data/gammas_diffuse.h5"
outfile = "build/gamma_test_num_array_events.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="array_events")}}}')

outfile = "build/gamma_test_num_tel_events.txt"
with open(outfile, "w") as f:
    f.write(f'\\num{{{h5py_get_n_rows(infile, key="telescope_events")}}}')

outfile = "build/gamma_test_mean_multiplicity.txt"
with open(outfile, "w") as f:
    f.write(f"\\num{{{mean_multiplicity(infile):.1f}}}")


files = [
    "./data/cta_data/gammas.h5",
    "./data/cta_data/gammas_diffuse.h5",
    "./data/cta_data/protons.h5",
    "./data/cta_data/electrons.h5",
]
table = build_dataset_table(files)

outfile = "build/dataset_info.txt"
with open(outfile, "w") as f:
    f.write(table)
