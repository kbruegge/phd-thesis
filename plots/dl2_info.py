import h5py
import numpy as np
import pandas as pd


# def scatter_radius(group):
#     r = group["mc_max_scatter_range"][0].astype(np.int)
#     return f"{r}"

def get_n_ints(group, key, nrows=2):
    return [f"{n}" for n in group[key][:nrows].astype(np.int)]


def get_n_floats(group, key, nrows=2):
    return [f"{n:.3f}" for n in group[key][:nrows]]


def build_run_table(file, nrows=3):
    with h5py.File(file, "r") as f:
        group = f.get("runs")
        d = {
            "run\_id": get_n_ints(group, key='run_id', nrows=nrows),
            "sim\_index": get_n_ints(group, key='mc_spectral_index', nrows=nrows),
            "num\_showers": get_n_ints(group, key='mc_num_showers', nrows=nrows),
            "shower\_reuse": get_n_ints(group, key='mc_shower_reuse', nrows=nrows),
        }

    df = pd.DataFrame(d)
    df[' '] = ['$\\cdots$'] * len(df)
    # from IPython import embed; embed()
    df = df[df.columns[[0, 1, 2, 4, 3]]]

    print(df)
    table = ""
    table += f"{df.columns[0]} &  &  &  " + "&".join(df.columns[1:]) + "\\\\ "
    for _, series in df.iterrows():
        v = [str(s) for s in series.values]
        s = f"{v[0]} &  &  &  "  + "&".join(v[1:]) + "\\\\ "
        table += s

    # print(table)
    return table
    # from IPython import embed; embed()


def build_array_table(file, nrows=3):
    with h5py.File(file, "r") as f:
        group = f.get("array_events")
        # print(group.keys())
        d = {
            "run\_id": get_n_ints(group, key='run_id', nrows=nrows),
            "event\_id": get_n_ints(group, key='array_event_id', nrows=nrows),

            "num\_sst": get_n_ints(group, key='num_triggered_sst', nrows=nrows),
            "mc\_energy": get_n_floats(group, key='mc_energy', nrows=nrows),
            "total\_intensity": get_n_floats(group, key='total_intensity', nrows=nrows),
        }

    df = pd.DataFrame(d)

    df[' '] = ['$\\cdots$'] * len(df)
    df = df[df.columns[[0, 1, 2, 3, 5, 4]]]
    print(df)
    table = ""
    table += f"{df.columns[0]} & {df.columns[1]} &  &  " + "&".join(df.columns[2:]) + "\\\\ "
    for _, series in df.iterrows():
        v = [str(s) for s in series.values]
        s = f"{v[0]} & {v[1]} &  &  " + "&".join(v[2:]) + "\\\\ "
        table += s

    # print(table)
    return table



def build_tel_table(file, nrows=3):
    with h5py.File(file, "r") as f:
        group = f.get("telescope_events")
        # print(group.keys())
        d = {
            "run\_id": get_n_ints(group, key='run_id', nrows=nrows),
            "event\_id": get_n_ints(group, key='array_event_id', nrows=nrows),
            "telescope\_id": get_n_ints(group, key='telescope_id', nrows=nrows),
            "width": get_n_floats(group, key='width', nrows=nrows),
            "length": get_n_floats(group, key='length', nrows=nrows),
            "intensity": get_n_floats(group, key='intensity', nrows=nrows),
        }

    df = pd.DataFrame(d)
    df[' '] = ['$\\cdots$'] * len(df)
    df = df[df.columns[[0, 1, 2, 3, 4, 6, 5]]]

    print(df)
    table = ""
    table += "&".join(df.columns[:]) + "\\\\ "
    for _, series in df.iterrows():
        v = [str(s) for s in series.values]
        s = "&".join(v[:]) + "\\\\ "
        table += s

    # print(table)
    return table


infile = "./data/cta_data/gammas.h5"

table = build_run_table(infile)
outfile = "./build/dl2_info_runs.txt"
with open(outfile, "w") as f:
    f.write(table)

table = build_array_table(infile)
outfile = "./build/dl2_info_array.txt"
with open(outfile, "w") as f:
    f.write(table)

table = build_tel_table(infile)
outfile = "./build/dl2_info_telescope.txt"
with open(outfile, "w") as f:
    f.write(table)