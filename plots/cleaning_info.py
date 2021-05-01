import pandas as pd
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

with open('./configs/preprocessing/config.yaml', 'r') as f:
    d = yaml.load(f)
    df = pd.DataFrame(d['cleaning_level'])
    # df['CHEC'] = ['', '', ''] 
    # df['FlashCam'] = ['', '', ''] 
    df.index = ['Neighbor Threshold', 'Core Threshold', 'Min Pixel']

    # df = df[['CHEC', 'FlashCam', 'DigiCam', 'NectarCam', 'LSTCam']]

    table = ""
    cols = [f'\\textbf{{{c}}}' for c in df.columns]
    s = f" &" + " & ".join(cols) + "\\\\ "
    table += s

    for n, series in df.iterrows():
        v = [str(s) for s in series.values]
        s = f"{n} &" + " & ".join(v) + "\\\\ "
        table += s


with open('build/cleaning_info.txt', 'w') as f:
    f.write(table)
