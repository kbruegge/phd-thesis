import pandas as pd
from decimal import Decimal


def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def fman(number, common_exponent):
    return Decimal(number).scaleb(common_exponent).normalize()


def format_exp(number):
    return f'{fman(l, common_exponent=-fexp(m)):+.2f}'


df = pd.read_csv('build/sensitivity.csv')


df['sensitivity_low'] = -(df['sensitivity'] - df['sensitivity_low'])
df['sensitivity_high'] = -(df['sensitivity'] - df['sensitivity_high'])

sensi_text = []
for _, series in df.iterrows():
    l, m, u = series['sensitivity_low'], series['sensitivity'], series['sensitivity_high']
    t = (f'{fman(m, common_exponent=-fexp(m)):.2f}', f'{fman(l, common_exponent=-fexp(m)):+.2f}', f'{fman(u, common_exponent=-fexp(m)):+.2f}')
    s = f'$\\left(\\num{{{t[0]}}}\substack{{ {t[2]} \\\ {t[1]} }}\\right) \; 10^{{{fexp(m)}}} $'
    sensi_text.append(s) 

df['sensitivity'] = sensi_text

valid = df['valid']
df = df[['e_min', 'e_max', 'multiplicity', 'theta_cut', 'prediction_cut', 'significance', 'sensitivity']]
for c in ['e_min', 'e_max', 'prediction_cut']:
    df[c] = df[c].map('{:.2f}'.format)

# df['sensitivity'] = df['sensitivity'].map('{:.4f}'.format)
df['theta_cut'] = df['theta_cut'].map('{:.3f}'.format)
df['significance'] = df['significance'].map('{:.1f}'.format)
df['multiplicity'] = df['multiplicity'].astype(int).map('{:d}'.format)

print(df)

table = ""
for (_, series), v in zip(df.iterrows(), valid):
    if not v:
        l = [f'{{\\color{{gray}} {s} }}' for s in series.values]
        s = " & ".join(l)
    else:
        s = "&".join(series.values)

    s += "\\\\ "
    table += s

outfile = "./build/event_selection.txt"
with open(outfile, "w") as f:
    f.write(table)