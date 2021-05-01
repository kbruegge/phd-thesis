import pandas as pd
import matplotlib.pyplot as plt
from cta_plots import colors


def name_finder(s):
    if 'MST' in s:
        return 'MST'
    if 'SST' in s:
        return 'SST'
    if 'LST' in s:
        return 'LST'


def read_layout(path):
    df = pd.read_csv(path, engine='python', delim_whitespace=True, names=['x', 'y', 'z', 'type'], comment='#',usecols=[5, 6, 7, 9])
    df['type'] = df.type.apply(name_finder)
    return df


sst_color = colors.SST
sst_marker= 'o'

mst_color = colors.MST
mst_marker= 's'

lst_color = colors.LST
lst_marker= 'v'

df_lapalma = read_layout('plots/data/lapalma_coords.lis')
df_paranal = read_layout('plots/data/paranal_coords.lis')

df_lapalma.x -= 2000


size = plt.gcf().get_size_inches()
f, ax1 = plt.subplots(1, 1, figsize=(size[0], 4))

df= df_lapalma[df_lapalma.type == 'SST']
p_sst = ax1.scatter(df.x, df.y, c=sst_color, marker=sst_marker)

df= df_lapalma[df_lapalma.type == 'MST']
p_mst = ax1.scatter(df.x, df.y, c=mst_color, marker=mst_marker)

df= df_lapalma[df_lapalma.type == 'LST']
p_lst = ax1.scatter(df.x, df.y, c=lst_color, marker=lst_marker)


df= df_paranal[df_paranal.type == 'SST']
p_sst = ax1.scatter(df.x, df.y, c=sst_color, marker=sst_marker, label='SST')

df= df_paranal[df_paranal.type == 'MST']
p_mst = ax1.scatter(df.x, df.y, c=mst_color, marker=mst_marker, label='MST')

df= df_paranal[df_paranal.type == 'LST']
p_lst = ax1.scatter(df.x, df.y, c=lst_color, marker=lst_marker, label='LST')


ax1.arrow(x=-1500, y=0, dx=0, dy=500, color='black', width=5, head_width=20, alpha=0.7)
ax1.arrow(x=-1500, y=0, dx=0, dy=-500, color='black', width=5, head_width=20, alpha=0.7)

ax1.text(-1400, 0, '\\SI{1}{\kilo\metre}', rotation=90, ha='center', va ='center', alpha=0.7)

ax1.text(df_lapalma.x.mean(), 550, 'La Palma', ha='center', va ='center',)
ax1.text(df_paranal.x.mean(), 1300, 'Paranal', ha='center', va ='center',)

ax1.set_aspect('equal')
ax1.legend(loc=(0.04, 0.815), fancybox=False)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([-1200, 1400])
ax1.set_facecolor('white')
plt.tight_layout(pad=0, rect=(-0.05, 0, 1.044, 1))

plt.savefig('build/array_layout.pdf')

with open('build/num_mst_north.txt','w') as f:
    f.write(f'{len(df_lapalma[df_lapalma.type == "MST"])}')
with open('build/num_lst_north.txt','w') as f:
    f.write(f'{len(df_lapalma[df_lapalma.type == "LST"])}')
with open('build/num_tel_north.txt','w') as f:
    f.write(f'{len(df_lapalma)}')


with open('build/num_sst_south.txt', 'w') as f:
    f.write(f'{len(df_paranal[df_paranal.type == "SST"])}')
with open('build/num_mst_south.txt','w') as f:
    f.write(f'{len(df_paranal[df_paranal.type == "MST"])}')
with open('build/num_lst_south.txt','w') as f:
    f.write(f'{len(df_paranal[df_paranal.type == "LST"])}')
with open('build/num_tel_south.txt','w') as f:
    f.write(f'{len(df_paranal)}')