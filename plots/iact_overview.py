from gammapy.data import DataStore
from dateutil import parser


table_string  = ''

for telescope in ['hess', 'fact' , 'magic',  'veritas']:
    ds = DataStore.from_dir(f'./plots/data/joint_crab/dl3/{telescope}')
    t = ds.obs_table
    # print(t.columns)
    try:
        year = parser.parse(t['DATE-OBS'][0]).year
    except KeyError:
        year = 2013 if telescope == 'magic' else 2011

    alt_max = t['ALT_PNT'].max()   
    alt_min = t['ALT_PNT'].min()   

    duration = sum(t['ONTIME']) / 3600

    ids = t['OBS_ID']
    obs = ds.get_observations(ids)
    n_events = sum([len(o.events.energy) for o in obs])

    table_string += f'\{telescope} & {year} & {duration:.2f} & {n_events} & \SIrange{{{alt_min:.1f}}}{{{alt_max:.1f}}}{{\degree}} \\\\  \n '

with open('build/iact_overview.txt', 'w') as f:
    f.write(table_string)


