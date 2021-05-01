import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df = pd.read_csv('./data/rta/output.csv.gz', names=['alt', 'az', 'prediction', 'energy', 'datetime', 'host'], parse_dates=['datetime'])
df = df.set_index(df.datetime)
df.drop(columns=['datetime'], inplace=True)
# drop two last rows. they might be incomplete
df.drop(df.tail(2).index, inplace=True)

start_offset = df.index.min().second + df.index.min().microsecond / 1000000 + df.index.min().minute * 60
start_offset_hours = df.index.min().hour
df.index = df.index + pd.DateOffset(hours=-start_offset_hours, seconds=-start_offset, )

aggregated_rate = df.prediction.groupby(pd.Grouper(freq='100ms', base=0)).count() * 10
fig, ax = plt.subplots()

ax.scatter(aggregated_rate.index, aggregated_rate, color='gray', alpha=0.5, s=2)
ax.plot(aggregated_rate.rolling(window='60s').mean(), lw=2)

df_phobos = df[df.host == "phobos.app.tu-dortmund.de"]
df_vollmond = df[df.host == "vollmond.app.tu-dortmund.de"]

aggregated_rate = df_phobos.prediction.groupby(pd.Grouper(freq='100ms', base=0)).count() * 10
ax.plot(aggregated_rate.rolling(window='60s').mean(), lw=1, alpha=0.8)

aggregated_rate = df_vollmond.prediction.groupby(pd.Grouper(freq='100ms', base=0)).count() * 10
ax.plot(aggregated_rate.rolling(window='60s').mean(), lw=1, alpha=0.8)


seconds = mdates.SecondLocator(bysecond=[30])
minutes = mdates.MinuteLocator()  
minutes_fmt = mdates.DateFormatter('%-M')

ax.xaxis.set_minor_locator(seconds)
ax.xaxis.set_major_locator(minutes)
ax.xaxis.set_major_formatter(minutes_fmt)

ax.set_xlim([aggregated_rate.index.min(), aggregated_rate.index.max()])
ax.set_ylim([0, 40000])
ax.set_xlabel('Elapsed time / \\si{\\minute}')
ax.set_ylabel('Event rate / \\si[per-mode=reciprocal]{\per\second}')
plt.tight_layout(pad=0)

plt.savefig('./build/rta.pdf')



