import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white', grid=False, titlesize='small', labelsize='small', labelcolor='black', edgecolor='gray', linewidth=2)
# axes.titlesize: x-large
# axes.labelsize: medium/
# axes.labelcolor: 555555

hfont = {'fontname': 'Minion Pro'}

# with plt.xkcd():
fig, axs = plt.subplots(2, 3)

axs = axs.ravel()
for ax in axs:
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.spines['bottom'].set_visible(True)    
    ax.spines['left'].set_visible(False)    

ax = axs[0]
x = np.linspace(0, 1, 100)
ax.plot(x, -x + 1)
# ax.plot(x, -x + 1 + 0.15 * np.sin(x * 2 * np.pi))
# ax.set_xlabel('Energy', hfont)
# ax.set_ylabel('Flux', hfont)
ax.set_title('Source Spectrum', hfont)
ylim = np.array(ax.get_ylim())
ylim[0] = 0
ax.set_ylim(ylim)

y = ax.get_ylim()[1]/2
w = ((y*2)) * 0.04

x = ax.get_xlim()[1]/2
l = ((x*2)) * 0.1
print(w, l, y)
ax.arrow(0.8, y, 0.4, 0, length_includes_head=True, width=w, head_width=2 * w, head_length=l, clip_on=False, color='gray')

ax = axs[1]
x = np.random.normal(0.5, 0.13, size=1000)
bins = np.linspace(0, 1, 10)
ax.hist(x, bins=bins, histtype='step', lw=2, )
# x = np.random.normal(0.35, 0.2, size=2000)
# ax.hist(x, bins=bins, histtype='step', lw=2, )
# ax.set_xlabel('Energy', hfont)
# ax.set_ylabel('Counts', hfont)
ax.set_title('Triggered Counts', hfont)
ylim = np.array(ax.get_ylim())
ylim[0] = 0
ax.set_ylim(ylim)

print(ax.get_ylim())
y = ax.get_ylim()[1]/2
w = ((y*2)) * 0.04

x = ax.get_xlim()[1]/2
l = ((x*2)) * 0.1
print(w, l, y)
ax.arrow(0.8, y, 0.4, 0, length_includes_head=True, width=w, head_width=2 * w, head_length=l, clip_on=False, color='gray')

ax = axs[2]
x = np.random.normal(0.5, 0.13, size=1000)
bins = np.linspace(0, 1, 10)
h, _, _ = ax.hist(x, bins=bins, histtype='step', lw=2, density=True)
x = np.random.normal(0.4, 0.17, size=1000)
ax.hist(x, bins=bins, histtype='step', lw=2, density=True, color='xkcd:crimson')
# ax.set_xlabel('Energy', hfont)
# ax.set_ylabel('Counts', hfont)
ax.set_title('Analyzed Counts', hfont)
ylim = np.array(ax.get_ylim())
ylim[0] = 0
ylim[1] = h.max() + 1.
ax.set_ylim(ylim)
# ...

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.2, wspace=0.3)
plt.savefig('build/iact_sketch.pdf')