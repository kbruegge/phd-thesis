import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def f(x, gamma):
    return x ** (-gamma) + 0.2


def f_prime(x, gamma):
    return -x ** (-gamma) * np.log(x)


def taylored_f(x, gamma, gm):
    return f(x=E, gamma=gm) + f_prime(x=E, gamma=gm) * (gamma - gm)

size = plt.gcf().get_size_inches()
plt.figure(figsize=(size[0]/2, 2.5))

E = 5

gamma = np.linspace(0.1, 3.5, 100)
plt.plot(gamma, f(x=E, gamma=gamma), label="$E^{-\gamma}$")

gm = 1.2
gamma = np.linspace(0.0, 2.5, 100)
y_hat = taylored_f(E, gamma, gm)
plt.plot(gamma, y_hat, label="$Taylor Approximation$")

gamma = np.linspace(0.1, 3.5, 100)
sigma = 0.5
x = np.linspace(gm - 1.2, gm + 1.2, 100)
n = norm.pdf(x, loc=gm, scale=sigma)
n = n / n.max() * 0.15

plt.plot(x, n, color="gray")
plt.fill_between(
    x, 0, n, where=(x > (gm - sigma)) & (x < (gm + sigma)), color="gray", alpha=0.3
)

y = (f(E, gm) - sigma / 2) + x * 0.2

plt.plot(n * 4 - 1, y, color="gray")


y_lim_max = 0.8
x_lim_max = 3

y_t_1 = taylored_f(E, gamma=gm - sigma, gm=gm) / y_lim_max
plt.axvline(x=(gm - sigma), ymax=y_t_1, color="gray", linestyle="--")
plt.axhline(
    y_t_1 * y_lim_max,
    xmax=(gm - sigma + 1) / (x_lim_max + 1),
    color="gray",
    linestyle="--",
)

y_t_2 = taylored_f(E, gamma=gm + sigma, gm=gm) / y_lim_max
plt.axvline(x=(gm + sigma), ymax=y_t_2, color="gray", linestyle="--")
plt.axhline(
    y_t_2 * y_lim_max,
    xmax=(gm + sigma + 1) / (x_lim_max + 1),
    color="gray",
    linestyle="--",
)

plt.fill_betweenx(
    y,
    x1=n * 4 - 1,
    x2=-1,
    where=((y > y_t_2 * y_lim_max) & (y < y_t_1 * y_lim_max)),
    color="gray",
    alpha=0.3,
)

plt.ylim([0, y_lim_max])
plt.xlim([-1, x_lim_max])

# plt.gca().spines["right"].set_visible(False)
# plt.gca().spines["top"].set_visible(False)

plt.gca().set_xticks([])
plt.gca().set_yticks([])

plt.text(gm - sigma, -0.05, '$\mu_{\gamma} - \sigma_{\gamma}$', va='center', ha='center')
plt.text(gm + sigma, -0.05, '$\mu_{\gamma} + \sigma_{\gamma}$', va='center', ha='center')
# plt.xlabel("$\gamma$")
# plt.ylabel("$E^{-\gamma}$")
plt.tight_layout(pad=0, rect=(-0.006, 0.02, 1.006, 1))
plt.savefig('build/error_propagation.pdf')
