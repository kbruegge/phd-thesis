from scipy.special import erf
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def log_par_integral(lower, upper, N, alpha, beta):
    beta = beta / np.log(10)

    k = N * 0.5 * np.sqrt(np.pi / beta)
    exp_arg = ((alpha - 1)**2) / (4 * beta)

    c = (alpha - 1) / (2 * np.sqrt(beta))
    erf_arg_lower = np.log(lower) * np.sqrt(beta) + c
    erf_arg_upper = np.log(upper) * np.sqrt(beta) + c

    return k * np.exp(exp_arg) * (erf(erf_arg_upper) - erf(erf_arg_lower))


def f(E, phi, alpha, beta):
    return phi * E**(-alpha - beta * np.log10(E))


bins = np.logspace(-2, 2, 50)

reference_result = quad(lambda e: f(e, 4.0, 2.0, 0.5), a=1, b=2)[0]
analytical_result = log_par_integral(1, 2, 4.0, 2.0, 0.5)
print(reference_result)
print(analytical_result)


reference_result = np.array([quad(lambda e: f(e, 4.0, 2.0, 0.5), a=a, b=b)[0] for a, b in zip(bins[:-1], bins[1:])])
analytical_result = log_par_integral(bins[:-1], bins[1:], 4.0, 2.0, 0.5)
# print(reference_result)
# print(analytical_result)
assert np.allclose(reference_result, analytical_result)
print('Testing equality')

N = 150

print(np.log(bins))
invalid_coords = []
valid_coords = []
zero_coords = []
nan_coords = []
for b_param in tqdm(np.linspace(0.01, 4, N)):
    for a_param in np.linspace(0.01, 7, N):
        reference_result = np.array([quad(lambda e: f(e, 4.0, a_param, b_param), a=a, b=b)[0] for a, b in zip(bins[:-1], bins[1:])])
        analytical_result = log_par_integral(bins[:-1], bins[1:], 4.0, a_param, b_param)
        c = (a_param - 1) / (2 * np.sqrt(b_param / np.log(10)))
        erf_arg = np.sqrt(b_param / np.log(10)) + c
        
        if np.allclose(reference_result, analytical_result):
            valid_coords.append([a_param, b_param, erf_arg])
        else:
            invalid_coords.append([a_param, b_param, erf_arg])
        if np.isnan(analytical_result).any():
            nan_coords.append([a_param, b_param, erf_arg])
        if (analytical_result == 0).all():
            # print('Had zero!')
            zero_coords.append([a_param, b_param, erf_arg])
        

invalid_coords = np.array(invalid_coords)
valid_coords = np.array(valid_coords)
zero_coords = np.array(zero_coords)
nan_coords = np.array(nan_coords)


kwargs = dict(alpha=0.8, linewidth=0, edgecolor='none', s=5)

_, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15, 7))
ax1.scatter(invalid_coords[:, 0], invalid_coords[:, 1], color='yellow', label='result != truth', **kwargs)
ax1.scatter(zero_coords[:, 0], zero_coords[:, 1], color='orange', label='Zero Result', **kwargs)
ax1.scatter(nan_coords[:, 0], nan_coords[:, 1], color='crimson', label='NaN result', **kwargs)
ax1.scatter(valid_coords[:, 0], valid_coords[:, 1], color='green', label='Valid result', **kwargs)
ax1.set_xlabel('$\\alpha$')
ax1.set_ylabel('$\\beta$')
ax1.legend()
ax1.set_aspect('equal')
# ax2.hist2d(invalid_coords[:, 0], invalid_coords[:, 1], bins=20)
ax2.scatter(invalid_coords[:, 2], invalid_coords[:, 1], color='yellow', label='result != truth' , **kwargs)
ax2.scatter(zero_coords[:, 2], zero_coords[:, 1], color='orange', label='Zero Result', **kwargs)
ax2.scatter(nan_coords[:, 2], nan_coords[:, 1], color='crimson', label='NaN result', **kwargs)
ax2.scatter(valid_coords[:, 2], valid_coords[:, 1], color='green', label='Valid result', **kwargs)
ax2.set_xlabel('argument of error function')
ax1.set_ylabel('$\\beta$')
# ax2.set_aspect('equal')
ax2.legend()

ax3.scatter(invalid_coords[:, 2], invalid_coords[:, 0], color='yellow', label='result != truth', **kwargs)
ax3.scatter(zero_coords[:, 2], zero_coords[:, 0], color='orange', label='Zero Result', **kwargs)
ax3.scatter(nan_coords[:, 2], nan_coords[:, 0], color='crimson', label='NaN result', **kwargs)
ax3.scatter(valid_coords[:, 2], valid_coords[:, 0], color='green', label='Valid result', **kwargs)
ax3.set_xlabel('argument of error function')
ax1.set_xlabel('$\\alpha$')
# ax3.set_aspect('equal')
ax3.legend()
plt.tight_layout()
plt.savefig('build/validity.pdf')

