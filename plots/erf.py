import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

size = plt.gcf().get_size_inches()
plt.figure(figsize=(size[0]/2, 3.05 / 1.7))
x = np.linspace(-2.5, 2.5, 400)
plt.plot(x, erf(x))
plt.xlim([-2.5, 2.5])
plt.xlabel('$x$')
plt.ylabel('$\operatorname{erf}(x)$')
plt.tight_layout(pad=0, rect=(00.039, 0, 1.04, 1.019))
plt.savefig('build/erf.pdf')
