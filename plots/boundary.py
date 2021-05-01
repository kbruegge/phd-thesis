import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D  

offset = 0.6
X, y = make_blobs(n_samples=400, centers=[[offset, 0], [-offset, 0]], cluster_std=0.5, random_state=0)

# train the linear regressor and save the coefficents
reg = linear_model.LinearRegression()
reg.fit(X, y)
# b_1, b_2 = reg.coef_
# b_0 = reg.intercept_

# solve the function y = b_0 + b_1*X_1 + b_2 * X_2 for X2
# x1s = np.linspace(-1, 4)
# x2s = (0.5 - b_0 - b_1 * x1s) / b_2

size = plt.gcf().get_size_inches()
fig = plt.figure(figsize=(size[0] / 2, 2.2))
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.patch.set_facecolor('white')

m = y == 1
ax.scatter(X[m, 0], X[m, 1], y[m], s=25, )
ax.scatter(X[~m, 0], X[~m, 1], y[~m], s=25, )

ts = np.linspace(-1, 1)
xx, yy = np.meshgrid(ts, ts)
X_plane = np.vstack([xx.ravel(), yy.ravel()]).T
height = reg.predict(X_plane)
# from IPython import embed; embed()
ax.plot_surface(xx, yy, height.reshape(xx.shape), alpha=0.4, color='gray', antialiased=False, rstride=50, cstride=50,)

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels(['', '0', '', '', '', '', '1'])

# ax.set_zticks([0, 1])

# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# fig.set_facecolor('white')
ax.set_facecolor('white') 

ax.set_zlim([-0.1, 1.1])
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])

# ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
# ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
# ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

# ax.set_zlabel('$y$')

plt.tight_layout(pad=0, rect=(-0.115, -0.05, 1.0015, 1.05)) # (left, bottom, right, top
plt.savefig('build/boundary.pdf')
# plt.plot(x1s, x2s, color='gray', linestyle='--')

# plt.fill_between(x1s, x2s, 6, color='crimson', alpha=0.1)
# plt.fill_between(x1s, x2s, -1, color='blue', alpha=0.1)

# plt.grid()
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.margins(x=0, y=0)
# None

