import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# % matplotlib
# inline
plt.rcParams['figure.figsize'] = (15.0, 15.0)

delta = 0.1
num = 101
x = np.linspace(0, 1, num)
y = np.linspace(0, 1, num)
u = np.zeros((num, num), dtype='float64')
# init
for i in range(num):
    for j in range(num):
        if x[i] == 0 or y[j] == 0 or y[j] == 1:
            u[i, j] = 0
        elif x[i] == 1:
            u[i, j] = y[j] * (1 - y[j])

eps = 0.1
stop = False
step = 0
while not stop:
    step += 1

    u_new = np.zeros_like(u, dtype='float64')
    for i in range(num):
        for j in range(num):
            if x[i] == 0 or y[j] == 0 or y[j] == 1:
                u_new[i, j] = 0
            elif x[i] == 1:
                u_new[i, j] = y[j] * (1 - y[j])
            else:
                u_new[i, j] = (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1] -
                               delta ** 2 * (6 * x[i] * y[j] * (1 - y[j]) - 2 * x[i] ** 3)) / 4

    if step % 100 == 0:
        print(step)
        print(np.linalg.norm(u - u_new))

    if np.linalg.norm(u - u_new) <= eps:
        stop = True
    u = copy.deepcopy(u_new)

print(step)
ans = np.zeros((num, num), dtype='float64')
for i in range(num):
    for j in range(num):
        ans[i, j] = x[i] ** 3 * y[j] * (1 - y[j])

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, ans, color='r')
ax.plot_wireframe(X, Y, u)
