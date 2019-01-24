import matplotlib.pyplot as plt
import numpy as np


# %matplotlib inline


class Euler:
    def __init__(self, l, dt):
        self.l = l
        self.dt = dt

    def step(self, u):
        return u + self.dt * self.l.dot(u)


class BackwardEuler:
    def __init__(self, l, dt):
        self.l = l
        self.dt = dt

    def step(self, u):
        return np.linalg.inv(np.eye(len(l[0])) - self.dt * self.l).dot(u)


class CrankNicholson:
    def __init__(self, l, dt):
        self.l = l
        self.dt = dt

    def step(self, u):
        return np.linalg.inv(np.eye(len(l[0])) - self.dt / 2 * self.l).dot(
            np.eye(len(l[0])) + self.dt / 2 * self.l).dot(u)


def simulate(simulator, initial, step_num):
    import copy
    import time
    s = time.time()
    u = copy.deepcopy(initial)
    ret = [u]
    for i in range(step_num):
        u = simulator.step(u)
        ret.append(u)
    elapsed_time = time.time() - s
    return np.array(ret), elapsed_time


if __name__ == '__main__':
    l = np.array([[998, 1998], [-999, -1999]])
    u = np.array([1, 0])
    rows_count = 2
    columns_count = 3
    graph_count = rows_count * columns_count
    axes = []
    dt = 0.1
    fig = plt.figure(figsize=(20, 12))
    errs_euler = []
    errs_backward = []
    errs_crank = []
    ts_euler = []
    ts_barkward = []
    ts_crank = []
    for i in range(1, graph_count + 1):
        dt /= 4
        axes.append(fig.add_subplot(rows_count, columns_count, i))
        euler = Euler(l, dt)
        backwardEuler = BackwardEuler(l, dt)
        crankNicholson = CrankNicholson(l, dt)
        x_end = 0.01
        n_step = max(10, int(x_end / dt))
        ans_euler, t_euler = simulate(euler, u, n_step)
        ans_backward, t_backward = simulate(backwardEuler, u, n_step)
        ans_crank, t_crank = simulate(crankNicholson, u, n_step)

        X = np.linspace(0, x_end, n_step + 1)
        ans_correct_u = [2 * np.exp(-x) - np.exp(-1000 * x) for x in X]
        ans_correct_v = [-np.exp(-x) + np.exp(-1000 * x) for x in X]
        err_euler = (np.linalg.norm(ans_euler[:, 0] - ans_correct_u) + np.linalg.norm(
            ans_euler[:, 1] - ans_correct_v)) / (len(X))
        err_backward = (np.linalg.norm(ans_backward[:, 0] - ans_correct_u) + np.linalg.norm(
            ans_backward[:, 1] - ans_correct_v)) / len(X)
        err_crank = (np.linalg.norm(ans_crank[:, 0] - ans_correct_u) + np.linalg.norm(
            ans_crank[:, 1] - ans_correct_v)) / (len(X))

        errs_euler.append(err_euler)
        errs_backward.append(err_backward)
        errs_crank.append(err_crank)
        ts_euler.append(t_euler)
        ts_barkward.append(t_backward)
        ts_crank.append(t_crank)
        print('=================================')
        print(f'dt: {dt}')
        print(f'err_euler: {err_euler}')
        print(f'err_backward_euler: {err_backward}')
        print(f'err_crank_nicholson: {err_crank}\n')

        print(f'time_euler: {t_euler}')
        print(f'time_backward_euler: {t_backward}')
        print(f'time_crank_nicholson: {t_crank}\n')

        axes[i - 1].annotate(f'dt: {dt}', xy=(0.5, 0.7), fontsize=16, xycoords='axes fraction',
                             horizontalalignment='center')
        axes[i - 1].plot(X, ans_euler[:, 0], label='u_euler')
        axes[i - 1].plot(X, ans_backward[:, 0], label='u_backward')
        axes[i - 1].plot(X, ans_crank[:, 0], label='u_crank')
        axes[i - 1].plot(X, ans_euler[:, 1], label='v_euler')
        axes[i - 1].plot(X, ans_backward[:, 1], label='v_backward')
        axes[i - 1].plot(X, ans_crank[:, 1], label='v_crank')
        plt.legend()
        # axes[i - 1].scatter(X, ans_correct_u)
        # axes[i - 1].scatter(X, ans_correct_v)
    plt.show()
