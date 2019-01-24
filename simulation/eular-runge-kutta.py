import matplotlib.pyplot as plt


class Eular:
    def __init__(self, h, f):
        self.h = h
        self.f = f

    # get y_i and return y_i+1
    def step(self, x):
        return x + self.h * self.f(x)


class RungeKutta:
    def __init__(self, h, f):
        self.h = h
        self.f = f

    def step(self, x):
        func = self.f
        k1 = func(x)
        k2 = func(x + self.h / 2 * k1)
        k3 = func(x + self.h / 2 * k2)
        k4 = func(x + self.h * k3)
        return x + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def f(x):
    return - 500 * x


if __name__ == '__main__':
    h = 0.001
    x_next_eular = 1
    x_next_runge = 1
    eular = Eular(h, f)
    rungeKutta = RungeKutta(h, f)

    ans_eular = [x_next_eular]
    ans_runge_kutta = [x_next_eular]
    for i in range(100):
        x_next = eular.step(x_next_eular)
        x_next = eular.step(x_next_runge)
        ans_eular.append(x_next_eular)
        ans_runge_kutta.append(x_next_runge)

    plt.plot(ans_eular)
    plt.plot(ans_runge_kutta)
