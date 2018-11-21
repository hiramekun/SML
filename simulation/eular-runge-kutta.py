class Eular:
    def __init__(self, h, f):
        self.h = h
        self.f = f

    # get y_i and return y_i+1
    def step(self, x, y):
        return y + self.h * self.f(x, y)


class RungeKutta:
    def __init__(self, h, f):
        self.h = h
        self.f = f

    def step(self, x, y):
        func = self.f
        k1 = func(x, y)
        k2 = func(x + self.h / 2, y + self.h / 2 * k1)
        k3 = func(x + self.h / 2, y + self.h / 2 * k2)
        k4 = func(x + self.h, y + self.h * k3)
        return y + self.h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


