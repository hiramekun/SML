import numpy as np


class CGMethod:
    def __init__(self, A, b, x_0, eps):
        self.A = A
        self.b = b
        self.r = b - A.dot(x_0)
        self.p = self.r
        self.eps = eps

    def step(self, x):
        alpha = self.r.dot(self.r) / (self.p.dot(self.A).dot(self.p))
        r = self.r - alpha * self.A.dot(self.p)
        beta = r.dot(r) / (self.r.dot(self.r))
        x_1 = x + alpha * self.p
        self.p = r + beta * self.p
        self.r = r
        return x_1, np.linalg.norm(r) <= self.eps


if __name__ == '__main__':
    A = np.array([[5, 2, 1], [2, 1, 4], [1, 4, 2]])
    x = np.array([0, 0, 0])
    b = np.array([17, 11, 16])
    cg = CGMethod(A, b, x, 10 ** (-8))
    ended = False

    while not ended:
        x, ended = cg.step(x)
        print(x)
