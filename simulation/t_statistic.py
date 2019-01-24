import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

n = 10
mu = 3
var = 2
for _ in range(10):
    X = [np.random.normal(mu, np.math.sqrt(var)) for i in range(n)]
    X_mean = np.mean(X)
    sigma_hat = np.sum((X - X_mean) ** 2) / (n - 1)
    T = np.math.sqrt(n) * (X_mean - mu) / sigma_hat
    plt.plot(T)
