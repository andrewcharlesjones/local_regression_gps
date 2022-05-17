import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal as mvn, norm
from scipy.linalg import sqrtm

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

limits = [-5, 5]
n = 400
noise_variance = 0  # 1e-1
X = np.linspace(*limits, n).reshape(-1, 1)
kernel = RBF()
K = kernel(X) + 1e-8 * np.eye(n)
xstar = np.zeros((1, 1))
k_xstar_X = kernel(xstar, X)

n_functions = 5

plt.figure(figsize=(28, 7))

## Linear model
plt.subplot(141)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title("Linear model\n(homoskedastic noise)")
for _ in range(n_functions):
    curr_K = X @ X.T + 1e-8 * np.eye(n)
    y = mvn.rvs(mean=np.zeros(n), cov=curr_K)
    plt.plot(X, y)

## LOESS
plt.subplot(142)
plt.xlabel(r"$x$")
plt.title("LOESS")
for _ in range(n_functions):
    curr_K = X @ X.T + np.diag(1 / k_xstar_X.squeeze())
    y = mvn.rvs(mean=np.zeros(n), cov=curr_K)
    plt.plot(X, y)


## NW
plt.subplot(143)
plt.xlabel(r"$x$")
plt.title("Nadaraya-Watson")
for _ in range(n_functions):
    curr_K = np.ones((n, n)) + np.diag(1 / k_xstar_X.squeeze())
    y = mvn.rvs(mean=np.zeros(n), cov=curr_K)
    plt.plot(X, y)


## GP
plt.subplot(144)
plt.xlabel(r"$x$")
plt.title("GP")
for _ in range(n_functions):
    curr_K = K + noise_variance * np.eye(n)
    y = mvn.rvs(mean=np.zeros(n), cov=curr_K)
    plt.plot(X, y)

plt.tight_layout()
plt.savefig("./out/sampled_functions.png")
plt.show()
