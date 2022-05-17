import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal as mvn, norm
from scipy.linalg import sqrtm
import matplotlib.patches as patches

font = {"size": 30}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

np.random.seed(17)

limits = [-10, 10]
n = 50
noise_variance = 5 * 1e-1
X = np.linspace(*limits, n).reshape(-1, 1)
X_with_intercept = np.concatenate([X, np.ones((n, 1))], axis=1)
kernel = RBF()
K = kernel(X, X) + np.eye(n) * noise_variance
Y = mvn.rvs(mean=np.zeros(n), cov=K)
xstar = np.ones(1).reshape(-1, 1) * 0.01
K_xstarX = kernel(xstar, X)


plt.figure(figsize=(14, 10))
LINEWIDTH = 3

ntest = 250
Xtest = np.linspace(limits[0] - 10, limits[1] + 10, ntest).reshape(-1, 1)
Xtest_with_intercept = np.concatenate([Xtest, np.ones((ntest, 1))], axis=1)

plt.subplot(221)

## Loess
preds_loess = np.zeros(ntest)
for ii in range(ntest):
    xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    K_xstarX = kernel(xstar, X)
    W = np.diag(K_xstarX.squeeze())
    beta = np.linalg.solve(
        X_with_intercept.T @ W @ X_with_intercept, X_with_intercept.T @ W @ Y
    )
    preds_loess[ii] = np.array([xstar.squeeze(), 1.0]) @ beta

plt.scatter(X, Y, color="black", alpha=0.5, s=50)
plt.scatter(X[-1], Y[-1], color="orange", s=50)
plt.scatter(X[-2], Y[-2], color="blue", s=50)
plt.plot(Xtest, preds_loess, color="black")
# plt.xlim([0, limits[1] + 10])
# plt.ylim([-10, 10])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

rect = patches.Rectangle(
    (0, -5), limits[1] + 10, 10, linewidth=1, edgecolor="r", facecolor="none"
)
plt.gca().add_patch(rect)


plt.subplot(222)


## Loess
preds_loess = np.zeros(ntest)
for ii in range(ntest):
    xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    K_xstarX = kernel(xstar, X)
    W = np.diag(K_xstarX.squeeze())
    beta = np.linalg.solve(
        X_with_intercept.T @ W @ X_with_intercept, X_with_intercept.T @ W @ Y
    )
    preds_loess[ii] = np.array([xstar.squeeze(), 1.0]) @ beta

plt.scatter(X, Y, color="black", alpha=0.5, s=200)
plt.scatter(X[-1], Y[-1], color="orange", s=200)
plt.scatter(X[-2], Y[-2], color="blue", s=200)
plt.plot(Xtest, preds_loess, color="black")
plt.xlim([0, limits[1] + 10])
plt.ylim([-5, 5])
plt.xlabel(r"$x$")


plt.subplot(223)

tmp = Y[-1].copy()
Y[-1] = Y[-2].copy()
Y[-2] = tmp

## Loess
preds_loess = np.zeros(ntest)
for ii in range(ntest):
    xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    K_xstarX = kernel(xstar, X)
    W = np.diag(K_xstarX.squeeze())
    beta = np.linalg.solve(
        X_with_intercept.T @ W @ X_with_intercept, X_with_intercept.T @ W @ Y
    )
    preds_loess[ii] = np.array([xstar.squeeze(), 1.0]) @ beta

plt.scatter(X, Y, color="black", alpha=0.5, s=50)
plt.scatter(X[-1], Y[-1], color="blue", s=50)
plt.scatter(X[-2], Y[-2], color="orange", s=50)
plt.plot(Xtest, preds_loess, color="black")
# plt.xlim([0, limits[1] + 10])
# plt.ylim([-10, 10])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

rect = patches.Rectangle(
    (0, -5), limits[1] + 10, 10, linewidth=1, edgecolor="r", facecolor="none"
)
plt.gca().add_patch(rect)


plt.subplot(224)

# tmp = Y[-1].copy()
# Y[-1] = Y[-2].copy()
# Y[-2] = tmp

## Loess
preds_loess = np.zeros(ntest)
for ii in range(ntest):
    xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    K_xstarX = kernel(xstar, X)
    W = np.diag(K_xstarX.squeeze())
    beta = np.linalg.solve(
        X_with_intercept.T @ W @ X_with_intercept, X_with_intercept.T @ W @ Y
    )
    preds_loess[ii] = np.array([xstar.squeeze(), 1.0]) @ beta

plt.scatter(X, Y, color="black", alpha=0.5, s=200)
plt.scatter(X[-1], Y[-1], color="blue", s=200)
plt.scatter(X[-2], Y[-2], color="orange", s=200)
plt.plot(Xtest, preds_loess, color="black")
plt.xlim([0, limits[1] + 10])
plt.ylim([-5, 5])
plt.xlabel(r"$x$")

plt.tight_layout()
plt.savefig("./out/loess_edge_pathology.png")

plt.show()
import ipdb

ipdb.set_trace()
