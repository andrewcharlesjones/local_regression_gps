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

np.random.seed(17)

limits = [-10, 10]
n = 200
noise_variance = 1e-1
X = np.linspace(*limits, n).reshape(-1, 1)
X_with_intercept = np.concatenate([X, np.ones((n, 1))], axis=1)
kernel = RBF()
K = kernel(X, X) + np.eye(n) * noise_variance
Y = mvn.rvs(mean=np.zeros(n), cov=K)
xstar = np.ones(1).reshape(-1, 1) * 0.01
K_xstarX = kernel(xstar, X)


plt.figure(figsize=(20, 5))
LINEWIDTH = 3

ntest = 250
Xtest = np.linspace(limits[0] - 10, limits[1] + 10, ntest).reshape(-1, 1)
Xtest_with_intercept = np.concatenate([Xtest, np.ones((ntest, 1))], axis=1)

plt.subplot(121)

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
    # xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    # K_xstarX = kernel(xstar, X)
    # W = np.diag(K_xstarX.squeeze())
    # beta = np.linalg.solve(
    #     X.T @ W @ X, X.T @ W @ Y
    # )
    # preds_loess[ii] = xstar @ beta

# import ipdb

# ipdb.set_trace()

# preds_loess = (
#     kernel(Xtest, X) * (Xtest @ X.T) @ np.linalg.solve(kernel(X) * (X @ X.T) + noise_variance * np.eye(n), Y)
# )

plt.scatter(X, Y, color="black", alpha=0.5)
plt.plot(Xtest, preds_loess, label="LOESS", linewidth=LINEWIDTH)

## NW
K_xstarX = kernel(Xtest, X)
preds_nw = (K_xstarX * Y).sum(1) / K_xstarX.sum(1)
plt.plot(Xtest, preds_nw, label=r"NW", linewidth=LINEWIDTH)
# import ipdb; ipdb.set_trace()
# plt.show()

## GP
K_xstarX = kernel(Xtest, X)
preds_gp = K_xstarX @ np.linalg.solve(K, Y)
plt.plot(Xtest, preds_gp, label=r"GP", linewidth=LINEWIDTH)

# plt.show()
## GP with multiplicative kernel
# K = kernel(X, X) * (X_with_intercept @ X_with_intercept.T)
# K_xstarX = kernel(Xtest, X) * (Xtest_with_intercept @ X_with_intercept.T)
# preds_gp_mult = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# # preds_gp_mult = K @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# plt.plot(
#     Xtest,
#     preds_gp_mult,
#     label=r"GP, $k_{\textsc{rbf}} k_{\textsc{lin}}$",
#     linewidth=LINEWIDTH,
# )
# import ipdb; ipdb.set_trace()
## AR(1) kernel
# beta = 0.9
# sigma2 = 1.0
# K = beta ** (np.abs(X - X.T)) * sigma2 / (1 - beta ** 2)
# K_xstarX = beta ** (np.abs(Xtest - X.T)) * sigma2 / (1 - beta ** 2)
# preds_gp_ar1 = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# plt.plot(Xtest, preds_gp_ar1, label="AR(1)", linewidth=LINEWIDTH)

# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.xticks([])
# plt.yticks([])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
# plt.legend()

plt.subplot(122)

keep_idx = np.where(np.abs(X.squeeze()) > 2.0)[0]
X = X[keep_idx]
Y = Y[keep_idx]
X_with_intercept = X_with_intercept[keep_idx]
n = len(keep_idx)

# ntest = 200
Xtest = np.linspace(limits[0], limits[1], ntest).reshape(-1, 1)
Xtest_with_intercept = np.concatenate([Xtest, np.ones((ntest, 1))], axis=1)

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

    # import ipdb; ipdb.set_trace()
    # xstar = np.array([Xtest[ii]]).reshape(-1, 1)
    # K_xstarX = kernel(xstar, X)
    # W = np.diag(K_xstarX.squeeze())
    # preds_loess[ii] = (
    #     xstar.T
    #     @ X.T
    #     @ W
    #     @ np.linalg.solve(W @ X @ X.T @ W.T + noise_variance * np.eye(n), Y)
    # )

plt.scatter(X, Y, color="black", alpha=0.5)
plt.plot(Xtest, preds_loess, label="LOESS", linewidth=LINEWIDTH)

## NW
K_xstarX = kernel(Xtest, X)
preds_nw = (K_xstarX * Y).sum(1) / K_xstarX.sum(1)
plt.plot(Xtest, preds_nw, label=r"NW", linewidth=LINEWIDTH)

## GP
K = kernel(X, X) + np.eye(n) * noise_variance
K_xstarX = kernel(Xtest, X)
preds_gp = K_xstarX @ np.linalg.solve(K, Y)
plt.plot(Xtest, preds_gp, label=r"GP", linewidth=LINEWIDTH)
# import ipdb; ipdb.set_trace()

## GP with multiplicative kernel
# K = kernel(X, X) * (X_with_intercept @ X_with_intercept.T)
# K_xstarX = kernel(Xtest, X) * (Xtest_with_intercept @ X_with_intercept.T)
# preds_gp_mult = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# # preds_gp_mult = K @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# plt.plot(
#     Xtest,
#     preds_gp_mult,
#     label=r"GP, $k_{\textsc{rbf}} k_{\textsc{lin}}$",
#     linewidth=LINEWIDTH,
# )

## AR(1) kernel
# beta = 0.9
# sigma2 = 1.0
# K = beta ** (np.abs(X - X.T)) * sigma2 / (1 - beta ** 2)
# K_xstarX = beta ** (np.abs(Xtest - X.T)) * sigma2 / (1 - beta ** 2)
# preds_gp_ar1 = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# plt.plot(Xtest, preds_gp_ar1, label="AR(1)", linewidth=LINEWIDTH)


plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.xticks([])
# plt.yticks([])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
plt.savefig("./out/smoother_predictions.png")
plt.show()

K_xstarX = kernel(Xtest, X)
import ipdb

ipdb.set_trace()
