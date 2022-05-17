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

np.random.seed(16)

limits = [-5, 5]
n = 400
noise_variance = 0.1
X = np.linspace(*limits, n).reshape(-1, 1)
X_with_intercept = np.concatenate([X, np.ones((n, 1))], axis=1)
kernel = RBF()
K = kernel(X, X) + np.eye(n) * noise_variance
Y = mvn.rvs(mean=np.zeros(n), cov=K)
xstar = np.ones(1).reshape(-1, 1) * 0.001
K_xstarX = kernel(xstar, X)


smoother_linear_model = (
    X.T @ np.linalg.solve(X @ X.T + 1e-8 * np.eye(n), np.eye(n))
).squeeze()

W = np.diag(K_xstarX.squeeze())


smoother_gp = (K_xstarX @ np.linalg.solve(K, np.eye(n))).squeeze()
smoother_nw = K_xstarX.squeeze() / np.sum(K_xstarX)

# import ipdb; ipdb.set_trace()
K = kernel(X, X) * (X @ X.T) + np.eye(n) * noise_variance
K_xstarX = kernel(xstar, X) * (xstar @ X.T)
smoother_gp2 = (K_xstarX @ np.linalg.solve(K, np.eye(n))).squeeze()


plt.figure(figsize=(18, 10))
plt.subplot(231)
plt.plot(X.squeeze(), smoother_linear_model, color="gray", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title("Linear model")
# plt.xlabel(r"$x$")
plt.xticks([0], labels=[r"$x_\star$"])
plt.ylabel(r"$h_i$")
plt.yticks([0])

plt.subplot(232)
smoother_loess = np.linalg.solve(X.T @ W @ X + 1e-8 * np.eye(1), X.T @ W).squeeze()
plt.plot(X.squeeze(), smoother_loess, color="gray", linewidth=3)
# smoother_loess = np.linalg.solve(X.T @ W ** 2 @ X + 1e-8 * np.eye(1), X.T @ W).squeeze()
# plt.plot(X.squeeze(), smoother_loess, color="red", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title(r"LOESS, $\alpha = 1$")
# plt.xlabel(r"$x$")
plt.yticks([0])
plt.xticks([0], labels=[r"$x_\star$"])

plt.subplot(233)
smoother_loess_snapless1 = smoother_loess.copy()
smoother_loess_snapless1[np.abs(X.squeeze()) > 2] = 0
plt.plot(X.squeeze(), smoother_loess_snapless1, color="gray", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title(r"LOESS, $\alpha < 1$")
# plt.xlabel(r"$x$")
plt.xticks([0], labels=[r"$x_\star$"])
plt.yticks([0])

plt.subplot(234)
plt.plot(X.squeeze(), smoother_nw, color="gray", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title("Nadaraya-Watson")
# plt.xlabel(r"$x$")
plt.xticks([0], labels=[r"$x_\star$"])
plt.ylabel(r"$h_i$")
plt.yticks([0])


plt.subplot(235)
plt.plot(X.squeeze(), smoother_gp, color="gray", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title(r"GP")  # , $k_{\textsc{rbf}}$")
# plt.xlabel(r"$x$")
plt.xticks([0], labels=[r"$x_\star$"])
plt.yticks([0])

# plt.subplot(246)
# plt.plot(X.squeeze(), smoother_gp2, color="gray", linewidth=3)
# plt.axvline(xstar.squeeze(), color="black", linestyle="--")
# plt.title(r"GP, $k_{\textsc{rbf}} k_{\textsc{lin}}$")
# plt.xlabel(r"$x$")
# plt.yticks([])

plt.subplot(236)
smoother_nngp = smoother_gp.copy()
smoother_nngp[np.abs(X.squeeze()) > 2] = 0
plt.plot(X.squeeze(), smoother_nngp, color="gray", linewidth=3)
plt.axvline(xstar.squeeze(), color="black", linestyle="--")
plt.title(r"NNGP")  # , $k_{\textsc{rbf}}$")
# plt.xlabel(r"$x$")
plt.xticks([0], labels=[r"$x_\star$"])
plt.yticks([0])

# plt.subplot(248)
# beta = 0.9
# sigma2 = 1.0
# K = beta ** (np.abs(X - X.T)) * sigma2 / (1 - beta ** 2)
# K_xstarX = beta ** (np.abs(xstar - X.T)) * sigma2 / (1 - beta ** 2)
# smoother_ar1 = (K_xstarX @ np.linalg.solve(K, np.eye(n))).squeeze()
# plt.plot(X.squeeze(), smoother_ar1, color="gray", linewidth=3)
# plt.axvline(xstar.squeeze(), color="black", linestyle="--")
# plt.title("AR(1)")
# plt.xlabel(r"$x$")

plt.tight_layout()
plt.savefig("./out/linear_smoother_hat_matrix.png")
plt.show()
plt.close()

import ipdb

ipdb.set_trace()
# plt.plot(((xstar @ X.T) * K_xstarX).squeeze())
# plt.show()


plt.figure(figsize=(12, 5))
LINEWIDTH = 3

ntest = 200
Xtest = np.linspace(limits[0] - 5, limits[1] + 5, ntest).reshape(-1, 1)
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
    # preds_loess[ii] = xstar.T @ beta
    preds_loess[ii] = np.array([xstar, 1.0]) @ beta

plt.scatter(X, Y, color="black", alpha=0.5)
plt.plot(Xtest, preds_loess, label="LOESS", linewidth=LINEWIDTH)

## GP
K_xstarX = kernel(Xtest, X)
preds_gp = K_xstarX @ np.linalg.solve(K, Y)
plt.plot(Xtest, preds_gp, label=r"GP, $k_{\textsc{rbf}}$", linewidth=LINEWIDTH)

## GP with multiplicative kernel
K = kernel(X, X) * (X_with_intercept @ X_with_intercept.T)
K_xstarX = kernel(Xtest, X) * (Xtest_with_intercept @ X_with_intercept.T)
preds_gp_mult = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
# preds_gp_mult = K @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
plt.plot(
    Xtest,
    preds_gp_mult,
    label=r"GP, $k_{\textsc{rbf}} k_{\textsc{lin}}$",
    linewidth=LINEWIDTH,
)

## AR(1) kernel
beta = 0.9
sigma2 = 1.0
K = beta ** (np.abs(X - X.T)) * sigma2 / (1 - beta ** 2)
K_xstarX = beta ** (np.abs(Xtest - X.T)) * sigma2 / (1 - beta ** 2)
preds_gp_ar1 = K_xstarX @ np.linalg.solve(K + np.eye(n) * noise_variance, Y)
plt.plot(Xtest, preds_gp_ar1, label="AR(1)", linewidth=LINEWIDTH)
# import ipdb; ipdb.set_trace()

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xticks([])
plt.yticks([])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
plt.savefig("./out/smoother_predictions.png")
plt.show()


import ipdb

ipdb.set_trace()
