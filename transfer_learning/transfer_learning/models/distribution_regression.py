import torch
import torch.linalg as LA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from torch import Tensor

from abc import ABC, abstractmethod

# Kernels
class Kernel(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass

class MeanEmbeddingKernel(ABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, x, y):
        pass

    def _mean_embedding_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        t, n, d = x.shape
        l, m, d = y.shape # noqa
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y).reshape(t, n, l, m)  # t x n x l x m
        # to get the actual embedding kernel we have to sum over the point cloud axes
        k = k0.sum(axis=(1, 3)) / (n * m)
        return k

class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * self.sigma ** 2))


class LinearMeanEmbeddingKernel(MeanEmbeddingKernel):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return self._mean_embedding_kernel(x, y)


class GaussianMeanEmbeddingKernel(MeanEmbeddingKernel):
    def __init__(self, kernel: Kernel, sigma=1.0):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        kxx = self._mean_embedding_kernel(x, x).diag().reshape(-1, 1)
        kyy = self._mean_embedding_kernel(y, y).diag().reshape(1, -1)
        kxy = self._mean_embedding_kernel(x, y)
        k = kxx + kyy - 2 * kxy # this is like ||x-y||^2 vectorized
        return torch.exp(-k / (2 * self.sigma ** 2))

# Estimators
### Create distribution regression
### Output is assumed to be in \R and
### we have T snapshots, indexed by t, and the system is
### described by N atoms, indexed by i.
### This means that the kernel (when using some pointwise kernel K)
### is of size T x T
### Call this gram matrix K, then
### K_{t, l} = torch.sum(G^{t, l}) / N**2 where
### G^{t, l}_{i, j} = K(x^{t}_{i}, x^{l}_{j})
### Thus, the only thing we need to do is to build a kernel tensor
### of size T x T x N x N and then sum over the last two dimensions
### (or equivalently)

class KernelMeanEmbeddingRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, kernel: MeanEmbeddingKernel, lmbda: float = 1.0):
        self.kernel = kernel
        self.lmbda = lmbda

    def fit(self, X: Tensor, y: Tensor):
        assert len(X.shape) == 3
        assert len(y.shape) == 2
        self._X = X
        self._y = y

        k = self.kernel(X, X)
        klmbda = k + torch.eye(k.shape[0]) * self.lmbda
        self._k = k
        self._alpha = LA.solve(klmbda, y)
        return self

    def predict(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 3
        k = self.kernel(X, self._X)
        return k @ self._alpha

    def score(self, X: Tensor, y: Tensor) -> float:
        y_pred = self.predict(X)
        return float(mean_squared_error(y, y_pred))


def median_heuristic(x: Tensor, y: Tensor) -> float:
    return float(torch.median(torch.cdist(x, y, p=2)))
