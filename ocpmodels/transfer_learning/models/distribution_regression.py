from abc import ABC, abstractmethod

import torch
import torch.linalg as LA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from torch import Tensor


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
        self.X_ = X
        self.y_ = y

        k = self.kernel(X, X)
        #klmbda = k + torch.eye(k.shape[0]) * self.lmbda
        # Below is the same as above but avoids the creation of a new tensor on a different device
        klmbda = (k - k.diag().diag()) + (self.lmbda + k.diag()).diag()
        self.k_ = k
        self.alpha_ = LA.solve(klmbda, y)
        return self

    def predict(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 3
        k = self.kernel(X, self.X_)
        return k @ self.alpha_

    def predict_y_and_grad(self, X: Tensor, pos: Tensor) -> Tensor:
        assert len(X.shape) == 3
        T, num_atoms, d = X.shape
        self.X_.requires_grad_(True)
        k = self.kernel(X, self.X_)
        y_pred = k @ self.alpha_
        grad_pred = (
            torch.autograd.grad(
                y_pred,
                pos,
                grad_outputs=torch.ones_like(y_pred),
                create_graph=False,
                allow_unused=False,
            )[0]
        ).reshape(T, num_atoms, -1)
        return y_pred, grad_pred

    def score(self, X: Tensor, y: Tensor) -> float:
        y_pred = self.predict(X)
        return float(mean_squared_error(y, y_pred))

def median_heuristic(x: Tensor, y: Tensor) -> float:
    return float(torch.median(torch.cdist(x, y, p=2)))


# Sklearn utilities
class TorchStandardScaler:
    """Standardization for torch"""
    def __init__(self, eps=1e-7):
        self.eps = eps

    def fit(self, x):
        self.mean_ = x.mean(0, keepdim=True)
        self.std_ = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean_
        x /= (self.std_ + self.eps)
        return x

    def inverse_transform(self, x):
        x *= (self.std_ + self.eps)
        x += self.mean_
        return x

class StandardizedOutputRegression(BaseEstimator, RegressorMixin):
    """Wrapper class which standardizes the output of a univariate regression model (torch)

    Parameters:
        ------------
        regressor: object
            The regression model to be wrapped.

        eps: float, default=1e-7
            A small number to be added to the standard deviation when dividing to avoid division
            by zero.

    Methods:
        ---------
        fit(self, X, y):
            Fit the standardized regression model to the training data.

        predict(self, X):
            Predict using the standardized regression model.

        Returns: y_pred
        --------
    """

    def __init__(self, regressor, eps=1e-7):
        self.regressor = regressor
        self.scaler = TorchStandardScaler(eps)

    def fit(self, X, y):
        assert len(y.shape) == 2 and y.shape[1] == 1, "y should have shape 2 and be univariate"
        self.scaler.fit(y)
        y = self.scaler.transform(y)
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        return self.scaler.inverse_transform(self.regressor.predict(X))

    def predict_y_and_grad(self, X: Tensor, pos: Tensor) -> Tensor:
        y_pred, grad_pred = self.regressor.predict_y_and_grad(X, pos)
        y_pred = self.scaler.inverse_transform(y_pred)
        grad_pred = self.scaler.std_.item() * grad_pred
        return y_pred, grad_pred
