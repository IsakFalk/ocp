#!/usr/bin/env python3

"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.data.batch import DataBatch

from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    load_xyz_to_pyg_batch,
)
from ocpmodels.transfer_learning.loaders import BaseLoader
from ocpmodels.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    KernelGroupEmbeddingRidgeRegression,
    KernelMeanEmbeddingRidgeRegression,
    LinearGroupEmbeddingKernel,
    LinearMeanEmbeddingKernel,
    StandardizedOutputRegression,
    median_heuristic,
)

### Load checkpoint
CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

### Load data
DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

representation_layer = 3
base_loader = BaseLoader(
    checkpoint["config"],
    representation=True,
    representation_kwargs={
        "representation_layer": representation_layer,
    },
)
base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
model = base_loader.model
model.regress_forces = False
model.to(device)

# Kernels
from abc import ABC, abstractmethod

from torch import Tensor


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass


class EmbeddingKernel:
    def __init__(self, kernel: Kernel, alpha: int = 0.0, aggregation: str = "mean"):
        self.kernel = kernel
        self.alpha = alpha
        self.aggregation = aggregation  # dispatch on this: mean, sum

    def __call__(self, x: Tensor, zx: Tensor, y: Tensor, zy: Tensor) -> Tensor:
        return self._embedding_kernel(x, zx, y, zy)

    def _embedding_kernel(self, x: Tensor, zx: Tensor, y: Tensor, zy: Tensor) -> Tensor:
        t, n, d = x.shape
        l, m, d = y.shape  # noqa
        assert zx.shape[:2] == (t, n)
        assert zy.shape[:2] == (l, m)
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y)
        # The embedding kernel is a (1-a) * agg_c * K + a * group_c * K

        # Linear algebra magic: We make the mask needed for the group embedding kernel
        # essentially we vectorize the coefficient matrix which for each pair x_t, y_l of systems
        # has value B_ij = (alpha - 1) * c0^2 + alpha * delta(zx_i, zy_j) c_zx_i * c_zy_j
        # where c_zx_i and c_zy_j are the number of points in the point clouds of x_t and y_l if we do mean embedding
        # or just 1 if we do sum embedding
        zx = zx.reshape(t * n, -1)
        zy = zy.reshape(l * m, -1)

        delta = (zx[:, None] == zy[None, :]).squeeze()

        agg_c = 1.0
        if self.aggregation == "mean":
            agg_c /= n * m
            # Get all groups and possibly calculate the number
            groups = torch.unique(torch.cat([zx, zy]))
            group_nx = (zx[:, None] == groups[None, :]).reshape(t, n, -1).sum(axis=1)
            group_ny = (zy[:, None] == groups[None, :]).reshape(l, m, -1).sum(axis=1)
            select_zx = (zx[:, None] == groups[None, :]).reshape(t, n, -1)
            select_zy = (zy[:, None] == groups[None, :]).reshape(l, m, -1)
            group_nx_flatten = (select_zx * group_nx[:, None, :]).reshape(t * n, -1).sum(axis=1)
            group_ny_flatten = (select_zy * group_ny[:, None, :]).reshape(l * m, -1).sum(axis=1)
            # Normalize by the number of kernels in the group \sum_s K_s / S
            mask = (1.0 / group_nx_flatten[:, None]) * (1.0 / group_ny_flatten[None, :])
        elif self.aggregation == "sum":
            mask = 1.0
        else:
            raise ValueError(f"Unknown aggregation {self.aggregation}")
        group_c = mask * delta

        k = (k0 * ((1 - self.alpha) * agg_c + self.alpha * group_c)).reshape(t, n, l, m).sum(axis=(1, 3))
        return k


with torch.no_grad():
    phi = model(data_batch).cpu()
    d = phi.shape[-1]
    sigma = median_heuristic(phi.reshape(-1, d), phi.reshape(-1, d))

t = 100
l = 2
n = num_atoms
m = num_atoms
batch_x = Batch.from_data_list(data_batch[:t])
batch_y = Batch.from_data_list(data_batch[:l])
x = model(batch_x).reshape(t, num_atoms, -1)
zx = batch_x.atomic_numbers.reshape(t, num_atoms, -1)
y = model(batch_y).reshape(l, num_atoms, -1)
zy = batch_y.atomic_numbers.reshape(l, num_atoms, -1)

alpha = 1.0
gk = GaussianKernel(sigma)
ek = EmbeddingKernel(gk, alpha=alpha, aggregation="sum")

from sklearn.preprocessing import KernelCenterer

k_xx_1 = ek(x, zx, x, zx).detach().cpu()
k_xx_1_centered = KernelCenterer().fit_transform(k_xx_1)

alpha = 0.0
gk = GaussianKernel(sigma)
ek = EmbeddingKernel(gk, alpha=alpha, aggregation="sum")

k_xx_0 = ek(x, zx, x, zx).detach().cpu()
k_xx_0_centered = KernelCenterer().fit_transform(k_xx_0)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
im = axs[0].imshow(k_xx_0_centered, cmap="viridis")
axs[0].set_title(f"Kernel with alpha={alpha:.1f}")
fig.colorbar(im, ax=axs[0])
im = axs[1].imshow(k_xx_1_centered, cmap="viridis")
axs[1].set_title(f"Kernel with alpha={alpha:.1f}")
fig.colorbar(im, ax=axs[1])
fig.tight_layout()
fig.savefig("kernels.png")

k_xx_0_centered / k_xx_1_centered
k_xx_0_centered


import itertools

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
alpha_values = np.linspace(0.0, 1.0, 11)

for i, j in itertools.product(range(5), range(2)):
    ax = axs[i][j]
    alpha = alpha_values[i * 2 + j]
    gk = GaussianKernel(sigma)
    ek = EmbeddingKernel(gk, alpha=alpha, aggregation="mean")
    k_xx = ek(x, zx, x, zx).detach().cpu()
    im = ax.imshow(k_xx, cmap="viridis")
    ax.set_title(f"Kernel with alpha={alpha:.1f}")
    fig.colorbar(im, ax=ax)

fig.tight_layout()
fig.savefig("kernels_mean.png")


class GroupEmbeddingKernel(ABC):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, x, y, **kwargs):
        pass

    def _group_embedding_kernel(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        assert x.shape[:2] == freq_x.shape
        assert y.shape[:2] == freq_y.shape
        t, n, d = x.shape
        l, m, d = y.shape  # noqa
        x = x.reshape(t * n, d)
        y = y.reshape(l * m, d)
        k0 = self.kernel(x, y)  # t*n x l*m
        # We weigh each point by the inverse of the number of atoms of
        # that species in each frame
        freq_x = freq_x.reshape(t * n)
        freq_y = freq_y.reshape(l * m)
        k0 = k0 * torch.outer(freq_x, freq_y)
        k = k0.reshape(t, n, l, m).sum(axis=(1, 3))
        return k


class GaussianKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        D2 = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-D2 / (2 * self.sigma**2))


# Kernel Mean Embeddings
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
        k = kxx + kyy - 2 * kxy  # this is like ||x-y||^2 vectorized
        return torch.exp(-k / (2 * self.sigma**2))


# Group Kernel Embeddings
class LinearGroupEmbeddingKernel(GroupEmbeddingKernel):
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def __call__(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        return self._group_embedding_kernel(x, y, freq_x, freq_y)


class GaussianGroupEmbeddingKernel(GroupEmbeddingKernel):
    def __init__(self, kernel: Kernel, sigma=1.0):
        self.kernel = kernel
        self.sigma = sigma

    def __call__(self, x: Tensor, y: Tensor, freq_x: Tensor, freq_y: Tensor) -> Tensor:
        kxx = self._group_embedding_kernel(x, x, freq_x, freq_x).diag().reshape(-1, 1)
        kyy = self._group_embedding_kernel(y, y, freq_y, freq_y).diag().reshape(1, -1)
        kxy = self._group_embedding_kernel(x, y, freq_x, freq_y)
        k = kxx + kyy - 2 * kxy  # this is like ||x-y||^2 vectorized
        return torch.exp(-k / (2 * self.sigma**2))


#####


def get_freq_tensor(data_batch):
    """Get the frequency of each species"""
    group_id = torch.unique(data_batch.atomic_numbers)
    frame_atomic_mask = torch.stack([data_batch[0].atomic_numbers == k for k in group_id])
    num_atoms_per_species = frame_atomic_mask.sum(dim=1)
    freq_atoms_per_species = 1 / num_atoms_per_species

    freq_idx = torch.zeros_like(data_batch.atomic_numbers)
    for id, group_freq in enumerate(freq_atoms_per_species):
        freq_idx[data_batch.atomic_numbers == group_id[id]] = group_freq
    return freq_idx


# Get multi-weight kernel
data = data_batch
with torch.no_grad():
    h = model(data).cpu()
    d = h.shape[-1]
    sigma = median_heuristic(h.reshape(-1, d), h.reshape(-1, d))

gk = GaussianKernel()
gk.sigma = sigma
lingroupk = LinearGroupEmbeddingKernel(gk)
freq = get_freq_tensor(data_batch)
freq = torch.ones_like(freq)
h = h.reshape(num_frames, num_atoms, d)
freq = freq.reshape(num_frames, num_atoms)
k = lingroupk(h, h, freq, freq)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(k)
fig.savefig("k_new.png")
k = gk(h, h)


# Get the frequency of each species
group_id = torch.unique(data_batch.atomic_numbers)
frame_atomic_mask = torch.stack([data_batch[0].atomic_numbers == k for k in group_id])
num_atoms_per_species = frame_atomic_mask.sum(dim=1)
freq_atoms_per_species = 1 / num_atoms_per_species
