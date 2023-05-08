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

# freq_idx = torch.zeros_like(data_batch.atomic_numbers)
# for id, group_freq in enumerate(freq_atoms_per_species):
#     freq_idx[data_batch.atomic_numbers == group_id[id]] = group_freq

freq_idx = torch.ones_like(data_batch.atomic_numbers)
freq_mask = torch.outer(freq_idx, freq_idx)

k_reweighted = k * freq_mask
k_reweighted = k_reweighted.reshape(num_frames, num_atoms, num_frames, num_atoms).sum(axis=(1, 3))
k = k.reshape(num_frames, num_atoms, num_frames, num_atoms).mean(axis=(1, 3))

fig, ax = plt.subplots()
ax.imshow(k_reweighted)
fig.savefig("k_reweighted.png")

fig, ax = plt.subplots()
ax.imshow(k)
fig.savefig("k.png")

# Get data
y = data_batch.y.float().reshape(-1, 1)
grad = -data_batch.force.reshape(-1, num_atoms, 3)
train_idx, test_idx = train_test_split(np.arange(num_frames), test_size=0.2)
data_train = Batch.from_data_list(data_batch[train_idx]).to(device)
data_test = Batch.from_data_list(data_batch[test_idx]).to(device)
y_train, y_test = y[train_idx].reshape(-1, 1), y[test_idx].reshape(-1, 1)
grad_train, grad_test = grad[train_idx], grad[test_idx]

# Median heuristics sigma
with torch.no_grad():
    phi_train = model(data_train).cpu()
    d = phi_train.shape[-1]
    sigma = median_heuristic(phi_train.reshape(-1, phi_train.shape[-1]), phi_train.reshape(-1, phi_train.shape[-1]))

for lmbda in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    print(f"Lambda: {lmbda}")
    y_train, y_test = y[train_idx].reshape(-1, 1), y[test_idx].reshape(-1, 1)
    # Estimator
    gk = GaussianKernel()
    gk.sigma = sigma
    lingroupk = LinearGroupEmbeddingKernel(gk)
    gkgerr = KernelGroupEmbeddingRidgeRegression(lingroupk, lmbda=lmbda)

    # Estimator
    gk = GaussianKernel()
    gk.sigma = sigma
    gklme = LinearMeanEmbeddingKernel(gk)
    gkmerr = KernelMeanEmbeddingRidgeRegression(gklme, lmbda=lmbda)

    phi_train = phi_train.reshape(-1, num_atoms, d)
    y_train_mean = y_train.mean()
    y_train = y_train - y_train_mean
    # Fit models
    gkmerr.fit(phi_train, y_train)
    freq_train = get_freq_tensor(data_train).reshape(*phi_train.shape[:2])
    gkgerr.fit(phi_train, y_train, freq_train)

    # Predict
    data_test.pos.requires_grad = True
    phi_test = model(data_test).reshape(-1, num_atoms, d)
    # y_pred = gkmerr.predict(phi_test)
    y_pred, grad_pred = gkmerr.predict_y_and_grad(phi_test, data_test.pos)
    y_pred += y_train_mean
    print("gkmerr")
    with torch.no_grad():
        print(mean_squared_error(y_test, y_pred))

    print(mean_squared_error(grad_pred.reshape(-1, 3), grad_test.reshape(-1, 3)))

    data_test.pos.requires_grad = True
    phi_test = model(data_test).reshape(-1, num_atoms, d)

    freq_test = get_freq_tensor(data_test).reshape(*phi_test.shape[:2])
    # y_pred = gkgerr.predict(phi_test, freq_test)
    y_pred, grad_pred = gkgerr.predict_y_and_grad(phi_test, data_test.pos, freq_test)
    y_pred += y_train_mean
    print("gkgerr")
    with torch.no_grad():
        print(mean_squared_error(y_test, y_pred))

    print(mean_squared_error(grad_pred.reshape(-1, 3), grad_test.reshape(-1, 3)))
