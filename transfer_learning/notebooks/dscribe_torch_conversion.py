"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
import copy
import logging
import random
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import quippy
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ocpmodels.common.utils import save_checkpoint
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler
from ocpmodels.transfer_learning.common.logger import WandBLogger
from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    aggregate_metric,
    load_xyz_to_pyg_batch,
    load_xyz_to_pyg_data,
    torch_tensor_to_npy,
)
from ocpmodels.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    KernelMeanEmbeddingRidgeRegression,
    LinearMeanEmbeddingKernel,
    median_heuristic,
)

### Load data
DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_batch = data_batch.to(device)

# quippy, tutorial https://libatoms.github.io/GAP/quippy-descriptor-tutorial.html
from quippy.descriptors import Descriptor

Z = np.unique(raw_data[0].get_atomic_numbers())
nZ = len(Z)
Zstring = " ".join([str(z) for z in Z])
Zstring = "{" + Zstring + "}"
Zstring = f"soap l_max=5 n_max=2 cutoff=1.0 atom_sigma=0.5 nZ={nZ} Z{Zstring}"
# Zstring = f"distance_2b cutoff=6.0 nZ={nZ} Z{Zstring}"
descriptor = Descriptor(Zstring)
desc = descriptor.calc(raw_data[0][:3], grad=True)
available_keys = [
    "has_grad_data",
    "ii",
    "pos",
    "grad_covariance_cutoff",
    "covariance_cutoff",
    "data",
    "has_data",
    "grad_data",
    "ci",
    "grad_index_0based",
]

pprint(desc["grad_data"])
desc["grad_data"].shape
desc["data"].shape
desc["data"]
desc["pos"]
desc["grad_index_0based"]
descriptor.get_n_perm()

atoms = raw_data[0][:10]
Zstring = " ".join([str(z) for z in Z])
Zstring = "{" + Zstring + "}"
Zstring = f"soap l_max=5 n_max=2 cutoff=6.0 atom_sigma=0.5 nZ={nZ} Z{Zstring}"
print(Zstring)
descriptor = Descriptor(Zstring)
descriptor.sizes(atoms)  # n_descriptors, n_cross
desc = descriptor.calc(atoms, grad=True)
desc["grad_data"].shape
desc["grad_index_0based"]
desc["ci"]
# Column 0 is the descriptor index (for SOAP coincides with the atom index)
# Column 1 is the index of neighbouring atom the descriptor was differentiated with respect to
desc["grad_index_0based"]
k = 1
ij = desc["grad_index_0based"][k]
desc["pos"].shape
pos = atoms.get_positions()
desc["pos"][k]
desc["pos"]


# Gather indicies
i_to_j = {}
for ij in desc["grad_index_0based"]:
    i, j = ij
    if i not in i_to_j:
        i_to_j[i] = []
    i_to_j[i].append(j)

i_to_grad = {}
for k, grad in enumerate(desc["grad_data"]):
    i, j = desc["grad_index_0based"][k]
    if i not in i_to_grad:
        i_to_grad[i] = []
    i_to_grad[i].append(grad)

i_to_j
i_to_grad[0][0]

# Create the Jacobian (May want to consider jvp since this is also something that they can use in the torch things)
import torch

atoms = raw_data[0][:5]  # Small object of atoms
# Set up descriptor
Zstring = " ".join([str(z) for z in Z])
Zstring = "{" + Zstring + "}"
Zstring = f"soap l_max=5 n_max=2 cutoff=6.0 atom_sigma=0.5 nZ={nZ} Z{Zstring}"
descriptor = Descriptor(Zstring)
descriptor.sizes(atoms)  # n_descriptors, n_cross
desc = descriptor.calc(atoms, grad=True)

# Gather indicies
grad_idx = desc["grad_index_0based"].astype(int)
grad_idx = grad_idx.transpose()
grad_data = desc["grad_data"].astype(float)
grad_idx = torch.tensor(grad_idx)
grad_data = torch.tensor(grad_data)
grad_data.shape

sparse_grad = torch.sparse_coo_tensor(grad_idx, values=grad_data)
grad = sparse_grad.to_dense()


# Feature Descriptor Class
class TorchSOAP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atoms, Zstring):
        descriptor = Descriptor(Zstring)
        desc = descriptor.calc(atoms, grad=True)
        # Gather indicies
        grad_idx = desc["grad_index_0based"].astype(int)
        grad_idx = grad_idx.transpose()
        grad_data = desc["grad_data"].astype(float)
        grad_idx = torch.tensor(grad_idx)
        grad_data = torch.tensor(grad_data)
        sparse_grad = torch.sparse_coo_tensor(grad_idx, values=grad_data)
        grad = sparse_grad.to_dense()
        ctx.save_for_backward(grad)
        return torch.tensor(desc["data"].astype(float))

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad_output @ grad, None
