"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data

from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    load_xyz_to_pyg_batch,
    load_xyz_to_pyg_data,
)
from ocpmodels.transfer_learning.loaders import BaseLoader
from ocpmodels.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    KernelMeanEmbeddingRidgeRegression,
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
raw_data, data_list, num_frames, num_atoms = load_xyz_to_pyg_data(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

representation_layer = 1
base_loader = BaseLoader(
    checkpoint["config"],
    representation=False,
    representation_kwargs={
        "representation_layer": representation_layer,
    },
)
base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
model = base_loader.model
model.to(device)
model.mekrr_forces = False


### First create random feature kernel
class RFFFeatureMap(torch.nn.Module):
    def __init__(self, D, d, sigma):
        super(RFFFeatureMap, self).__init__()
        self.D = D
        self.d = d
        self.sigma = sigma
        self.w = self._sample_w()

    def _sample_w(self):
        return torch.randn(self.D, self.d)

    def forward(self, x):
        q = torch.matmul(x, self.w.T)
        zcos = torch.cos(q / self.sigma)
        zsin = torch.sin(q / self.sigma)
        z = math.sqrt(2 / self.D) * torch.cat([zcos, zsin], dim=-1)
        return z


class MERFFGNN(torch.nn.Module):
    def __init__(self, D, d, sigma, model, regress_forces=True):
        super(MERFFGNN, self).__init__()
        self.D = D
        self.d = d
        self.sigma = sigma
        self.gnn = model
        self.feature_map = RFFFeatureMap(D, d, sigma)
        self.w = nn.Linear(2 * D, 1, bias=False)  # 2 * D because of concatenation of sin and cos
        self.regress_forces = True

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        frames = len(torch.unique(data.batch))
        num_atoms = data.natoms[0]
        h = self.gnn(data)[0]
        d = h.shape[-1]
        mu = self.feature_map(h.reshape(frames, num_atoms, d)).mean(1)
        energy = self.w(mu)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy


# Median heuristics sigma
frames = 3
with torch.no_grad():
    phi = model(Batch.from_data_list(data_list[:frames]))[0].cpu()
    d = phi.shape[-1]
    sigma = median_heuristic(phi.reshape(-1, d), phi.reshape(-1, d))


from torch_geometric.data import DataLoader


def get_dataloader(list_of_data):
    loader = DataLoader(
        list_of_data,
        batch_size=5,
        num_workers=4,
        pin_memory=True,
    )
    return loader


loader = get_dataloader(data_list[:frames])
for batch in loader:
    print(batch)
    break

D = 1000
merffgnn = MERFFGNN(D, d, sigma, model, regress_forces=True)
merffgnn.to(device)

energy, forces = merffgnn(batch)
forces.shape

energy, forces = model(batch)

merffgnn.to(device)
for p in merffgnn.parameters():
    p.requires_grad = False
merffgnn.w.weight.requires_grad = True

for name, param in merffgnn.named_parameters():
    print(name, param.requires_grad)
