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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# representation_layer = 3
base_loader = BaseLoader(
    checkpoint["config"],
    # representation=True,
    # representation_kwargs={
    #     "representation_layer": representation_layer,
    # },
)
base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
model = base_loader.model
model.to(device)

import re

freeze_layers_up_to = 3
freeze_layers = list(range(0, freeze_layers_up_to))
for name, param in model.named_parameters():
    try:
        if int(re.findall(r"[0-9]", name)[0]) in freeze_layers and not re.match(r"lin[0-9]\.*", name):
            param.requires_grad = False
    except IndexError:
        if "embedding" in name:
            param.requires_grad = False

for name, param in model.named_parameters():
    print(name, param.requires_grad)

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
    phi_train = model(data_train)[0]
    d = phi_train.shape[-1]
    sigma = median_heuristic(phi_train.reshape(-1, phi_train.shape[-1]), phi_train.reshape(-1, phi_train.shape[-1]))


# Estimator
gk = GaussianKernel()
gk.sigma = sigma
gklme = LinearMeanEmbeddingKernel(gk)
data_train.pos.requires_grad = True
g_ = gklme(phi_train.reshape(-1, num_atoms, d), phi_train.reshape(-1, num_atoms, d))


def f(data, pos):
    pos_ = pos.clone().detach()
    data.pos = pos
    data_ = data.clone().detach()
    data_.pos = pos_
    pos_.requires_grad = True
    phi = model(data)[0]
    phi_ = model(data_)[0]
    return gklme(phi.reshape(-1, num_atoms, d), phi_.reshape(-1, num_atoms, d))


g = f(data_train, data_train.pos)


phi = model(data_train)[0]
data_train.clone().detach().pos

data_train.pos


from torch.func import grad, hessian, jvp, vjp, vmap
