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

from transfer_learning.transfer_learning.common.utils import load_xyz_to_pyg_batch, ATOMS_TO_GRAPH_KWARGS
from ocpmodels.transfer_learning.loaders import BaseLoader
from transfer_learning.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    LinearMeanEmbeddingKernel,
    KernelMeanEmbeddingRidgeRegression,
    median_heuristic,
    StandardizedOutputRegression,
)


### Load checkpoint
CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

### Load data
DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

representation_layer = 1
base_loader = BaseLoader(
    checkpoint["config"],
    representation=True,
    representation_kwargs={
        "representation_layer": representation_layer,
    },
)
base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
model = base_loader.model
model.to(device)

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


# Estimator
gk = GaussianKernel()
gk.sigma = sigma
gklme = LinearMeanEmbeddingKernel(gk)
gkmerr = StandardizedOutputRegression(regressor=KernelMeanEmbeddingRidgeRegression(gklme, lmbda=1e-6))

# Get data in the right shape
phi_train = model(data_train).reshape(-1, num_atoms, d)
# Fit model
gkmerr.fit(phi_train, y_train)
# Predict
data_test.pos.requires_grad = True
phi_test = model(data_test).reshape(-1, num_atoms, d)
y_pred = gkmerr.predict(phi_test)
y_pred, grad_pred = gkmerr.predict_y_and_grad(phi_test, data_test.pos)
with torch.no_grad():
    print(mean_squared_error(y_test, y_pred))

mean_squared_error(grad_pred.reshape(-1, 3), grad_test.reshape(-1, 3))
