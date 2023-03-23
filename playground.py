"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""

from pathlib import Path
import yaml
from pprint import pprint

from torch_geometric.data import Batch
from torch_geometric.nn import SumAggregation
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.extra import BaseLoader
import ase.io


### Load checkpoint
CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

### Load data
DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
raw_data = ase.io.read(DATA_PATH, index=":")
a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,
    r_forces=True,
    r_distances=False,
    r_edges=True,
    r_fixed=True,
)
data_object = a2g.convert_all(raw_data, disable_tqdm=True)
data_batch = Batch.from_data_list(data_object)
num_atoms = data_batch[0].num_nodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

am = data_batch.atomic_numbers
am = am.reshape(-1, 100, num_atoms)

# Instantiate model and load checkpoint
# the model can then be used like a standard torch model (although
# the data need to be batched according to torch_geometric, see Batch.from_data_list)

# representation_layer = 1
# base_loader = BaseLoader(checkpoint["config"],
#                          representation=True,
#                          representation_kwargs={
#                              "representation_layer": representation_layer,
#                          })
# base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)

# model = base_loader.model
# model.to(device)
# model.eval()


# Create figure of distance and kernel matrix of different layers over time
fig, ax = plt.subplots(5, 2, figsize=(2 * 3, 5 * 3))

for i, representation_layer in enumerate(range(1, 6)):
    base_loader = BaseLoader(checkpoint["config"],
                             representation=True,
                             representation_kwargs={
                                 "representation_layer": representation_layer,
                             })
    base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)

    model = base_loader.model
    model.to(device)
    model.eval()

    h = model(data_batch)
    phi = h.reshape(-1, num_atoms, h.shape[-1]).mean(dim=1)
    K = phi @ phi.t()
    pos = ax[i, 0].imshow(K.detach().cpu().numpy(), vmin=K.min(), vmax=K.max())
    ax[i, 0].set_title(f"K_{representation_layer}")
    fig.colorbar(pos, ax=ax[i, 0])

    D = torch.cdist(phi, phi, p=2)
    pos = ax[i, 1].imshow(D.detach().cpu().numpy(), vmin=D.min(), vmax=D.max())
    ax[i, 1].set_title(f"D_{representation_layer}")
    fig.colorbar(pos, ax=ax[i, 1])

fig.tight_layout()
fig.savefig("intermediate_representation.png")

Ds = {}
# Create figure of distance and kernel matrix gif
for representation_layer in range(1, 6):
    base_loader = BaseLoader(checkpoint["config"],
                             representation=True,
                             representation_kwargs={
                                 "representation_layer": representation_layer
                             })
    base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
    model = base_loader.model
    model.to(device)
    model.eval()

    h = model(data_batch)
    h = h.reshape(-1, num_atoms, h.shape[-1])

    D = torch.cdist(h, h, p=2)
    D_min = D.min()
    D_max = D.max()

    Ds[representation_layer] = {"D": D, "D_min": D_min, "D_max": D_max}

fig, ax = plt.subplots(1, 5, figsize=(5 * 3, 3), sharey=True)
imshows = [ax[i-1].imshow(Ds[i]["D"][0].detach().numpy(), vmin=Ds[i]["D_min"], vmax=Ds[i]["D_max"]) for i in range(1, 6)]
Ds[1]["D"].shape

def animate(t):
    for representation_layer in range(1, 6):
        imshows[representation_layer-1].set_data(Ds[representation_layer]["D"][t].detach().numpy())
        ax[representation_layer-1].set_title(f"D_{representation_layer}, t = {t:>2}")
    return imshows

ani = animation.FuncAnimation(fig, animate, frames=range(Ds[1]["D"].shape[0]), repeat=True)

writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)

ani.save(f"Ds_in_time.gif", writer=writer)

# Perform ridge regression on the mean representation as a feature vector describing the system.
# This is very crude and only serves as a sanity check that the representation is actually doing
# something useful at all.
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import  mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

ridge_scores = []
dummy_scores = []

for representation_layer in range(1, 6):
    base_loader = BaseLoader(checkpoint["config"],
                             representation=True,
                             representation_kwargs={
                                 "representation_layer": representation_layer
                             })
    base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
    model = base_loader.model
    model.to(device)
    model.eval()

    h = model(data_batch).mean(1).detach().cpu().numpy()
    h = h.reshape(-1, num_atoms)
    y = data_batch.y.detach().cpu().numpy()

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=3)
    ridge_scores.append(cross_val_score(ridge, h, y, cv=20, scoring="neg_mean_squared_error"))
    dummy = DummyRegressor()
    dummy_scores.append(cross_val_score(dummy, h, y, cv=20, scoring="neg_mean_squared_error"))

ridge_scores = -np.array(ridge_scores)
r_mean = ridge_scores.mean(1)
r_std = ridge_scores.std(1)

dummy_scores = -np.array(dummy_scores)
d_mean = dummy_scores.mean(1)
d_std = dummy_scores.std(1)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))

x = np.arange(1, 6)
ax.plot(x, r_mean, color="red", label='Ridge Regression MSE')
ax.fill_between(x, r_mean - r_std, r_mean + r_std, color='red', alpha=0.2)

ax.plot(x, d_mean, color="blue", label='Mean Regression MSE')
ax.fill_between(x, d_mean - d_std, d_mean + d_std, color='blue', alpha=0.2)

ax.set_xlabel("Representation Layer")
ax.set_ylabel("MSE")
ax.legend()
plt.tight_layout()
fig.savefig("regression.png")
