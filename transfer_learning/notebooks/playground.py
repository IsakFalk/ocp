"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dscribe.descriptors import ACSF
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from transfer_learning.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    load_xyz_to_pyg_batch,
)
from transfer_learning.transfer_learning.loaders import BaseLoader
from transfer_learning.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    KernelMeanEmbeddingRidgeRegression,
    LinearMeanEmbeddingKernel,
    median_heuristic,
)

### Load checkpoint
CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

### Load data
DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    base_loader = BaseLoader(
        checkpoint["config"], representation=True, representation_kwargs={"representation_layer": representation_layer}
    )
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
imshows = [
    ax[i - 1].imshow(Ds[i]["D"][0].detach().numpy(), vmin=Ds[i]["D_min"], vmax=Ds[i]["D_max"]) for i in range(1, 6)
]
Ds[1]["D"].shape


def animate(t):
    for representation_layer in range(1, 6):
        imshows[representation_layer - 1].set_data(Ds[representation_layer]["D"][t].detach().numpy())
        ax[representation_layer - 1].set_title(f"D_{representation_layer}, t = {t:>2}")
    return imshows


ani = animation.FuncAnimation(fig, animate, frames=range(Ds[1]["D"].shape[0]), repeat=True)

writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)

ani.save(f"Ds_in_time.gif", writer=writer)

# Perform ridge regression on the mean representation as a feature vector describing the system.
# This is very crude and only serves as a sanity check that the representation is actually doing
# something useful at all.

ridge_scores = []
dummy_scores = []

for representation_layer in range(1, 6):
    base_loader = BaseLoader(
        checkpoint["config"], representation=True, representation_kwargs={"representation_layer": representation_layer}
    )
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
ax.plot(x, r_mean, color="red", label="Ridge Regression MSE")
ax.fill_between(x, r_mean - r_std, r_mean + r_std, color="red", alpha=0.2)

ax.plot(x, d_mean, color="blue", label="Mean Regression MSE")
ax.fill_between(x, d_mean - d_std, d_mean + d_std, color="blue", alpha=0.2)

ax.set_xlabel("Representation Layer")
ax.set_ylabel("MSE")
ax.legend()
plt.tight_layout()
fig.savefig("regression.png")

# Benchmark the method against others
y = data_batch.y.detach().cpu()
y = y.reshape(-1, 1).float()
# soap = SOAP(species=["Fe", "N"], rcut=6, n_max=3, l_max=3, periodic=True, sparse=False)
# def create_soap_features(systems, soap_object):
#     features = []
#     for atoms in systems:
#         features.append(soap_object.create(atoms))
#     return torch.tensor(features)
# soap_features = create_soap_features(soap_data, soap)
# soap_features = soap_features.detach().cpu().float()
# scaler = StandardScaler()
# new_soap_features = scaler.fit_transform(soap_features.reshape(-1, soap_features.shape[-1]))
# new_soap_features = new_soap_features.reshape(soap_features.shape)
# _soap_features = soap_features
# soap_features = torch.tensor(new_soap_features).float()

rcut = 6
bins = np.linspace(0, rcut, 100)
sigma = rcut / len(bins)
g2_params = np.stack([np.ones_like(bins) * (1 / (2 * sigma**2)), bins]).T
acsf = ACSF(species=["Fe", "N"], rcut=rcut, g2_params=g2_params, periodic=True, sparse=False)


def create_acsf_features(systems, acsf_object):
    features = []
    for atoms in systems:
        features.append(acsf_object.create(atoms))
    return torch.tensor(features)


acsf_features = create_acsf_features(raw_data, acsf)
acsf_features = acsf_features.detach().cpu().float()
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.plot(bins, acsf_features.mean(0)[:45].mean(0)[102:], label="ACSF", linestyle="--", marker=None)
fig.savefig("acsf.png")
dscribe_features = acsf_features

gk = GaussianKernel()

mse_dist_krr = []
mse_dscribe_krr = []
mse_dummy = []
for representation_layer in range(1, 2):
    base_loader = BaseLoader(
        checkpoint["config"], representation=True, representation_kwargs={"representation_layer": representation_layer}
    )
    base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
    model = base_loader.model
    model.to(device)
    model.eval()

    with torch.no_grad():
        phi = model(data_batch)
    phi = phi.detach().cpu()
    phi = phi.reshape(num_frames, num_atoms, -1)
    mse_dist_krr_temp = []
    mse_dscribe_krr_temp = []
    for _ in range(25):
        train_idx, test_idx = train_test_split(np.arange(num_frames), test_size=0.2)
        dscribe_train, dscribe_test = dscribe_features[train_idx], dscribe_features[test_idx]
        phi_train, phi_test = phi[train_idx], phi[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # SchNet
        sigma = median_heuristic(phi_train.reshape(-1, phi_train.shape[-1]), phi_train.reshape(-1, phi_train.shape[-1]))
        gk.sigma = sigma
        gklme = LinearMeanEmbeddingKernel(gk)
        gkmerr = KernelMeanEmbeddingRidgeRegression(gklme, lmbda=1e-6)
        y_mean = y_train.mean()
        gkmerr.fit(phi_train, y_train - y_mean)
        y_pred = gkmerr.predict(phi_test) + y_mean
        mse_dist_krr_temp.append(mean_squared_error(y_test, y_pred))

        # DSCRIBE
        sigma = median_heuristic(
            dscribe_train.reshape(-1, dscribe_train.shape[-1]), dscribe_train.reshape(-1, dscribe_train.shape[-1])
        )
        y_mean = y_train.mean()
        gkmerr.fit(dscribe_train, y_train - y_mean)
        y_pred = gkmerr.predict(dscribe_test) + y_mean
        mse_dscribe_krr_temp.append(mean_squared_error(y_test, y_pred))

        # Dummy
        mse_dummy.append(mean_squared_error(y_test, torch.ones_like(y_test) * y_mean))
    mse_dist_krr.append(mse_dist_krr_temp)
    mse_dscribe_krr.append(mse_dscribe_krr_temp)

mse_dist_krr = torch.tensor(mse_dist_krr)
mse_dscribe_krr = torch.tensor(mse_dscribe_krr)

print(f"mean and std (DSCRIBE): {torch.mean(mse_dscribe_krr):.3f}, {torch.std(mse_dscribe_krr):.3f}")
print(f"mean and std (SchNet): {torch.mean(mse_dist_krr):.3f}, {torch.std(mse_dist_krr):.3f}")

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
mse_dummy = np.array(mse_dummy)
boxplot_array = np.vstack([mse_dist_krr, mse_dscribe_krr]).T
ax.boxplot(boxplot_array, labels=["SchNet (layer1)", "DSCRIBE"], showbox=True, showcaps=False)
ax.set_ylabel("MSE")
ax.set_title("GKMERR")
ax.tick_params(axis="x", rotation=-45)
ax.set_yscale("log")
plt.tight_layout()
fig.savefig("gkmerr_regression.png")

print(np.mean(mse_dummy), torch.mean(mse_dist_krr).item(), torch.mean(mse_dscribe_krr).item())
