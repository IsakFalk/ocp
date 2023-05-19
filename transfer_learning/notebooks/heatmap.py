#!/usr/bin/env python3

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
    EmbeddingKernel,
    GaussianKernel,
    KernelGroupEmbeddingRidgeRegression,
    KernelMeanEmbeddingRidgeRegression,
    LinearGroupEmbeddingKernel,
    LinearMeanEmbeddingKernel,
    StandardizedOutputRegression,
    median_heuristic,
)

# from torch_geometric.data.batch import DataBatch


def plot_everything(dataset_path, representation_layer, subsample_k, alpha, aggregation, lmbda, plot_dir):
    ### Load checkpoint
    CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    ### Load data
    raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(dataset_path, ATOMS_TO_GRAPH_KWARGS["schnet"])

    ### Get representations
    # Get every k'th frame from data_batch

    data_batch = Batch.from_data_list([data_batch[i] for i in range(0, len(data_batch), subsample_k)])
    data_batch.batch *= subsample_k

    # Calculate the median sigma
    with torch.no_grad():
        h = model(data_batch)
        d = h.shape[-1]
        sigma = median_heuristic(h.reshape(-1, d), h.reshape(-1, d))
        del h

    # kernel matrix
    k0 = GaussianKernel(sigma=sigma)
    ek = EmbeddingKernel(k0, alpha, aggregation)

    X = model(data_batch)
    X = X.reshape(-1, num_atoms, X.shape[-1])
    ZX = data_batch.atomic_numbers.reshape(X.shape[0], num_atoms, 1)

    K = ek(X, ZX, X, ZX).detach().cpu()

    from sklearn.preprocessing import KernelCenterer

    K_c = torch.tensor(KernelCenterer().fit_transform(K))

    ### Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(np.arange(len(data_batch.y)) * subsample_k, data_batch.y)
    fig.savefig(plot_dir / "y.png")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # Plot a heatmap of K and K_c
    fig, ax = plt.subplots()
    im = ax.imshow(K, interpolation="none")
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.savefig(plot_dir / "k.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots()
    im = ax.imshow(K_c, interpolation="none")
    cbar = ax.figure.colorbar(im, ax=ax)
    fig.savefig(plot_dir / "k_c.png", dpi=300, bbox_inches="tight")

    # lmbda = 1e-6
    # torch.diag(K @ (K + lmbda * torch.eye(K.shape[0])).pinv())

    lmbda = 1e-9
    t = K.shape[0]
    leverage_scores = torch.linalg.lstsq(K + lmbda * torch.eye(t), K).solution.diag()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(np.arange(len(leverage_scores)) * subsample_k, leverage_scores)
    fig.savefig(plot_dir / "leverage_scores.png")

    t = K_c.shape[0]
    leverage_scores = torch.linalg.lstsq(K_c + lmbda * torch.eye(t), K_c).solution.diag()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(np.arange(len(leverage_scores)) * subsample_k, leverage_scores)
    fig.savefig(plot_dir / "leverage_scores_c.png")


datasets = [
    # Path("data/private/1_abinitio-md_Fe45N2.extxyz"),
    Path("data/private/2_abinitio-metad_Fe45N2.extxyz"),
    # Path("data/formate/fcu-nonreactive_4036-4507.xyz"),
    # Path("data/formate/fcu-reactive_6328-end.xyz")
]

alphas = [0.0, 1.0]
for dataset in datasets:
    print("Dataset:", dataset.stem)
    for alpha in alphas:
        print("Alpha:", alpha)
        for aggregation in ["mean"]:
            print("Aggregation:", aggregation)
            plot_dir = Path(f"transfer_learning/notebooks/figures/{dataset.stem}/alpha{alpha}_agg{aggregation}")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_everything(
                dataset,
                representation_layer=2,
                subsample_k=4,
                alpha=alpha,
                aggregation=aggregation,
                lmbda=1e-9,
                plot_dir=plot_dir,
            )

import matplotlib.pyplot as plt

plt.close()

### Create plot
dataset_dir = Path("data/private/2_abinitio-metad_Fe45N2.extxyz")
subsample_k = 2
representation_layer = 2
alpha = 1.0
aggregation = "mean"
lmbda = 1e-9
plot_dir = Path(f"transfer_learning/notebooks/figures/{dataset_dir.stem}/alpha{alpha}_agg{aggregation}")

CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

### Load data
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(dataset_dir, ATOMS_TO_GRAPH_KWARGS["schnet"])

### Get representations
# Get every k'th frame from data_batch

data_batch = Batch.from_data_list(data_batch[1600:])

# Calculate the median sigma
with torch.no_grad():
    h = model(data_batch)
    d = h.shape[-1]
    sigma = median_heuristic(h.reshape(-1, d), h.reshape(-1, d))
    del h

# kernel matrix
k0 = GaussianKernel(sigma=sigma)
ek = EmbeddingKernel(k0, alpha, aggregation)

X = model(data_batch)
X = X.reshape(-1, num_atoms, X.shape[-1])
ZX = data_batch.atomic_numbers.reshape(X.shape[0], num_atoms, 1)

K = ek(X, ZX, X, ZX).detach().cpu().numpy()

from sklearn.cluster import SpectralClustering

# set number of clusters
n_clusters = 2

# perform spectral clustering
spec_cluster = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
labels = spec_cluster.fit_predict(K)

# plot clustering
t = np.arange(ts.shape[0] + 1)
ts = np.loadtxt("./transfer_learning/notebooks/2_abinitio-metad_Fe45N2_N2distance.txt")[1600:]
# K = np.random.rand(100, 100)
# labels = np.random.randint(0, 2, size=100)

from matplotlib import pyplot as plt

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 4]})
# fig.tight_layout(pad=3.0)

# Plot timeseries with shading indicating labels
axs[0].plot(t, color="k")
cmap = plt.get_cmap("gray")
cmap.set_bad(color="gray")  # Set color for invalid label values
im = axs[0].imshow(labels.reshape(1, -1), cmap=cmap, aspect="auto", extent=[t[0], t[-1], -0.5, 0.5])

# Set ticks and labels for x-axis
axs[0].set_xticks([t[0], t[-1]])
axs[0].set_xticklabels([t[0], t[-1]], fontsize=10)

# Add colorbar for labels
cbax = fig.add_axes([0.92, 0.7, 0.02, 0.2])
cb = plt.colorbar(im, cax=cbax)
cb.ax.tick_params(labelsize=10)

# Plot heatmap of K
im = axs[1].imshow(K, cmap="coolwarm", aspect="auto")
axs[1].set_xlabel("Time", fontsize=12)
axs[1].set_ylabel("Time", fontsize=12)
cbax = fig.add_axes([0.92, 0.1, 0.02, 0.5])
cb = plt.colorbar(im, cax=cbax)
cb.ax.tick_params(labelsize=10)

# Share x-axis between plots
plt.subplots_adjust(hspace=0.05)
plt.setp(axs[0].xaxis.get_majorticklabels(), visible=False)

fig.savefig(plot_dir / "spectral_clustering.png")

plt.show()
