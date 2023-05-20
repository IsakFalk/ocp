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
    LinearKernel,
    LinearMeanEmbeddingKernel,
    StandardizedOutputRegression,
    median_heuristic,
)

DATASET_DIR = Path("data/private/2_abinitio-metad_Fe45N2.extxyz")
representation_layer = 2
alpha = 1.0
aggregation = "mean"
# start_idx = 1600
# end_idx = -10
start_idx = 2000
end_idx = -10
kernel = "gaussian"
plot_dir = Path(f"transfer_learning/notebooks/figures/{DATASET_DIR.stem}/alpha{alpha}_agg{aggregation}_{kernel}")
plot_dir.mkdir(parents=True, exist_ok=True)

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
raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATASET_DIR, ATOMS_TO_GRAPH_KWARGS["schnet"])

### Get representations
data_batch = Batch.from_data_list(data_batch[start_idx:end_idx])

# Calculate the median sigma
with torch.no_grad():
    h = model(data_batch)
    d = h.shape[-1]
    sigma = median_heuristic(h.reshape(-1, d), h.reshape(-1, d))
    del h

# kernel matrix
if kernel == "gaussian":
    k0 = GaussianKernel(sigma)
elif kernel == "linear":
    k0 = LinearKernel()
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
ts = np.loadtxt("./transfer_learning/notebooks/2_abinitio-metad_Fe45N2_N2distance.txt")[start_idx:end_idx]
t = np.arange(ts.shape[0])
import seaborn as sns
from matplotlib import pyplot as plt

plt.style.use("seaborn-v0_8-paper")

# Styling
import matplotlib as mpl

mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.linestyle"] = "-"

font = {"family": "Times New Roman", "weight": "normal", "size": 12}
mpl.rc("mathtext", **{"default": "regular"})
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 14
mpl.rc("font", **font)
plt.rcParams.update({"text.usetex": True})
# End

### Plotting
heatmap_height = 1
ls_ratio = 0.4
fig, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]}, sharex=True)

ax[0].plot(ts)
cm = sns.color_palette("Set2")
eq1 = np.array(labels == 1)
ax[0].fill_between(t, 0, 1, where=eq1, color=cm[0], alpha=0.3, transform=ax[0].get_xaxis_transform())
ax[0].fill_between(
    t, 0, 1, where=np.roll(~eq1, -1) | np.roll(~eq1, 1), color=cm[1], alpha=0.3, transform=ax[0].get_xaxis_transform()
)
ax[0].set_ylabel("$d(\mathrm{N},\mathrm{N})$ [\AA]")
ax[0].yaxis.set_ticks([0, 4.0])
ax[0].set_box_aspect(1 / 3)

im = ax[1].imshow(K)
ax[1].set_aspect("equal")
ax[1].xaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].yaxis.set_major_locator(plt.MaxNLocator(4))
ax[1].set_xlabel("t")
ax[1].set_ylabel("t")
ax[1].set_box_aspect(1)

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = inset_axes(ax[1], width="5%", height="100%", loc="right", borderpad=-3)
cbar = fig.colorbar(im, cax=axins, orientation="vertical")
cbar.set_label("Similarity")
fig.savefig(plot_dir / "spectral_clustering", dpi=150, bbox_inches="tight")
