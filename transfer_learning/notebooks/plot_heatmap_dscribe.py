#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
from dscribe.descriptors import SOAP
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


def main():
    parser = argparse.ArgumentParser(description="A simple argparse example.")

    parser.add_argument("--representation_layer", "-rl", type=int, help="Representation layer", default=2)
    parser.add_argument("--alpha", "-a", type=float, help="Alpha value", default=0.0)
    parser.add_argument("--aggregation", "-ag", type=str, help="Aggregation method", default="mean")
    parser.add_argument("--kernel", "-k", type=str, help="Kernel type", default="linear")

    args = parser.parse_args()

    DATASET_DIR = Path("data/private/2_abinitio-metad_Fe45N2.extxyz")
    alpha = args.alpha
    aggregation = args.aggregation
    start_idx = 1600
    end_idx = -10
    kernel = args.kernel
    plot_dir = Path(
        f"transfer_learning/notebooks/figures/{DATASET_DIR.stem}/alpha{alpha}_agg{aggregation}_soap_{kernel}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_PATH = Path("checkpoints/s2ef_efwt/all/schnet/schnet_all_large.pt")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    device = "cpu"

    ### Load data
    raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATASET_DIR, ATOMS_TO_GRAPH_KWARGS["schnet"])
    data_batch = Batch.from_data_list(data_batch[start_idx:end_idx])
    raw_data = raw_data[start_idx:end_idx]
    soap = SOAP(species=["Fe", "N"], r_cut=6, n_max=12, l_max=6, periodic=True, sparse=False)

    def create_soap_features(systems, soap_object):
        features = []
        for atoms in systems:
            features.append(soap_object.create(atoms))
        return torch.tensor(features)

    soap_features = create_soap_features(raw_data, soap)
    X = soap_features.detach().cpu()

    # Calculate the median sigma
    with torch.no_grad():
        sigma = median_heuristic(X.reshape(-1, X.shape[-1]), X.reshape(-1, X.shape[-1]))

    # kernel matrix
    if kernel == "gaussian":
        k0 = GaussianKernel(sigma)
    elif kernel == "linear":
        k0 = LinearKernel()

    t, n, _ = X.shape
    l, m, _ = X.shape

    zx = data_batch.atomic_numbers.squeeze()
    zy = zx

    delta = (zx[:, None] == zy[None, :]).squeeze()
    delta.shape

    agg_c = 1.0
    if aggregation == "mean":
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
    elif aggregation == "sum":
        mask = 1.0
    else:
        raise ValueError(f"Unknown aggregation {aggregation}")
    group_c = mask * delta

    kernel0 = k0(X.reshape(t * n, -1), X.reshape(l * m, -1))

    K = (kernel0 * ((1 - alpha) * agg_c + alpha * group_c)).reshape(t, n, l, m).sum(axis=(1, 3))

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
    plt.rcParams.update({"text.usetex": False})  # Doesn't work on server
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
        t,
        0,
        1,
        where=np.roll(~eq1, -1) | np.roll(~eq1, 1),
        color=cm[1],
        alpha=0.3,
        transform=ax[0].get_xaxis_transform(),
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


if __name__ == "__main__":
    main()
