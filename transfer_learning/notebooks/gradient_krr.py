"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
from pathlib import Path

import numpy as np
import torch
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
    representation=True,
    representation_kwargs={
        "representation_layer": representation_layer,
    },
)
base_loader.load_checkpoint(CHECKPOINT_PATH, strict_load=False)
model = base_loader.model
model.to(device)
model.mekrr_forces = True

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.compile import compiled_function, draw_graph

# from torch.autograd.functional import jacobian, hessian, vjp, jvp, hvp

# def f(x, data, kernel, model):
#     data.pos = x
#     h = model(data)
#     h_ = h.clone().detach()
#     k = kernel(h, h_)
#     return k


class LinearKernel:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return x @ y.T


lkernel = LinearKernel()

from torchviz import make_dot

frames = 2
dat = Batch.from_data_list(data_batch[:frames])
pos = dat["pos"]
pos.requires_grad = True


def f(pos, data, model, kernel):
    h = model(data)[0]
    h_ = h.clone().detach()
    return kernel(h, h_)


y = f(
    pos,
    dat,
    model,
    lkernel,
)
m = y.shape[0]
gr = torch.autograd.grad(
    outputs=y,
    inputs=pos,
    grad_outputs=torch.ones_like(y),
    retain_graph=False,
    create_graph=False,
    allow_unused=False,
    is_grads_batched=False,
)[0]
pos.shape
gr.shape

output, vjp_fn = torch.func.vjp(lambda x: f(x, dat, model, lkernel), pos)

h = model(dat)[0]
h = h.reshape(-1, num_atoms, h.shape[-1])
gk = GaussianKernel()
gk.sigma = 1.0
gklme = LinearMeanEmbeddingKernel(gk)
k = gklme(h, h.clone().detach())

dot = make_dot(k, params={k: v for k, v in dat.to_dict().items() if v.requires_grad})
dot.render("graph", format="png")


# TODO
# 1. Explicitly create the matrix
# a. Matrix will have A_{00}, A_{01}, A_{11} and A_{10} = A_{01}^T
# A_{00} = G(x, x)
def A_00(kernel, data):
    h = model(data)
    h = h.reshape(-1, num_atoms, h.shape[-1])
    return kernel(h, h)


def A_01(kernel, data):
    data.pos.requires_grad = True
    h = model(data)
    h = h.reshape(-1, num_atoms, h.shape[-1])
    h_ = h.clone().detach()
    k = kernel(h, h_)
    gk = torch.autograd.grad(
        k,
        data.pos,
        grad_outputs=torch.ones_like(k),
        create_graph=False,
    )[0]
    return gk


class LinearKernel:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return x @ y.T


lkernel = LinearKernel()

from functorch.compile import compiled_function, draw_graph


def f(data):
    h = model(data)
    h = h.reshape(-1, num_atoms, h.shape[-1])
    return lkernel(h, h)


def fw(f, inps):
    draw_graph(f, "forward.svg")
    return f


def bw(f, inps):
    draw_graph(f, "backward.svg")
    return f


gkernel = GaussianKernel()
gkernel.sigma = 1.0
gklme = LinearMeanEmbeddingKernel(gkernel)

frames = 5
dat = Batch.from_data_list(data_batch[:frames])

k = A_00(gklme, dat)
gk = A_01(gklme, dat)
gk = gk.reshape(num_frames, num_atoms, 3)
gk

k.shape
gk.shape
5 * num_atoms


class LinearKernel:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return x @ y.T


lkernel = LinearKernel()
dat.pos.requires_grad = True
h = model(dat)
h_ = h.clone().detach()
k = lkernel(h, h_)
gk = torch.autograd.grad(
    k,
    dat.pos,
    grad_outputs=torch.ones_like(k),
    create_graph=False,
)[0]
# k = (k(x_i, x_j))_{i, j}^n

gk = gk.reshape(frames, num_atoms, 3)
gk[0]

##
frames = 2
dat = Batch.from_data_list(data_batch[:frames])
dat.pos.requires_grad = True
dat_ = Batch.from_data_list(data_batch_[:frames])
dat_.pos.requires_grad = True
h = model(dat).reshape(frames, num_atoms, -1).mean(dim=1)
h_ = model(dat_).reshape(frames, num_atoms, -1).mean(dim=1)
k = h @ h_.T


def f(dat1_pos, dat2_pos):
    h1 = model(dat).reshape(frames, num_atoms, -1).mean(dim=1)
    h2 = model(dat_).reshape(frames, num_atoms, -1).mean(dim=1)
    return h1 @ h2.T


# fpart = partial(f, dat1=dat, dat2=dat_)

grad(f)(dat.pos, dat_.pos)

gk = torch.autograd.grad(
    k,
    dat.pos,
    grad_outputs=torch.ones_like(k),
    create_graph=True,
)[0]

hess = torch.autograd.grad(
    gk[0],
    dat_.pos,
    grad_outputs=torch.ones(gk.shape),
    is_grads_batched=True,
)[0]


torch.ones((frames, *gk.shape)).shape


gk = GaussianKernel()
gk.sigma = 1.0
gklme = LinearMeanEmbeddingKernel(gk)

# # Estimator
gk = GaussianKernel()
gk.sigma = sigma
gklme = LinearMeanEmbeddingKernel(gk)
data_train.pos.requires_grad = True
g_ = gklme(phi_train.reshape(-1, num_atoms, d), phi_train.reshape(-1, num_atoms, d))

# 2. If we can do that, move to using the autograd thing + Pylinear ops (could possibly reuse the Chimela)
