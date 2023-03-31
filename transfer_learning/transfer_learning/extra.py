"""Added by Isak Falk

This file contains extra classes and functions that are not part of the original OCP code.

At some point we will refactor this code to be more in line with the original
OCP code. For now, we just want to get it working.
"""
import datetime
import errno
import logging
import os
import random
import subprocess
import copy
from abc import ABC, abstractmethod
from collections import defaultdict

from pathlib import Path
import yaml
from pprint import pprint

from torch_geometric.data import Batch
from torch_geometric.nn import SumAggregation
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import ocpmodels
from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict, save_checkpoint
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import AtomwiseL2Loss, DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.scheduler import LRScheduler

class BaseLoader:
    """Base class for model loaders.

    Load a model from a config file and potentially load the checkpoint state_dict
    """

    # Unused for now
    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, (OCPDataParallel, DistributedDataParallel)):
            module = module.module
        return module

    def __init__(
        self,
        model,
        representation: bool =False,
        representation_kwargs={},
        seed=None,
        cpu=False,
        name="base_model_loader",
    ):
        self.name = name
        self.representation = representation
        self.representation_kwargs = representation_kwargs
        self.cpu = cpu # TODO: Have not been tested with cuda but should work
        self.num_targets = 1 # NOTE: This is due to OCP code and should be fixed to 1

        # Don't mutate the original model dict
        model = copy.deepcopy(model)

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:0")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        # Due to how the internals of OCP work, we need to separate the name of the model class
        # and the attributes of the model class. This is done by the "model" and "model_attributes".
        # If the config is loaded from a checkpoint file, this allows us to recreate the correct model
        # and load the state_dict automatically
        self.config = {
            "model": model.pop("model"),
            "model_attributes": model["model_attributes"],
            "seed": seed
        }

        # Print the current config to stdout
        print(yaml.dump(self.config, default_flow_style=False))
        self.load()

    def load(self):
        self.load_seed_from_config()
        self.load_model()

    def load_seed_from_config(self):
        """Set random seed from config file."""
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = self.config["seed"]
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_model(self):
        """Load the model from the config file."""

        # The OCP repo use a funny registry which allows them to load classes
        # using strings through a key-value store mapping strings to the correct
        # class object
        # This makes the registry available
        from ocpmodels.common.utils import setup_imports
        setup_imports()

        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")
            if self.representation:
                logging.info(f"Model used for representation")

        # TODO: Says it's deprecated in the OCP code but it's required for now
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        # Load the model class from the registry
        self.model = registry.get_model_class(self.config["model"])(
            None,
            bond_feat_dim,
            self.num_targets,
            self.representation,
            **self.representation_kwargs,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

    def load_checkpoint(self, checkpoint_path, strict_load=True):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        # NOTE: Their custom state_dict loader breaks for some unknown reason
        # related to the keys. This is due to the method ocpmodels.common.utils._report_incompat_keys
        # load_state_dict(self.model, new_dict, strict=strict_load)
        # Instead we mimic the checks they would do here taking care of errors
        incompat_keys = self.model.load_state_dict(new_dict, strict=strict_load)

        error_msgs = []
        if len(incompat_keys.unexpected_keys) > 0:
            error_msgs.insert(
                0,
                "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in incompat_keys.unexpected_keys)
                ),
            )
        if len(incompat_keys.missing_keys) > 0:
            error_msgs.insert(
                0,
                "Missing key(s) in state_dict: {}. ".format(
                    ", ".join('"{}"'.format(k) for k in incompat_keys.missing_keys)
                ),
            )

        if len(error_msgs) > 0:
            error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
                self.model.__class__.__name__, "\n\t".join(error_msgs)
            )
            if strict_load:
                raise RuntimeError(error_msg)
            else:
                logging.warning(error_msg)

### Estimators and distribution regression
def gaussian_mean_embedding_kernel(x, y, sigma=1.0):
    """
    Calculate the Gaussian mean embedding kernel between two sets of points.

    Parameters:
        x (torch.Tensor): A 3-dimensional tensor representing a set of t sets of n d-dimensional points.
        y (torch.Tensor): A 3-dimensional tensor representing a set of l sets of m d-dimensional points.
        sigma (float): The standard deviation of the Gaussian kernel used when calculating the kernel.
    Returns:
        torch.Tensor: A 2-dimensional tensor representing the kernel matrix of size t x l.
    """
    t, n, d = x.shape
    l, m, d = y.shape
    x = x.reshape(t * n, d)
    y = y.reshape(l * m, d)
    Dsq = torch.cdist(x, y, p=2)**2
    K = torch.exp(-Dsq / (2 * sigma**2))
    K = K.reshape(t, n, l, m).sum(axis=(1, 3)) / (n * m)
    return K

def median_heuristic(x, y):
    return torch.median(torch.cdist(x, y, p=2))

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import torch.linalg as LA

### Create distribution regression
### Output is assumed to be in \R and
### we have T snapshots, indexed by t, and the system is
### described by N atoms, indexed by i.
### This means that the kernel (when using some pointwise kernel K)
### is of size T x T
### Call this gram matrix K, then
### K_{t, l} = torch.sum(G^{t, l}) / N**2 where
### G^{t, l}_{i, j} = K(x^{t}_{i}, x^{l}_{j})
### Thus, the only thing we need to do is to build a kernel tensor
### of size T x T x N x N and then sum over the last two dimensions
### (or equivalently)

class GaussianKernelMeanEmbeddingRidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lmbda=1.0, sigma=1.0, fit_sigma_using_median_heuristic=False):
        self.lmbda = lmbda
        self.sigma = sigma
        self.fit_sigma_using_median_heuristic = fit_sigma_using_median_heuristic

    def fit(self, X, y):
        assert len(X.shape) == 3
        if self.fit_sigma_using_median_heuristic:
            self.sigma = median_heuristic(X, X)
        self._X = X
        self._y = y

        K = gaussian_mean_embedding_kernel(X, X, sigma=self.sigma)
        Kl = K + torch.eye(K.shape[0]) * self.lmbda
        self._K = K
        self._alpha = LA.solve(Kl, y)
        return self

    def predict(self, X):
        assert len(X.shape) == 3
        K = gaussian_mean_embedding_kernel(X, self.X, sigma=self.sigma)
        return K @ self._alpha

    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)


### Utility functions
from ocpmodels.preprocessing import AtomsToGraphs
from torch_geometric.data import Batch
import ase.io
# SchNet
# atoms_to_graph_kwargs=dict(max_neigh=50,
#                            radius=6,
#                            r_energy=True,
#                            r_forces=True,
#                            r_distances=False,
#                            r_edges=True,
#                            r_fixed=True)
def load_xyz_to_pyg_batch(path, atoms_to_graph_kwargs):
    """
    Load XYZ data from a given path using ASE and convert it into a PyTorch Geometric Batch object.

    Args:
        path (Path): Path to the XYZ data file.
        **atoms_to_graph_kwargs: Optional keyword arguments for AtomsToGraphs class.

    Returns:
        Tuple consisting of raw_data, data_batch, num_frames, and num_atoms.
        raw_data (Atoms): Raw data loaded from the file using ASE.
        data_batch (Batch): Batch object containing converted data for all frames.
        num_frames (int): Number of frames in the loaded XYZ data file.
        num_atoms (int): Number of atoms in each frame.
    """
    raw_data = ase.io.read(path, index=":")
    num_frames = len(raw_data)
    a2g = AtomsToGraphs(
        **atoms_to_graph_kwargs,
    )
    data_object = a2g.convert_all(raw_data, disable_tqdm=True)
    data_batch = Batch.from_data_list(data_object)
    num_atoms = data_batch[0].num_nodes
    return raw_data, data_batch, num_frames, num_atoms
