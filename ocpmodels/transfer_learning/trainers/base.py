import copy
import datetime
import logging
import random
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pprint

import ase.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ocpmodels.common.utils import save_checkpoint
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scheduler import LRScheduler
from ocpmodels.transfer_learning.common.logger import WandBLogger
from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    aggregate_metric,
    load_xyz_to_pyg_batch,
    load_xyz_to_pyg_data,
    torch_tensor_to_npy,
)
from ocpmodels.transfer_learning.models.distribution_regression import (
    GaussianKernel,
    KernelMeanEmbeddingRidgeRegression,
    LinearMeanEmbeddingKernel,
    median_heuristic,
)

from ..loaders import BaseLoader


# TODO: Move common methods here especially metric calculations
# Should decouple load_losses from metrics
class BaseTrainer:
    def __init__(self):
        pass

    def load_seed_from_config(self):
        # https://pytorch.org/docs/stable/notes/randomness.html
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_logger(self):
        self.logger = None
        if not self.is_debug:
            self.logger = WandBLogger(self.config)

    def _compute_metrics(self):
        # TODO: Move this to a common method
        pass

    def load_dataset(self):
        # TODO: Handle datasets commonly and then do internal conversions after
        pass

    def load_normalizers(self):
        self.normalizer = self.config["dataset"]["train"]
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", True):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                y = self.datasets["train"].y.clone().detach()
                self.normalizers["target"] = Normalizer(
                    mean=y.mean().float(),
                    std=y.std().float(),
                    device=self.device,
                )
            if "grad_target_mean" in self.normalizer:
                self.normalizers["grad_target"] = Normalizer(
                    mean=self.normalizer["grad_target_mean"],
                    std=self.normalizer["grad_target_std"],
                    device=self.device,
                )
            else:
                forces = self.datasets["train"].force.clone().detach()
                self.normalizers["grad_target"] = Normalizer(
                    mean=0.0,
                    std=forces.std().float(),
                    device=self.device,
                )
                self.normalizers["grad_target"].mean.fill_(0)
        else:
            self.normalizers["target"] = Normalizer(
                mean=0.0,
                std=1.0,
                device=self.device,
            )
            self.normalizers["grad_target"] = Normalizer(
                mean=0.0,
                std=1.0,
                device=self.device,
            )

    def _load_dataset_internal(self):
        pass
        # Here we will make it into the internal version
