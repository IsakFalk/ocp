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
from .base import BaseTrainer


class MEKRRTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_config,
        kernel_config,
        optimizer_config,
        logger_config,
        print_every=10,
        seed=None,
        cpu=False,
        name="MeanEmbeddingKRRtrainer",
        run_dir="checkpoints",
        is_debug=False,
    ):
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.kernel_config = kernel_config  # Config for kernel algorithm
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
        self.optimizer = copy.deepcopy(optimizer_config)  # Config for optimizer
        self.optimizer["energy_loss_coefficient"] = 1 - self.optimizer.get("force_loss_coefficient", 0.0)
        self.run_dir = run_dir
        self.path_run_dir = Path(self.run_dir)
        self.path_run_dir.mkdir(parents=True, exist_ok=True)
        self.cpu = cpu
        self.print_every = print_every  # Not used here
        self.seed = seed
        self.run_dir = run_dir
        self.timestamp_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Setup paths
        self.base_path = self.path_run_dir / self.timestamp_id
        self.checkpoint_dir = self.base_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.base_path / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        # Load model config directly from pretrained checkpoint
        # and massage into right form
        self.model_config = torch.load(self.kernel_config["model_checkpoint_path"], map_location="cpu")["config"]
        name = self.model_config["model"]
        self.model_config = self.model_config["model_attributes"]
        self.model_config["regress_forces"] = True

        self.config = {
            "model": name,
            "model_attributes": self.model_config,
            "kernel": self.kernel_config,
            "logger": self.logger_config,
            "optim": self.optimizer,
            "name": name,
            "timestamp_id": self.timestamp_id,
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_normalizers()
        self._load_data_internal()
        self.load_model()
        self.load_loss()

    def _load_data_internal(self):
        # Tranform the data into the a full batch
        for split in ["train", "val", "test"]:
            self.datasets[split] = Batch.from_data_list(self.datasets[split])

    def load_model(self):
        _config = copy.deepcopy(self.config)
        loader = BaseLoader(
            _config,
            representation=True,
            representation_kwargs=_config["kernel"]["representation_params"],
            regress_forces=_config["model_attributes"].pop("regress_forces"),
            seed=self.seed,
            cpu=True,  # Since we loaded the model onto the cpu to start
        )
        loader.load_checkpoint(self.kernel_config["model_checkpoint_path"], strict_load=False)
        self.model = loader.model.to(self.device)
        logging.info("Loaded model from checkpoint successfully!")
        if self.logger is not None:
            self.logger.watch(self.model)

    def load_loss(self):
        # TODO: Allow for different losses
        self.loss_fn = {
            "energy": nn.MSELoss(),
            "forces": nn.MSELoss(),
        }
        # for loss, loss_name in self.loss_fn.items():
        #     # NOTE: DPPLoss is for distributed training
        #     # but also does things like taking care of nans,
        #     # we generally won't use the para   llel stuff
        #     # and only the other QOL features
        #     self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])

    def train(self):
        # Set up data, taking care of normalization
        y = self.normalizers["target"].norm(self.datasets["train"].y.float().to(self.device)).reshape(-1, 1)
        # TODO: Implement gradient fitting
        # grad = self.normalizers["grad_target"].norm(self.datasets["train"].force.float().to(self.device))
        X, _ = self.model(self.datasets["train"].to(self.device))
        self.d = X.shape[-1]
        # TODO: View
        X = X.reshape(-1, self.config["dataset"]["train"]["num_atoms"], self.d)

        # Dispatch to kernel
        if self.config["kernel"].get("k0", "gaussian") == "gaussian":
            # Use median heuristic for Gaussian kernel.
            if (
                self.config["kernel"].get("k0", "gaussian") == "gaussian"
                and self.config["kernel"]["k0_params"].get("median_heuristic", True) is True
            ):
                with torch.no_grad():
                    self.k0_sigma = median_heuristic(X.reshape(-1, self.d), X.reshape(-1, self.d))
            else:
                self.k0_sigma = self.config["kernel"]["k0_params"].get("sigma", 1.0)
            self.k0 = GaussianKernel(self.k0_sigma)
        else:
            raise NotImplementedError

        if self.config["kernel"].get("k1", "linear") == "linear":
            self.k1 = LinearMeanEmbeddingKernel(self.k0)
        else:
            raise NotImplementedError

        self.regressor = KernelMeanEmbeddingRidgeRegression(self.k1, **self.kernel_config["regressor_params"])
        assert (
            X.shape[0] == self.dataset_config["train"]["num_frames"]
            and X.shape[1] == self.dataset_config["train"]["num_atoms"]
        ), "X should have shape (num_frames, num_atoms, d)"
        self.regressor.fit(X, y)

    def validate(self, split="val"):
        self.model.eval()
        dataset = self.datasets[split]

        metrics = {"energy_loss": [], "forces_loss": []}
        # Forward.
        out = self._forward(split)

        # Compute metrics.
        metrics = self._compute_metrics(out, dataset, metrics)

        log_dict = {k: aggregate_metric(metrics[k]) for k in metrics}
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        logging.info(", ".join(log_str))

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=0,
                split=split,
            )

        return metrics

    def _compute_metrics(self, out, batch_list_or_batch, metrics):
        # NOTE: Removed some additional things we probably want
        if not isinstance(batch_list_or_batch, list):
            batch_list = [batch_list_or_batch]
        else:
            batch_list = batch_list_or_batch

        losses = {}

        # Energy loss.
        # TODO: Remove unnecessary reshapes
        energy_target = torch.cat([batch.y.to(self.device) for batch in batch_list], dim=0).float()
        losses["energy"] = self.loss_fn["energy"](
            self.normalizers["target"].denorm(out["energy"]).reshape(-1), energy_target.reshape(-1)
        )
        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0).float()
            losses["forces"] = self.loss_fn["forces"](
                self.normalizers["grad_target"].denorm(out["forces"]).reshape(-1, 3), force_target.reshape(-1, 3)
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in losses.values():
            assert hasattr(lc, "grad_fn")

        # Note that loss is normalized
        metrics["energy_loss"].append(losses["energy"].item())
        if self.config["model_attributes"].get("regress_forces", True):
            metrics["forces_loss"].append(losses["forces"].item())
        metrics["loss"] = (
            self.config["optim"].get("energy_loss_coefficient") * losses["energy"]
            + self.config["optim"].get("force_loss_coefficient") * losses["forces"]
        )
        return metrics

    def _forward(self, split):
        dataset = self.datasets[split]
        num_atoms = self.dataset_config[split]["num_atoms"]
        # forward pass
        dataset = Batch.from_data_list(dataset[:]).to(self.device)
        dataset.pos.requires_grad_(True)
        X, _ = self.model(dataset)  # (num_frames, num_atoms, d)
        X = X.reshape(-1, num_atoms, self.d)
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_grad = self.regressor.predict_y_and_grad(X, dataset.pos)
            out_forces = -out_grad.reshape(-1, 3)
        else:
            out_energy = self.regressor.predict(X)

        out_energy = out_energy.view(-1)
        # if out_energy.shape[-1] == 1:
        #     out_energy = out_energy.view(-1)

        # TODO: Don't hardcode float
        out = {
            "energy": out_energy.float(),
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces.float()

        return out

    def _compute_loss(self, out, batch_list_or_batch):
        # NOTE: Removed some additional things we probably want
        if not isinstance(batch_list_or_batch, list):
            batch_list = [batch_list_or_batch]
        else:
            batch_list = batch_list_or_batch

        losses = {}

        # Energy loss.
        energy_target = torch.cat([batch.y.to(self.device) for batch in batch_list], dim=0).float()
        energy_target = self.normalizers["target"].norm(energy_target)
        losses["energy"] = self.loss_fn["energy"](out["energy"], energy_target)
        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0).float()
            force_target = self.normalizers["grad_target"].norm(force_target)
            losses["forces"] = self.loss_fn["forces"](out["forces"], force_target)

        # Sanity check to make sure the compute graph is correct.
        for lc in losses.values():
            assert hasattr(lc, "grad_fn")

        loss = (
            self.config["optim"].get("energy_loss_coefficient") * losses["energy"]
            + self.config["optim"].get("force_loss_coefficient") * losses["forces"]
        )
        return loss, losses

    # Takes in a new data source and generates predictions on it.
    def predict(self, split):
        logging.info(f"Predicting {split}.")
        self.model.eval()
        predictions = {"energy": [], "forces": []}
        out = self._forward(split)
        # denorm
        out["energy"] = self.normalizers["target"].denorm(out["energy"])
        out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        predictions["energy"] = out["energy"].detach()
        predictions["forces"] = out["forces"].detach().reshape(-1, self.dataset_config[split]["num_atoms"], 3)
        return predictions
