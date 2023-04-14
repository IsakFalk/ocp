import copy
import logging
import random
from pathlib import Path
from pprint import pprint

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

from .loaders import BaseLoader


# TODO: Move common methods here especially metric calculations
# Should decouple load_losses from metrics
class BaseTRainer:
    def __init__(self):
        pass


class MEKRRTrainer:
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
        # Create run dir if it doesn't exist

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
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

        # Setup paths
        self.checkpoint_dir = self.path_run_dir / "checkpoints" / self.logger.timestamp_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.path_run_dir / "predictions" / self.logger.timestamp_id
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

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
                y = torch.tensor([data.y.clone().detach() for data in self.train_dataset])
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
                forces = torch.cat([data.force.clone().detach() for data in self.train_dataset], dim=0)
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

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_model()
        self.load_loss()

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

    def load_datasets(self):
        _, self.train_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["train"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["train"]["num_frames"] = num_frames
        self.config["dataset"]["train"]["num_atoms"] = num_atoms
        _, self.val_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["val"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["val"]["num_frames"] = num_frames
        self.config["dataset"]["val"]["num_atoms"] = num_atoms
        _, self.test_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["test"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["test"]["num_frames"] = num_frames
        self.config["dataset"]["test"]["num_atoms"] = num_atoms

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
        y = self.normalizers["target"].norm(self.train_dataset.y.float().to(self.device)).reshape(-1, 1)
        # TODO: Implement gradient fitting
        # grad = self.normalizers["grad_target"].norm(self.train_dataset.force.float().to(self.device))
        X, _ = self.model(self.train_dataset.to(self.device))
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
        dataset = self.val_dataset if split == "val" else self.test_dataset

        metrics = {"energy_loss": [], "forces_loss": []}
        # Forward.
        out = self._forward(dataset, self.dataset_config[split]["num_atoms"])

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

    def _forward(self, dataset, num_atoms):
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
    def predict(
        self,
        dataset,
        num_atoms,
    ):
        logging.info("Predicting.")
        # self.model.eval()
        predictions = {"energy": [], "forces": []}
        out = self._forward(dataset, num_atoms)
        # denorm
        out["energy"] = self.normalizers["target"].denorm(out["energy"])
        out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        predictions["energy"].extend(out["energy"].detach())
        predictions["forces"].extend(out["forces"].detach())

        predictions["energy"] = torch.stack(predictions["energy"])
        predictions["forces"] = torch.stack(predictions["forces"])

        return predictions


class GAPTrainer:
    def __init__(
        self,
        dataset_config,
        model_config,
        logger_config,
        print_every=10,
        seed=None,
        cpu=False,
        name="MeanEmbeddingKRRtrainer",
        run_dir="checkpoints",
        is_debug=False,
    ):
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
        self.run_dir = run_dir
        self.path_run_dir = Path(self.run_dir)
        self.path_run_dir.mkdir(parents=True, exist_ok=True)
        self.cpu = cpu
        self.print_every = print_every  # Not used here
        self.seed = seed
        self.run_dir = run_dir
        # Create run dir if it doesn't exist

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
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

        # Setup paths
        self.checkpoint_dir = self.path_run_dir / "checkpoints" / self.logger.timestamp_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.path_run_dir / "predictions" / self.logger.timestamp_id
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

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
                y = torch.tensor([data.y.clone().detach() for data in self.train_dataset])
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
                forces = torch.cat([data.force.clone().detach() for data in self.train_dataset], dim=0)
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

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_model()
        self.load_loss()

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

    def load_datasets(self):
        _, self.train_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["train"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["train"]["num_frames"] = num_frames
        self.config["dataset"]["train"]["num_atoms"] = num_atoms
        _, self.val_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["val"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["val"]["num_frames"] = num_frames
        self.config["dataset"]["val"]["num_atoms"] = num_atoms
        _, self.test_dataset, num_frames, num_atoms = load_xyz_to_pyg_batch(
            self.dataset_config["test"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.config["dataset"]["test"]["num_frames"] = num_frames
        self.config["dataset"]["test"]["num_atoms"] = num_atoms

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
        y = self.normalizers["target"].norm(self.train_dataset.y.float().to(self.device)).reshape(-1, 1)
        # TODO: Implement gradient fitting
        # grad = self.normalizers["grad_target"].norm(self.train_dataset.force.float().to(self.device))
        X, _ = self.model(self.train_dataset.to(self.device))
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
        dataset = self.val_dataset if split == "val" else self.test_dataset

        metrics = {"energy_loss": [], "forces_loss": []}
        # Forward.
        out = self._forward(dataset, self.dataset_config[split]["num_atoms"])

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

    def _forward(self, dataset, num_atoms):
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
    def predict(
        self,
        dataset,
        num_atoms,
    ):
        logging.info("Predicting.")
        # self.model.eval()
        predictions = {"energy": [], "forces": []}
        out = self._forward(dataset, num_atoms)
        # denorm
        out["energy"] = self.normalizers["target"].denorm(out["energy"])
        out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        predictions["energy"].extend(out["energy"].detach())
        predictions["forces"].extend(out["forces"].detach())

        predictions["energy"] = torch.stack(predictions["energy"])
        predictions["forces"] = torch.stack(predictions["forces"])

        return predictions


class GNNTrainer:
    def __init__(
        self,
        task_config,
        model_config,
        dataset_config,
        optimizer_config,
        logger_config,
        print_every=30,
        seed=None,
        cpu=False,
        name="trainer",
        run_dir="checkpoints",
        is_debug=False,
    ):
        self.task_config = copy.deepcopy(task_config)
        self.model_config = copy.deepcopy(model_config)  # Config for model
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.optimizer = copy.deepcopy(optimizer_config)  # Config for optimizer
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
        self.optimizer["energy_loss_coefficient"] = 1 - self.optimizer.get("force_loss_coefficient", 0.0)
        self.model_config["regress_forces"] = True
        self.cpu = cpu
        self.print_every = print_every
        self.seed = seed
        self.run_dir = run_dir
        # Create run dir if it doesn't exist
        self.path_run_dir = Path(self.run_dir)
        self.path_run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.path_run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.path_run_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug
        self.epoch = 0
        self.step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        self.config = {
            "model": self.model_config.pop("name"),
            "model_attributes": self.model_config,
            "optim": self.optimizer,
            "logger": self.logger_config,
            "name": name,
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

        # Setup paths
        self.checkpoint_dir = self.path_run_dir / "checkpoints" / self.logger.timestamp_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.path_run_dir / "predictions" / self.logger.timestamp_id
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        self.normalizer = self.config["dataset"]["train"]
        self.normalizers = dict()
        if self.normalizer.get("normalize_labels", True):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                y = torch.tensor([data.y.clone().detach() for data in self.train_dataset])
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
                forces = torch.cat([data.force.clone().detach() for data in self.train_dataset], dim=0)
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

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_model()
        self.load_loss()
        self.load_optimizer()
        self.load_extras()

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

    def load_datasets(self):
        _, self.train_dataset, _, _ = load_xyz_to_pyg_data(
            self.dataset_config["train"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.train_loader = self.get_dataloader(self.train_dataset)
        _, self.val_dataset, _, _ = load_xyz_to_pyg_data(
            self.dataset_config["val"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.val_loader = self.get_dataloader(self.val_dataset)
        _, self.test_dataset, _, _ = load_xyz_to_pyg_data(
            self.dataset_config["test"]["src"], ATOMS_TO_GRAPH_KWARGS[self.config["model"]]
        )
        self.test_loader = self.get_dataloader(self.test_dataset)

    def get_dataloader(self, list_of_data):
        loader = DataLoader(
            list_of_data,
            batch_size=self.config["optim"]["batch_size"],
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
        )
        return loader

    def load_model(self):
        _config = copy.deepcopy(self.config)
        self.model = BaseLoader(
            _config, regress_forces=_config["model_attributes"].pop("regress_forces"), seed=self.seed, cpu=self.cpu
        ).model.to(self.device)
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

    def load_optimizer(self):
        optimizer = self.config["optim"].get("optimizer", "AdamW")
        optimizer = getattr(optim, optimizer)

        if self.config["optim"].get("weight_decay", 0) > 0:
            # Do not regularize bias etc.
            params_decay = []
            params_no_decay = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "embedding" in name:
                        params_no_decay += [param]
                    elif "frequencies" in name:
                        params_no_decay += [param]
                    elif "bias" in name:
                        params_no_decay += [param]
                    else:
                        params_decay += [param]

            self.optimizer = optimizer(
                [
                    {"params": params_no_decay, "weight_decay": 0},
                    {
                        "params": params_decay,
                        "weight_decay": self.config["optim"]["weight_decay"],
                    },
                ],
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )
        else:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.config["optim"]["lr_initial"],
                **self.config["optim"].get("optimizer_params", {}),
            )

    def load_extras(self):
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])  # for now no decay
        pass

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        current_metric = aggregate_metric(val_metrics[primary_metric])
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            # Log best model
            if self.logger is not None:
                self.logger.save_model(f"{self.run_dir}/checkpoints/best_checkpoint.pt")

    def train(self, disable_eval_tqdm=False):
        # ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        # primary_metric = self.config["task"].get(
        #     "primary_metric", self.evaluator.task_primary_metric[self.name]
        # )
        # TODO: If we want to have other primary metrics, would have to change this
        primary_metric = "loss"
        self.best_val_metric = np.inf

        self.metrics = {"energy_loss": [], "forces_loss": []}
        self.step = 0
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.epoch = epoch
            for batch in self.train_loader:
                self.model.train()

                # Forward, loss, backward.
                out = self._forward(batch)
                loss, losses = self._compute_loss(out, batch)
                self._backward(loss)

                # Compute metrics.
                self.metrics = self._compute_metrics(out, batch, self.metrics)

                # Log metrics.
                log_dict = {k: aggregate_metric(self.metrics[k]) for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if self.step % self.print_every == 0:
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {"energy_loss": [], "forces_loss": []}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                # if self.scheduler.scheduler_type == "ReduceLROnPlateau"
                #     if self.step % eval_every == 0:
                #         self.scheduler.step(
                #             metrics=val_metrics[primary_metric]["metric"],
                #         )
                # else:
                #     self.scheduler.step()
                self.scheduler.step()
                self.step += 1

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        # Get predictions of best model
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "best_checkpoint.pt")["state_dict"])

        # Save best performance
        if self.task_config.get("test", True):
            self.validate(
                split="test",
                disable_tqdm=False,
            )
        # Save best predictions
        if "predict" in self.task_config:
            array_dict = {}
            for split in self.task_config["predict"]:
                if split == "train":
                    loader = self.train_loader
                elif split == "val":
                    loader = self.val_loader
                elif split == "test":
                    loader = self.test_loader

                out = self.predict(loader, disable_tqdm=False)
                out = {k: torch_tensor_to_npy(v) for k, v in out.items()}
                for key, val in out.items():
                    array_dict[f"{split}_{key}"] = val
            with open(self.predictions_dir / "predictions.npz", "wb") as f:
                np.savez(f, **array_dict)

            self.logger.log_predictions(self.predictions_dir)

    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False):
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader

        metrics = {"energy_loss": [], "forces_loss": []}
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=disable_tqdm,
        ):
            # Forward.
            out = self._forward(batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, metrics)

            log_dict = {k: aggregate_metric(metrics[k]) for k in metrics}
            log_dict.update({"epoch": self.epoch})
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

            if self.logger is not None:
                self.logger.log(
                    log_dict,
                    step=self.step,
                    split=split,
                )

        return metrics

    def _compute_metrics(self, out, batch_list_or_batch, metrics):
        # NOTE: Removed some additional things we probably want
        if not isinstance(batch_list_or_batch, list):
            batch_list = [batch_list_or_batch]
        else:
            batch_list = batch_list_or_batch

        losses = dict()

        # Energy loss.
        energy_target = torch.cat([batch.y.to(self.device) for batch in batch_list], dim=0).float()
        losses["energy"] = self.loss_fn["energy"](self.normalizers["target"].denorm(out["energy"]), energy_target)
        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0).float()
            losses["forces"] = self.loss_fn["forces"](
                self.normalizers["grad_target"].denorm(out["forces"]), force_target
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

    def _forward(self, batch):
        batch.to(self.device)
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces = self.model(batch)
        else:
            out_energy = self.model(batch)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

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

        losses = dict()

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

    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        disable_tqdm=False,
    ):
        logging.info("Predicting.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
                torch_geometric.loader.dataloader.DataLoader,
            ),
        )

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        predictions = {"energy": [], "forces": []}
        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=disable_tqdm,
        ):
            out = self._forward(batch)
            # denorm
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
            predictions["energy"].extend(out["energy"].detach())
            predictions["forces"].extend(out["forces"].detach())

        predictions["energy"] = torch.stack(predictions["energy"])
        predictions["forces"] = torch.stack(predictions["forces"])

        return predictions

    def save(
        self,
        metrics=None,
        checkpoint_file="checkpoint.pt",
        training_state=True,
    ):
        if training_state:
            config = {
                "epoch": self.epoch,
                "step": self.step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.scheduler.state_dict() if self.scheduler.scheduler_type != "Null" else None,
                "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
                "config": self.config,
                "val_metrics": metrics,
                # "ema": self.ema.state_dict() if self.ema else None,
                # "amp": self.scaler.state_dict()
                # if self.scaler
                # else None,
                "best_val_metric": self.best_val_metric,
                # "primary_metric": self.config["task"].get(
                #     "primary_metric",
                #     self.evaluator.task_primary_metric[self.name],
                # ),
            }
            save_checkpoint(
                config,
                checkpoint_dir=str(self.checkpoint_dir),
                checkpoint_file=checkpoint_file,
            )
        else:
            config = {
                "state_dict": self.model.state_dict(),
                "normalizers": {key: value.state_dict() for key, value in self.normalizers.items()},
                "config": self.config,
                "val_metrics": metrics,
                # "amp": self.scaler.state_dict()
                # if self.scaler
                # else None,
            }
            ckpt_path = save_checkpoint(
                config,
                checkpoint_dir=str(self.checkpoint_dir),
                checkpoint_file=checkpoint_file,
            )
            return ckpt_path
        return None
