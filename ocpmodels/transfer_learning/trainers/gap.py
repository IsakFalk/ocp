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


class GAPTrainer(BaseTrainer):
    def __init__(
        self,
        dataset_config,
        model_config,
        logger_config,
        print_every=10,
        seed=None,
        cpu=False,
        name="GAPtrainer",
        run_dir="checkpoints",
        is_debug=False,
    ):
        self.dataset_config = copy.deepcopy(dataset_config)  # Config for dataset
        self.model_config = copy.deepcopy(model_config)  # Config for model
        self.logger_config = copy.deepcopy(logger_config)  # Config for logger
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
        self.aux_dir = self.base_path / "aux"
        self.aux_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug

        # Not needed here
        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        self.config = {
            "model_attributes": self.model_config,
            "logger": self.logger_config,
            "name": name,
            "timestamp_id": self.timestamp_id,
            "dataset": self.dataset_config,
        }

        pprint(self.config)

        self.load()

    @staticmethod
    def _build_cmd_string(kwargs):
        kw_string = ""
        for key, val in kwargs.items():
            if isinstance(val, list):
                kw_string += f"{key}=" + "{" + " ".join([f"{x:.1e}" for x in val]) + "} "
            else:
                kw_string += f"{key}={val} "
        return kw_string.rstrip(" ")

    def load(self):
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_model()
        self.load_loss()

    def load_model(self):
        pass

    def load_loss(self):
        # TODO: Allow for different losses
        self.loss_fn = {
            "energy": nn.MSELoss(),
            "forces": nn.MSELoss(),
        }

    def train(self):
        self.gap_kwargs = self.config["model_attributes"]["gap_params"]
        self.gap_name = self.config["model_attributes"]["gap_params"].pop("name")
        self.gap_kw_string = self._build_cmd_string(self.gap_kwargs)
        self.gap_kw_string = "gap=" + "{" + f"{self.gap_name} " + self.gap_kw_string + "}"

        self.other_kwargs = self.config["model_attributes"]["other_params"]
        self.other_kw_string = self._build_cmd_string(self.other_kwargs)

        self.gap_fit_kw_string = " ".join([self.gap_kw_string, self.other_kw_string])

        # This runs the gap_fit command line tool from python
        # essentially this is a .fit() method from the command line
        cmd = ["gap_fit"]
        cmd.append("do_copy_at_file=F")
        cmd.append("sparse_separate_file=T")
        cmd.append(f"at_file={self.dataset_config['train']['src']}")
        cmd.append(f"gap_file={self.aux_dir / 'gap_train_output.xml'}")
        cmd.append(self.gap_fit_kw_string)
        subprocess.run(cmd, check=True)

    def validate(self, split="val"):
        metrics = {"energy_loss": [], "forces_loss": []}
        predictions = self.predict(split)
        # Forward.
        metrics = self._compute_metrics(predictions, self.datasets[split], metrics)

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
        # NOTE: we do not do any normalization for GAP
        energy_target = torch.tensor(np.array([batch.y for batch in batch_list])).float()
        losses["energy"] = self.loss_fn["energy"](out["energy"], energy_target.reshape(-1))
        # Force loss.
        force_target = torch.cat([batch.force.to(self.device) for batch in batch_list], dim=0).float()
        losses["forces"] = self.loss_fn["forces"](out["forces"].reshape(-1, 3), force_target.reshape(-1, 3))

        # Note that loss is normalized
        metrics["energy_loss"].append(losses["energy"].item())
        if self.config["model_attributes"].get("regress_forces", True):
            metrics["forces_loss"].append(losses["forces"].item())
        return metrics

    def predict(
        self,
        split,
    ):
        logging.info(f"Predicting on {split}.")
        # Here we predict using the trained parameters from
        # gap_train_output.xml
        # and using the dataset path we generate predictions
        # We finally read these in again
        predictions = {"energy": [], "forces": []}
        data_path = self.dataset_config[split]["src"]
        cmd = ["quip"]
        cmd.append("E=T")
        cmd.append("F=T")
        cmd.append(f"atoms_filename={data_path}")
        cmd.append(f"param_filename={self.aux_dir / 'gap_train_output.xml'}")
        pred_file = self.aux_dir / f"{split}_predictions.xyz"
        cmd.append(f"| grep AT | sed 's/AT//' > {pred_file}")
        _cmd = ""
        for c in cmd:
            _cmd += c + " "
        subprocess.run(_cmd, check=True, shell=True)
        pred = ase.io.read(self.aux_dir / f"{split}_predictions.xyz", index=":")
        pred_energy = torch.tensor(np.array([x.get_potential_energy() for x in pred]))
        # NOTE: GAP uses force instead of forces (using get_forces is data leakage!!)
        forces = []
        for atom in pred:
            forces.append(atom.arrays["force"])
        pred_forces = torch.tensor(np.array(forces))
        predictions = {
            "energy": pred_energy,
            "forces": pred_forces,
        }

        return predictions