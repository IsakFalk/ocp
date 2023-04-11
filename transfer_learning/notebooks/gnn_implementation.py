"""GNN implementation for learning

We basically take the parts from the OCP repo and fix it to use here"""

from pathlib import Path
import copy

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric

from transfer_learning.transfer_learning.common.utils import load_xyz_to_pyg_data, ATOMS_TO_GRAPH_KWARGS

#######################
# Functions and stuff #
#######################
from transfer_learning.transfer_learning.loaders import BaseLoader


## Trainer
torch.set_default_dtype(torch.float32)
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import datetime
import errno
import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
#from torch.utils.data import DataLoader
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
from ocpmodels.modules.scheduler import LRScheduler

from pprint import pprint

class Trainer:
    def __init__(
        self,
        model_config,
        dataset_config,
        optimizer_config,
        print_every=10,
        seed=None,
        logger="wandb",
        cpu=False,
        name="trainer",
    ):
        self.model_config = copy.deepcopy(model_config)  # Config for model
        self.optimizer = copy.deepcopy(optimizer_config)  # Config for optimizer
        self.optimizer["energy_loss_coefficient"] = 1 - self.optimizer.get("force_loss_coefficient", 0.0)
        self.model_config["regress_forces"] = True
        self.dataset_config = copy.deepcopy(dataset_config)  # Config
        self.cpu = cpu
        self.print_every = print_every
        self.seed = seed
        self.epoch = 0
        self.step = 0

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        #logger_name = logger if isinstance(logger, str) else logger["name"]
        self.logger = None
        self.config = {
            "model": self.model_config.pop("name"),
            "model_attributes": self.model_config,
            "optim": self.optimizer,
            "name": name,
            "dataset": self.dataset_config,
        }
        pprint(self.config)

        self.load()

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
                y = torch.tensor([data.y for data in trainer.train_dataset])
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
                forces = torch.cat([data.force.clone().detach() for data in trainer.train_dataset], dim=0)
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
            _config,
            regress_forces=_config["model_attributes"].pop("regress_forces"),
            seed=self.seed,
            cpu=self.cpu
        ).model.to(self.device)

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
        self.scheduler = LRScheduler(self.optimizer, self.config["optim"]) # for now no decay
        pass

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        current_metric = torch.tensor(val_metrics[primary_metric]).mean().item()
        if current_metric < self.best_val_metric:
            self.best_val_metric = current_metric
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def train(self, disable_eval_tqdm=False):
        #ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        # primary_metric = self.config["task"].get(
        #     "primary_metric", self.evaluator.task_primary_metric[self.name]
        # )
        # TODO: If we want to have other primary metrics, would have to change this
        primary_metric = "loss"
        self.best_val_metric = np.inf

        self.metrics = {
            "energy": [],
            "forces": []
        }
        for epoch in range(self.config["optim"]["max_epochs"]):
            self.epoch = epoch
            for step, batch in enumerate(self.train_loader):
                self.step = step
                self.model.train()

                # Forward, loss, backward.
                out = self._forward(batch)
                loss, losses = self._compute_loss(out, batch)
                self._backward(loss)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out, losses, self.metrics
                )

                # Log metrics.
                log_dict = {k: torch.tensor(self.metrics[k]).clone().detach().mean() for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.print_every == 0
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {
                        "energy": [],
                        "forces": []
                    }

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

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

                # if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                #     if self.step % eval_every == 0:
                #         self.scheduler.step(
                #             metrics=val_metrics[primary_metric]["metric"],
                #         )
                # else:
                #     self.scheduler.step()
                self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False):
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader

        metrics = {
            "energy": [],
            "forces": []
        }
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=disable_tqdm,
        ):
            # Forward.
            out = self._forward(batch)
            loss, losses = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, losses, metrics)
            metrics["loss"] = loss.clone().detach().item()

        return metrics

    def _compute_metrics(self, out, losses, metrics):
        # TODO: Allow for additional metrics
        metrics["energy"].append(losses["energy"].item())
        if self.config["model_attributes"].get("regress_forces", True):
            metrics["forces"].append(losses["forces"].item())
        metrics["loss"] = (
            self.config["optim"].get("energy_loss_coefficient") * losses["energy"]
            + self.config["optim"].get("force_loss_coefficient") * losses["forces"]
        )
        return metrics

    def _forward(self, batch):
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
        losses["energy"] = (
            self.loss_fn["energy"](out["energy"], energy_target)
        )
        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ).float()
            force_target = self.normalizers["grad_target"].norm(force_target)
            losses["forces"] = (
                self.loss_fn["forces"](out["forces"], force_target)
            )

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
        results_file=None,
        disable_tqdm=False,
    ):
        logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
                torch_geometric.data.DataLoader
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
            out["energy"] = self.normalizers["target"].denorm(
                out["energy"]
            )
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )
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
            return save_checkpoint(
                {
                    "epoch": self.epoch,
                    "step": self.step,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.scheduler.state_dict()
                    if self.scheduler.scheduler_type != "Null"
                    else None,
                    "normalizers": {
                        key: value.state_dict()
                        for key, value in self.normalizers.items()
                    },
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
                    #),
                },
                #checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                checkpoint_file=checkpoint_file,
            )
        else:
            ckpt_path = save_checkpoint(
                {
                    "state_dict": self.model.state_dict(),
                    "normalizers": {
                        key: value.state_dict()
                        for key, value in self.normalizers.items()
                    },
                    "config": self.config,
                    "val_metrics": metrics,
                    #"amp": self.scaler.state_dict()
                    # if self.scaler
                    # else None,
                },
                # checkpoint_dir=self.checkpoint_dir,
                checkpoint_file=checkpoint_file,
            )
            return ckpt_path
        return None

    def save_results(self, predictions, results_file, keys=["energy", "forces"]):
        return

############
# Notebook #
############

# Load config for run
with open("./transfer_learning/notebooks/configs/schnet.yaml", "r") as f:
    original_config = yaml.safe_load(f)
config = copy.deepcopy(original_config)

# from pprint import pprint
# pprint(model_config)
# pprint(config["dataset"])
# pprint(config["optim"])

# Set up trainer
trainer = Trainer(
    config["model"],
    config["dataset"],
    config["optim"],
    cpu=True
)

for batch in trainer.train_loader:
    break


### Works
out = trainer._forward(batch)
loss, losses = trainer._compute_loss(out, batch)
trainer._backward(loss)

### Now train
trainer.train()
trainer.validate()
trainer.predict(trainer.test_loader)
