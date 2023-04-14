from abc import ABC, abstractmethod

import numpy as np

from ocpmodels.transfer_learning.common.utils import torch_tensor_to_npy
from ocpmodels.transfer_learning.trainers import GNNTrainer, MEKRRTrainer


class BaseRunner(ABC):
    def __init__(self, config, run_args):
        self.config = config
        self.run_args = run_args

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass


class MEKRRRunner(BaseRunner):
    def setup(self):
        self.trainer = MEKRRTrainer(
            self.config["dataset"],
            self.config["kernel"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
        )
        # TODO: add checkpoint resuming

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train()
        if self.config["task"].get("validate", True):
            self.trainer.validate(
                split="val",
            )
        if self.config["task"].get("test", True):
            self.trainer.validate(
                split="test",
            )
        if "predict" in self.config["task"]:
            array_dict = {}
            for split in self.config["task"]["predict"]:
                if split == "train":
                    dataset = self.trainer.train_dataset
                elif split == "val":
                    dataset = self.trainer.val_dataset
                elif split == "test":
                    dataset = self.trainer.test_dataset
                num_atoms = self.trainer.config["dataset"][f"{split}"]["num_atoms"]
                out = self.trainer.predict(dataset, num_atoms)
                out = {k: torch_tensor_to_npy(v) for k, v in out.items()}
                for key, val in out.items():
                    array_dict[f"{split}_{key}"] = val

            with open(self.trainer.predictions_dir / "predictions.npz", "wb") as f:
                np.savez(f, **array_dict)

            if not self.trainer.is_debug:
                self.trainer.logger.log_predictions(self.trainer.predictions_dir)


class GNNRunner(BaseRunner):
    def setup(self):
        self.trainer = GNNTrainer(
            self.config["task"],
            self.config["model"],
            self.config["dataset"],
            self.config["optim"],
            self.config["logger"],
            print_every=self.run_args.print_every,
            seed=self.run_args.seed,
            cpu=self.run_args.cpu,
            name=self.config["logger"]["name"],
            run_dir=self.run_args.run_dir,
            is_debug=self.run_args.debug,
        )
        # TODO: add checkpoint resuming

    def run(self):
        if self.config["task"].get("train", True):
            self.trainer.train(
                disable_eval_tqdm=self.config.get("hide_eval_progressbar", False),
            )
