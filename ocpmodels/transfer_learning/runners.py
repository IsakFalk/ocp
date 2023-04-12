from abc import ABC, abstractmethod

from ocpmodels.transfer_learning.common.utils import torch_tensor_to_npy
from ocpmodels.transfer_learning.trainers import GNNTrainer


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


class GNNRunner(BaseRunner):
    def setup(self):
        self.trainer = GNNTrainer(
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
        if self.config["task"].get("validate", True):
            self.trainer.validate(
                split="val",
                disable_tqdm=self.config.get("hide_eval_progressbar", False),
            )
        if self.config["task"].get("test", True):
            self.trainer.validate(
                split="test",
                disable_tqdm=self.config.get("hide_eval_progressbar", False),
            )
        if hasattr(self.config["task"], "predict"):
            log_dict = dict()
            for split in self.config["task"]["predict"]:
                log_dict[f"{split}"] = self.trainer.predict(
                    split=split,
                    disable_tqdm=self.config.get("hide_eval_progressbar", False),
                )
                log_dict[f"{split}"] = {k: torch_tensor_to_npy(v) for k, v in log_dict[f"{split}"].items()}
            self.trainer.logger.log_predictions(log_dict)
