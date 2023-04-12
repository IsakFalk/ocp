"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod

import datetime

import wandb

class Logger(ABC):
    """Generic class to interface with various logging modules, e.g. wandb,
    tensorboard, etc.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def watch(self, model):
        """
        Monitor parameters and gradients.
        """
        pass

    def log(self, update_dict, step=None, split=""):
        """
        Log some values.
        """
        assert step is not None
        if split != "":
            new_dict = {}
            for key in update_dict:
                new_dict[f"{split}/{key}"] = update_dict[key]
            update_dict = new_dict
        return update_dict

    @abstractmethod
    def log_plots(self, plots):
        pass

    @abstractmethod
    def log_predictions(self, predictions):
        pass

    @abstractmethod
    def mark_preempting(self):
        pass


class WandBLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        project = "mlcompchem"
        entity = "mlcompchem"

        # For unique logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.timestamp_id = timestamp

        wandb.init(
            config=self.config,
            save_code=True,
            id=self.timestamp_id,
            name=self.config["logger"].get("name", "noname"),
            entity=entity,
            project=project,
            resume="allow",
            tags=self.config["logger"].get("tags", []),
        )

    def watch(self, model):
        wandb.watch(model)

    def log(self, update_dict, step=None, split=""):
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption=""):
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_predictions(self, predictions):
        wandb.log({"predictions": predictions})

    def mark_preempting(self):
        wandb.mark_preempting()

    def save_model(self, path):
        wandb.save(path)
