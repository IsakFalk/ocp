"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import warnings

warnings.filterwarnings("ignore", module="torch_geometric")

from ocpmodels.common.utils import setup_logging
from ocpmodels.transfer_learning.common.flags import flags
from ocpmodels.transfer_learning.common.utils import get_config
from ocpmodels.transfer_learning.loaders import BaseLoader  # noqa: F401
from ocpmodels.transfer_learning.runners import (
    FTGNNRunner,
    GAPRunner,
    GNNRunner,
    MEKRRGNNRunner,
    MEKRRRunner,
)

if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = get_config(args.config_yml)

    if config["runner"] == "GNN":  # Supervised learning
        runner = GNNRunner(config, args)
    elif config["runner"] == "FTGNN":
        runner = FTGNNRunner(config, args)
    elif config["runner"] == "MEKRR":  # Transfer learning
        runner = MEKRRRunner(config, args)
    elif config["runner"] == "MEKRRGNN":  # Transfer learning
        runner = MEKRRGNNRunner(config, args)
    elif config["runner"] == "GAP":  # Supervised learning
        runner = GAPRunner(config, args)
    elif config["runner"] == "GDML":  # Supervised learning
        raise NotImplementedError
    elif config["runner"] == "BPNN":  # Supervised learning
        raise NotImplementedError
    else:
        raise NotImplementedError

    runner.setup()
    runner.run()
    logging.info("Done!")
    logging.info(f"Results saved to: {runner.trainer.base_path}")
