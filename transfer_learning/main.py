"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings

warnings.filterwarnings("ignore", module="torch_geometric")

from ocpmodels.common.utils import setup_logging
from ocpmodels.transfer_learning.common.flags import flags
from ocpmodels.transfer_learning.common.utils import get_config
from ocpmodels.transfer_learning.loaders import BaseLoader  # noqa: F401
from ocpmodels.transfer_learning.runners import GNNRunner, MEKRRRunner

if __name__ == "__main__":
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = get_config(args.config_yml)

    if config["runner"] == "GNN":
        runner = GNNRunner(config, args)
    elif config["runner"] == "MEKRR":
        runner = MEKRRRunner(config, args)

    runner.setup()
    runner.run()
