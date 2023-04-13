
# Table of Contents

-   [Transfer Learning](#org74c62aa)
    -   [Training](#org49ba22d)
    -   [Repo](#org2b0a0a7)
    -   [Config files](#orga068f14)
    -   [Logging](#org7ca5507)



<a id="org74c62aa"></a>

# Transfer Learning

This subrepo follows approximately how OCP is structuring theirs.


<a id="org49ba22d"></a>

## Training

There&rsquo;s a `main.py` file which is the entrypoint for all training. **Run this file from the parent directory** `ocp`. You run this
from the command line and running `python main.py --help` yields the output

    usage: main.py [-h] --config-yml CONFIG_YML [--debug] [--run-dir RUN_DIR] [--print-every PRINT_EVERY] [--seed SEED] [--cpu]
    
    Graph Networks for Electrocatalyst Design
    
    optional arguments:
      -h, --help            show this help message and exit
      --config-yml CONFIG_YML
                            Path to a config file listing data, model, optim parameters.
      --debug               Whether this is a debugging run or not
      --run-dir RUN_DIR     Directory to store checkpoint/log/result directory
      --print-every PRINT_EVERY
                            Log every N iterations (default: 10)
      --seed SEED           Seed for torch, cuda, numpy
      --cpu                 Run CPU only training

The `main.py` file works as follows

1.  Set up logging
2.  Get flags (flags defined in `ocpmodels.transfer_learning.common.flags` and can be added there as well)
3.  From the yaml config file passed it builds the `config`
4.  The `config` object is simply the yaml file represented as a `dict`
5.  The tuple `(config, args)` is passed to a **Runner**
    -   A runner is an abstraction for training an implements the methods `setup` and `run`.
    -   Depending on the method, these runners may look very different. If you want
        to implement something, just implement it as you like and then implement the runner
    -   Dispatching to the right runner is done through the `runner` argument of
        the yaml file, e.g. the `GNNRunner` which implements the GNN training,
        validation and prediction is dispatched by putting `runner: GNN` in the yaml config file
6.  The runner is run, by first calling `runner.setup()` and finally `runner.run()`
    -   `runner.run()` simply runs the algorithm and does what you want it to do
        (e.g. training, prediction and so on.)


<a id="org2b0a0a7"></a>

## Repo

The code is under `ocp/ocpmodels/transfer_learning` as it allows us to use this as a package. To use it, do

    pip install .

which will make the `ocpmodels` and the submodule `ocpmodels.transfer_learning` available.


<a id="orga068f14"></a>

## Config files

The config files specify such things as datasets and methods to use. The current
example can be found in `configs/gnn/schnet.yaml`. Please put the configs for
different runners under a corresponding subdirectory (here `gnn`). This config
without comments is shown below

    dataset:
      train:
        src: data/luigi/example-traj-Fe-N2-111.xyz
        normalize_labels: True
        target_mean: -34662.8389
        target_std: 0.6950
        grad_target_mean: 0.0
        grad_target_std: 0.5969
      val:
        src: data/luigi/example-traj-Fe-N2-111.xyz
      test:
        src: data/luigi/example-traj-Fe-N2-111.xyz
    
    logger:
      tags: []
      name: "schnet-very-good"
    
    model:
      name: "schnet"
      hidden_channels: 4
      num_filters: 4
      num_gaussians: 8
      num_interactions: 1
      otf_graph: True
    
    optim:
      batch_size: 8
      eval_batch_size: 8
      num_workers: 4
      eval_every: 5
      force_loss_coefficient: 0.5
      optimizer: AdamW
      lr_initial: 1.e-4
      optimizer_params: {"amsgrad": True}
      weight_decay: 0.0
      scheduler: 'Null'
      max_epochs: 20
    
    hide_eval_progressbar: False
    
    task:
      train: True
      validate: True
      test: True
      predict: ["train", "val", "test"]
    
    runner: "GNN"

Note that this config may change depending on the runner, but the top level keys
of `dataset, logger, model, optim, hide_eval_progressbar, tasks, runner` is a
good way to structure your config and may even be necessary.


<a id="org7ca5507"></a>

## Logging

We would like to log everything to wandb so that we can have it available to
look through. Let me know if you need help with this.

