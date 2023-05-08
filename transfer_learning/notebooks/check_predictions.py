"""Playground for testing out new ideas.

Currently this shows how to load a model and get the intermediate representations
from the model. We use this to output the distance and kernel matrices for the system over time
and atoms.
"""
from pathlib import Path

import pandas as pd
import torch

### Load data
DATA_PATH = Path("data/temp/predictions")


def read_in_data(path):
    with open(path, "r") as f:
        header = f.readline()
    columns = header.split()
    columns = columns[2:]
    return pd.read_csv(path, sep=" ", names=columns, skiprows=[0], header=None)


e_df = read_in_data(DATA_PATH / "deepmd_log.e.out")
f_df = read_in_data(DATA_PATH / "deepmd_log.f.out")
e_df
f_df

DATA_PATH = Path("data/luigi/formula=Fe45N2_surface=111_driver=qe-md_bias=-.xyz")
from ocpmodels.transfer_learning.common.utils import (
    ATOMS_TO_GRAPH_KWARGS,
    load_xyz_to_pyg_batch,
    load_xyz_to_pyg_data,
)

raw_data, data_batch, num_frames, num_atoms = load_xyz_to_pyg_batch(DATA_PATH, ATOMS_TO_GRAPH_KWARGS["schnet"])
num_frames
num_atoms

from ocpmodels.transfer_learning.modules.evaluator import Evaluator


def create_output_dicts(e_df, f_df, num_atoms, num_frames):
    pred_dict = {}
    pred_dict["energy"] = torch.tensor(e_df["pred_e"].values).reshape(num_frames)
    pred_dict["forces"] = torch.tensor(f_df[["pred_fx", "pred_fy", "pred_fz"]].values).reshape(num_frames, num_atoms, 3)

    target_dict = {}
    target_dict["energy"] = torch.tensor(e_df["data_e"].values).reshape(num_frames)
    target_dict["forces"] = torch.tensor(f_df[["data_fx", "data_fy", "data_fz"]].values).reshape(
        num_frames, num_atoms, 3
    )
    return pred_dict, target_dict


pred_dict, target_dict = create_output_dicts(e_df, f_df, num_atoms, 240)

evaluator = Evaluator()
evaluator.eval(pred_dict, target_dict)
