import os
import subprocess
from pathlib import Path

from ase.io import read

os.chdir("../..")
os.getcwd()

TRAIN_DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
VAL_DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")
TEST_DATA_PATH = Path("data/luigi/example-traj-Fe-N2-111.xyz")

# Create output directory
output_dir = Path("data/luigi/gap_fit")
output_dir.mkdir(parents=True, exist_ok=True)


def build_cmd_string(kwargs):
    kw_string = ""
    for key, val in kwargs.items():
        if isinstance(val, list):
            kw_string += f"{key}=" + "{" + " ".join([f"{x:.1e}" for x in val]) + "} "
        else:
            kw_string += f"{key}={val} "
    return kw_string.rstrip(" ")


soap_kwargs = {
    "atom_sigma": 0.5,
    "l_max": 2,
    "n_max": 2,
    "cutoff": 6.0,
    "cutoff_transition_width": 1.0,
    "delta": 1.0,
    "covariance_type": "dot_product",
    "n_sparse": 1000,
    "zeta": 4,
    "energy_scale": 1.0,
    "atom_gaussian_width": 1.0,
}
soap_kw_string = build_cmd_string(soap_kwargs)
soap_kw_string = "soap " + soap_kw_string
gap_string = "gap={" + soap_kw_string + "}"

lmbda = 1.0e-6
other_kwargs = {
    "default_sigma": [0.0001, 0.002, 0.0, 0.0],
    # "default_kernel_regularisation": [lmbda, lmbda, lmbda, lmbda],
    "e0": 0.0,
}
other_kw_string = build_cmd_string(other_kwargs)

gap_fit_kw_string = " ".join([gap_string, other_kw_string])

cmd = ["gap_fit"]
cmd.append("do_copy_at_file=F")
cmd.append("sparse_separate_file=T")
cmd.append(f"at_file={TRAIN_DATA_PATH}")
cmd.append(f"gap_file={output_dir / 'gap_train_output.xml'}")
cmd.append(gap_fit_kw_string)

subprocess.run(cmd, check=True)

# Get predictions on train, val, and etst
for split in ["train", "val", "test"]:
    if split != "train":
        continue
    if split == "train":
        data_path = TRAIN_DATA_PATH
    elif split == "val":
        data_path = VAL_DATA_PATH
    elif split == "test":
        data_path = TEST_DATA_PATH
    cmd = ["quip"]
    cmd.append("E=T")
    cmd.append("F=T")
    cmd.append(f"atoms_filename={data_path}")
    cmd.append(f"param_filename={output_dir / 'gap_train_output.xml'}")
    # cmd.append(f"output_file={output_dir / f'{split}_predictions.xyz'}")
    pred_file = output_dir / f"{split}_predictions.xyz"
    cmd.append(f"| grep AT | sed 's/AT//' > {pred_file}")
    _cmd = ""
    for c in cmd:
        _cmd += c + " "
    subprocess.run(_cmd, check=True, shell=True)

import numpy as np

## Read in predictions
import torch

train_pred = read(output_dir / "train_predictions.xyz", index=":")
# val_pred = read(output_dir / "val_predictions.xyz", index=":")
# test_pred = read(output_dir / "test_predictions.xyz", index=":")
train_pred_energy = torch.tensor(np.array([x.get_potential_energy() for x in train_pred]))
train_pred_forces = torch.tensor(np.array([x.get_forces() for x in train_pred]))

with open(TRAIN_DATA_PATH, "r") as f:
    num_atoms = int(f.readline().strip("\n"))
