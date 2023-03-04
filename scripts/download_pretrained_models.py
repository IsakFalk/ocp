"""Download pretrained model checkpoints and output them in the correct directory."""
from pathlib import Path
import os
from io import StringIO
import re
import pickle as pkl
import pandas as pd
import argparse
from glob import glob
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)

import pandas as pd

import ocpmodels

MODELS = {
    "s2ef_efwt": {
        "200k": ["CGCNN", "DimeNet", "SchNet", "DimeNet++"],
        "2M": [
            "CGCNN",
            "DimeNet",
            "SchNet",
            "DimeNet++",
            "SpinConv",
            "GemNet-dT",
            "GemNet-OC",
            "SCN",
            "SCN-t4-b2",
        ],
        "20M": ["CGCNN", "SchNet", "DimeNet++"],
        "All": [
            "CGCNN",
            "SchNet",
            "DimeNet++",
            "SpinConv",
            "GemNet-dT",
            "PaiNN",
            "GemNet-OC",
        ],
        "All+MD": ["GemNet-OC", "GemNet-OC-Large", "All+MD"],
    },
    "s2ef_force": {
        "All": ["SchNet", "DimeNet++", "DimeNet++-Large"],
        "20M+Rattled": ["DimeNet++"],
        "20M+MD": ["DimeNet++"],
    },
    "is2re": {
        "10k": ["CGCNN", "DimeNet", "SchNet", "DimeNet++"],
        "100k": ["CGCNN", "DimeNet", "SchNet", "DimeNet++"],
        "All": ["CGCNN", "DimeNet", "SchNet", "DimeNet++", "PaiNN"],
    },
}
# TODO: Add support for only outputting the command for running the model
def download_checkpoint(datapath, checkpointpath, model, task, split):
    """Download a pretrained model checkpoint from the OCP repo."""
    if task == "s2ef_efwt":
        path_end = "model_table_s2ef_optimized_efwt.tsv"
    elif task == "s2ef_force":
        path_end = "model_table_s2ef_optimized_force.tsv"
    elif task == "is2re":
        path_end = "model_table_is2re.tsv"
    else:
        raise ValueError(f"Task {task} not supported.")

    # Load df and extract all the necessary information.
    df = pd.read_csv(datapath / "auxiliary" / "oc20" / path_end, sep="\t")
    row = df.query(f"model == '{model}' and split == '{split}'")
    # Download the checkpoint and potentially the scalefile and output into the correct directory
    checkpoint_link = row["checkpoint"].item()
    scalefile_link = row["scalefile"].item()
    split = re.sub(r"\+(Rattled|MD)", r"", split).lower()
    # Create the correct output directory.
    output_checkpoint_dir_path = (
        checkpointpath
        / task
        / split
        / model.lower().replace("++", "_plus_plus")
    )
    output_checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    os.system(f"wget {checkpoint_link} -P {output_checkpoint_dir_path}")
    logging.info(f"Downloaded {model} for {task} {split}\n")
    logging.info(
        f"Checkpoint saved to {output_checkpoint_dir_path}/{checkpoint_link.split('/')[-1]}"
    )
    if not pd.isna(scalefile_link):
        os.system(f"wget {scalefile_link} -P {output_checkpoint_dir_path}")
        logging.info(
            f"Scalefile saved to {output_checkpoint_dir_path}/{scalefile_link.split('/')[-1]}"
        )
    logging.info(
        f"Config yaml find can be found at: ocp/{row['config'].item().split('main/')[-1]}"
    )


def main(args):
    datapath = Path(args.data_path)
    checkpointpath = Path(args.checkpoint_path)
    download_checkpoint(
        datapath, checkpointpath, args.model, args.task, args.split
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Specify path to data. Defaults to 'ocpmodels/data'",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join(
            os.path.dirname(ocpmodels.__path__[0]), "checkpoints"
        ),
        help="Specify path to save checkpoints and scalefile. Defaults to 'ocpmodels/checpoints'",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="s2ef_efwt",
        help="Specify task that model was trained on.",
        choices=["s2ef_efwt", "s2ef_force", "is2re"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="All",
        help="Specify split that model was trained on.",
        choices=[
            "200k",
            "2M",
            "20M",
            "All",
            "All+MD",
            "10k",
            "100k",
            "20M+Rattled",
            "20M+MD",
        ],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SchNet",
        help="Specify model checkpoint to download.",
        choices=[
            "CGCNN",
            "DimeNet",
            "SchNet",
            "DimeNet++",
            "SpinConv",
            "GemNet-dT",
            "GemNet-OC",
            "SCN",
            "SCN-t4-b2",
            "PaiNN",
            "DimeNet++-Large",
            "GemNet-OC-Large",
        ],
    )


    args, _ = parser.parse_known_args()
    main(args)
