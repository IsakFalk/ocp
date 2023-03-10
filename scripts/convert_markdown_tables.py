"""Convert model markdown tables to tsv tables."""
from pathlib import Path
import os
from io import StringIO
import re
import pickle as pkl
import pandas as pd
import argparse
from glob import glob
import logging

import ocpmodels


def transform_is2re_table(datapath):
    """Transforms the IS2RE table to a tsv file."""
    with open(datapath / "model_table_is2re.md", "r") as f:
        tab = f.read()

    # Preprocess
    tab = tab[1:]
    tab = tab.replace("\n|---\t|---\t|---\t|---\t|\n", "\n")
    tab = tab.replace("\|", "|")
    tab = tab.replace("\t", "")
    tab = tab.replace("|\n|", "\n")
    tab = tab.replace(" ", "")
    tab = tab.replace("github.com", "raw.githubusercontent.com").replace(
        "/blob/main", "/main"
    )  # for downloading
    tab = tab[:-2]  # Remove last |\n
    tab = tab.split("\n")
    # Remove the header since we'll create it implicitly, and null string
    tab = tab[1:]

    line_of_dicts = []
    for line in tab:
        line = line.split("|")
        data_dict = {}
        data_dict["model"] = line[0]
        data_dict["split"] = line[1]
        data_dict["val_id_energy_mae"] = line[-1]
        line = line[2:-1]

        keys = ["checkpoint", "config", "scalefile"]
        for entry in line:
            key = re.search(r"^\[(.*)\]", entry).group(1)
            match = re.search(r".*\((.*)\).*", entry)
            data_dict[key] = match.group(1)
            keys.remove(key)
        for key in keys:
            data_dict[key] = None
        line_of_dicts.append(data_dict)

    df = pd.DataFrame(line_of_dicts)
    df.to_csv(datapath / "model_table_is2re.tsv", sep="\t", index=False)


def transform_s2ef_optimized_efwt_table(datapath):
    """Transforms the S2EF table to a tsv file."""
    with open(datapath / "model_table_s2ef_optimized_efwt.md", "r") as f:
        tab = f.read()

    # Preprocess by removing the first
    tab = tab[1:]
    tab = tab.replace("\n|---\t|---\t|---\t|---\t|---\t|\n", "\n")
    tab = tab.replace("\|", "|")
    tab = tab.replace("\t", "")
    tab = tab.replace("|\n|", "\n")
    tab = tab.replace(" ", "")
    tab = tab.replace(
        "github.com", "raw.githubusercontent.com"
    )  # For downloading...
    tab = tab.replace("/blob/main", "/main")  # and for fixing the links
    tab = tab[:-2]  # Remove last |\n
    tab = tab.split("\n")

    # Remove the header since we'll create it implicitly, and null string
    tab = tab[1:]

    line_of_dicts = []
    for line in tab:
        line = line.split("|")
        data_dict = {}
        data_dict["model"] = line[0]
        data_dict["split"] = line[1]
        data_dict["val_id_energy_mae"] = line[-2]
        data_dict["val_id_energy_efwt"] = line[-1]
        line = line[2:-2]

        keys = ["checkpoint", "config", "scalefile"]
        for entry in line:
            key = re.search(r"^\[(.*)\]", entry).group(1)
            match = re.search(r".*\((.*)\).*", entry)
            data_dict[key] = match.group(1)
            keys.remove(key)
        for key in keys:
            data_dict[key] = None
        line_of_dicts.append(data_dict)

    df = pd.DataFrame(line_of_dicts)
    df.to_csv(
        datapath / "model_table_s2ef_optimized_efwt.tsv", sep="\t", index=False
    )


def transform_s2ef_optimized_force_table(datapath):
    """Transforms the S2EF table to a tsv file."""
    datapath = os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data")
    datapath = Path(datapath) / "auxiliary" / "oc20"
    with open(datapath / "model_table_s2ef_optimized_force.md", "r") as f:
        tab = f.read()

    # Preprocess
    tab = tab[1:]
    tab = tab.replace("\n|---\t|---\t|---\t|---\t|\n", "\n")
    tab = tab.replace("\|", "|")
    tab = tab.replace("\t", "")
    tab = tab.replace("|\n|", "\n")
    tab = tab.replace(" ", "")
    tab = tab.replace("github.com", "raw.githubusercontent.com").replace(
        "/blob/main", "/main"
    )  # for downloading
    tab = tab[:-3]  # Remove last |\n
    tab = tab.split("\n")
    # Remove the header since we'll create it implicitly, and null string
    tab = tab[1:]

    line_of_dicts = []
    for line in tab:
        line = line.split("|")
        data_dict = {}
        data_dict["model"] = line[0]
        data_dict["split"] = line[1]
        data_dict["val_id_energy_mae"] = line[-1]
        line = line[2:-1]

        keys = ["checkpoint", "config", "scalefile"]
        for entry in line:
            key = re.search(r"^\[(.*)\]", entry).group(1)
            match = re.search(r".*\((.*)\).*", entry)
            data_dict[key] = match.group(1)
            keys.remove(key)
        for key in keys:
            data_dict[key] = None
        line_of_dicts.append(data_dict)

    df = pd.DataFrame(line_of_dicts)
    df.to_csv(
        datapath / "model_table_s2ef_optimized_force.tsv",
        sep="\t",
        index=False,
    )


def main(args):
    datapath = args.data_path
    datapath = Path(datapath) / "auxiliary" / "oc20"
    logging.info("Transforming MD tables to tsv...")
    logging.info("Transforming IS2RE table...")
    transform_is2re_table(datapath)
    logging.info("Done!")
    logging.info("Transforming S2EF optimized force table...")
    transform_s2ef_optimized_efwt_table(datapath)
    logging.info("Done!")
    logging.info("Transforming S2EF optimized efwt table...")
    transform_s2ef_optimized_force_table(datapath)
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'ocpmodels/data'",
    )

    args, _ = parser.parse_known_args()
    main(args)
