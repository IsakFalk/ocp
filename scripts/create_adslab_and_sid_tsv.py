from pathlib import Path
import os
import pickle as pkl
import pandas as pd
import argparse
from glob import glob
import logging

import ocpmodels

def create_adslabid_and_sid(datadir: Path, num: int, data_mapping: dict) -> pd.DataFrame:
    """Takes the top is2re_adsorbate path and outputs the cat_id for all randomXXX strings"""

    # Extract the first column of system.txt containing all sids
    sids = list(
        pd.read_csv(
            Path(datadir) / f"{num}" / "system.txt",
            header=None,
            index_col=0,
        ).index
    )

    # Build a dataframe with columns sid, cat_id and ads_id
    s_ids = []
    cat_ids = []
    ads_ids = []
    for sid in sids:
        s_ids.append(sid)
        cat_ids.append(data_mapping[sid]["bulk_id"])
        ads_ids.append(data_mapping[sid]["ads_id"])
    return pd.DataFrame({"sid": s_ids, "cat_id": cat_ids, "ads_id": ads_ids})

def map_pairs_for_all_is2re(datadir: str) -> dict:
    datadir = Path(datadir)
    try:
        with open(datadir / "auxiliary" / "oc20_data_mapping.pkl", "rb") as f:
            data_mapping = pkl.load(f)
    except:
        logging.error("Missing data mapping file in data/auxiliary directory")
        return

    ads_datadir = datadir / "is2re_adsorbate"
    # Find all subsystem directories and extract the dataframe
    # finally output the dataframe to a tsv file
    subdirs = [f for f in ads_datadir.iterdir() if f.is_dir()]
    dfs = []
    for subdir in subdirs:
        # Get number, this will be combined to a path in the function
        num = subdir.name
        logging.info(f"Processing sub-directory is2re_adsorbate/{subdir}")
        # Open the adsorbate pairs file and add the new pairs
        dfs.append(create_adslabid_and_sid(ads_datadir, num, data_mapping))
    df = pd.concat(dfs)
    df.to_csv(datadir / "auxiliary" / "adslabid_and_sid.tsv", sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'ocpmodels/data'",
    )

    args, _ = parser.parse_known_args()
    map_pairs_for_all_is2re(
        args.data_path,
    )
