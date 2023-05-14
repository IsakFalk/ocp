#!/usr/bin/env python3

"""Extract the pickle key-value data descriptions for adsorbate+catalyst systems

See: https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md#oc20-mappings"""

import hashlib
import logging
from pathlib import Path

import requests

LINKS = {
    "oc20_data_mapping": "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl",
    "mapping_adslab_slab": "https://dl.fbaipublicfiles.com/opencatalystproject/data/mapping_adslab_slab.pkl",
}

MD5HASHS = {
    "oc20_data_mapping": "01c879067a05b4288055a1fdf821e068",
    "mapping_adslab_slab": "079041076c3f15d18ecb5d17c509cdfe",
}

DATA_PATH = Path(".") / "data" / "auxiliary"


def check_hash(binary_string, hash_string):
    return hashlib.md5(binary_string).hexdigest() == hash_string


for name, url in LINKS.items():
    # Get raw binary data from url
    r = requests.get(url)
    data = r.content
    # Check that checksum is correct
    assert check_hash(data, MD5HASHS[name]), f"MD5 Checksum not correct for {name}"
    # Write to file
    with open(DATA_PATH / f"{name}.pkl", "wb") as f:
        f.write(data)
    logging.info(f"Successfully wrote {name} to pickle file in {DATA_PATH}")
