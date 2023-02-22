import argparse
import glob
import logging
import os

import ocpmodels

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS = {
    "s2ef": {
        "200k": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
        "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
        "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
        "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz",
        "rattled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar",
        "md": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar",
    },
    "is2re_adsorbate": {
        "*O": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/0.tar",
        "*H": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/1.tar",
        "*OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/2.tar",
        "*OH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/3.tar",
        "*C": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/4.tar",
        "*CH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/6.tar",
        "*CHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/7.tar",
        "*COH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/8.tar",
        "*CH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/9.tar",
        "*CH2*O": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/10.tar",
        "*CHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/11.tar",
        "*CH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/12.tar",
        "*OCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/13.tar",
        "*CH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/14.tar",
        "*CH4": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/15.tar",
        "*OHCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/16.tar",
        "*C*C": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/17.tar",
        "*CCO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/18.tar",
        "*CCH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/19.tar",
        "*CHCO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/20.tar",
        "*CCHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/21.tar",
        "*COCHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/22.tar",
        "*CCHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/23.tar",
        "*CCH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/24.tar",
        "*CH*CH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/25.tar",
        "CH2*CO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/26.tar",
        "*CHCHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/27.tar",
        "*CH*COH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/28.tar",
        "*COCH2O": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/29.tar",
        "*CHO*CHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/30.tar",
        "*COHCHO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/31.tar",
        "*COHCOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/32.tar",
        "*CCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/33.tar",
        "*CHCH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/34.tar",
        "*COCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/35.tar",
        "*CHCHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/38.tar",
        "*CCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/39.tar",
        "*CHOCHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/40.tar",
        "*COCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/41.tar",
        "*COHCHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/42.tar",
        "*OCHCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/44.tar",
        "*COHCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/45.tar",
        "*CHOHCH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/46.tar",
        "*CHCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/47.tar",
        "*OCH2CHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/48.tar",
        "*CHOCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/49.tar",
        "*COHCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/50.tar",
        "*CHOHCHOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/51.tar",
        "*CH2CH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/52.tar",
        "*OCH2CH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/53.tar",
        "*CHOHCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/54.tar",
        "*CH2CH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/55.tar",
        "*CHOHCH2OH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/56.tar",
        "*OHCH2CH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/57.tar",
        "*NH2N(CH3)2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/58.tar",
        "*ONN(CH3)2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/59.tar",
        "*OHNNCH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/60.tar",
        "*ONH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/62.tar",
        "*NHNH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/63.tar",
        "*N*NH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/65.tar",
        "*NO2NO2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/67.tar",
        "*N*NO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/68.tar",
        "*N2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/69.tar",
        "*ONNH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/70.tar",
        "*NH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/71.tar",
        "*NH3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/72.tar",
        "*NONH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/73.tar",
        "*NH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/74.tar",
        "*NO2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/75.tar",
        "*NO": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/76.tar",
        "*N": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/77.tar",
        "*NO3": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/78.tar",
        "*OHNH2": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/79.tar",
        "*ONOH": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/80.tar",
        "*CN": "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/81.tar"
    },
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
}

S2EF_COUNTS = {
    "s2ef": {
        "200k": 200000,
        "2M": 2000000,
        "20M": 20000000,
        "all": 133934018,
        "val_id": 999866,
        "val_ood_ads": 999838,
        "val_ood_cat": 999809,
        "val_ood_both": 999944,
        "rattled": 16677031,
        "md": 38315405,
    },
}


def get_data(datadir, task, split, adsorbate, del_intmd_files):
    os.makedirs(datadir, exist_ok=True)

    if task == "s2ef" and split is None:
        raise NotImplementedError("S2EF requires a split to be defined.")

    if task == "s2ef":
        assert (
            split in DOWNLOAD_LINKS[task]
        ), f'S2EF "{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS["s2ef"].keys())}'
        download_link = DOWNLOAD_LINKS[task][split]
    elif task == "is2re":
        download_link = DOWNLOAD_LINKS[task][split]
    elif task == "is2re_adsorbate":
        download_link = DOWNLOAD_LINKS[task][adsorbate]

    os.system(f"wget {download_link} -P {datadir}")
    filename = os.path.join(datadir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {datadir}")

    dirname = os.path.join(
        datadir,
        os.path.basename(filename).split(".")[0]
    )
    if task == "s2ef" and split != "test":
        compressed_dir = os.path.join(dirname, os.path.basename(dirname))
        if split in ["200k", "2M", "20M", "all", "rattled", "md"]:
            output_path = os.path.join(datadir, task, split, "train")
        else:
            output_path = os.path.join(datadir, task, "all", split)
        uncompressed_dir = uncompress_data(compressed_dir)
        preprocess_data(uncompressed_dir, output_path)

        verify_count(output_path, task, split)
    elif task == "s2ef" and split == "test":
        os.system(f"mv {dirname}/test_data/s2ef/all/test_* {datadir}/s2ef/all")
    elif task == "is2re":
        os.system(f"mv {dirname}/data/is2re {datadir}")
    elif task == "is2re_adsorbate":
        output_path = os.path.join(datadir, "is2re_adsorbate")
        os.makedirs(output_path, exist_ok=True)
        os.system(f"mv {dirname} {output_path}")


    if del_intmd_files:
        cleanup(filename, dirname)


def uncompress_data(compressed_dir):
    import uncompress

    parser = uncompress.get_parser()
    args, _ = parser.parse_known_args()
    args.ipdir = compressed_dir
    args.opdir = os.path.dirname(compressed_dir) + "_uncompressed"
    uncompress.main(args)
    return args.opdir


def preprocess_data(uncompressed_dir, output_path):
    import preprocess_ef as preprocess

    parser = preprocess.get_parser()
    args, _ = parser.parse_known_args()
    args.data_path = uncompressed_dir
    args.out_path = output_path
    preprocess.main(args)


def verify_count(output_path, task, split):
    paths = glob.glob(os.path.join(output_path, "*.txt"))
    count = 0
    for path in paths:
        lines = open(path, "r").read().splitlines()
        count += len(lines)
    assert (
        count == S2EF_COUNTS[task][split]
    ), f"S2EF {split} count incorrect, verify preprocessing has completed successfully."


def cleanup(filename, dirname):
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if os.path.exists(dirname + "_uncompressed"):
        shutil.rmtree(dirname + "_uncompressed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to download")
    parser.add_argument("--adsorbate", type=str, help="Adsorbate for when task is is2re_adsorbate")
    parser.add_argument(
        "--split", type=str, help="Corresponding data split to download"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate directories and files upon data retrieval/processing",
    )
    # Flags for S2EF train/val set preprocessing:
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(ocpmodels.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'ocpmodels/data'",
    )

    args, _ = parser.parse_known_args()
    get_data(
        datadir=args.data_path,
        task=args.task,
        split=args.split,
        adsorbate=args.adsorbate,
        del_intmd_files=not args.keep,
    )
