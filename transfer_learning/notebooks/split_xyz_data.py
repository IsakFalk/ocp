from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

DATA_PATH = Path("data/luigi/")
trajectories = list(DATA_PATH.glob("*.xyz"))


def create_split(data, train_split=0.6, test_split=0.2, walk_forward=False):
    val_split = 1 - train_split - test_split
    n = len(data)
    index = np.arange(len(data))
    if not walk_forward:
        np.random.shuffle(index)
    train_index = index[: int(train_split * n)]
    val_index = index[int(train_split * n) : int((val_split + train_split) * n)]
    test_index = index[int((val_split + train_split) * n) :]
    data_train = [data[i] for i in train_index]
    data_val = [data[i] for i in val_index]
    data_test = [data[i] for i in test_index]
    return data_train, data_val, data_test


def dump_splits(data_train, data_val, data_test, original_path, walk_forward=False):
    if walk_forward:
        original_path = original_path.parent / f"{original_path.stem}_walk_forward"
    write(str((original_path.parent / "splits" / f"{original_path.stem}_train.xyz").resolve()), data_train)
    write(str((original_path.parent / "splits" / f"{original_path.stem}_val.xyz").resolve()), data_val)
    write(str((original_path.parent / "splits" / f"{original_path.stem}_test.xyz").resolve()), data_test)


for walk_forward in [True, False]:
    for traj in trajectories:
        data = read(traj, index=":")
        data_train, data_val, data_test = create_split(data, train_split=0.6, test_split=0.2, walk_forward=walk_forward)
        dump_splits(data_train, data_val, data_test, traj, walk_forward=walk_forward)
