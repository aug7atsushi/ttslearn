""""""

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import fire
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def transform(
    read_path: Path,
    scaler: StandardScaler,
    write_dir: Path,
    inverse: bool = False,
):
    x = np.load(read_path)
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)
    assert x.dtype == y.dtype
    np.save(write_dir / read_path.name, y, allow_pickle=False)


def main(
    read_dir: str,
    write_dir: str,
    scaler_path: str,
    cpu_counts: int = os.cpu_count(),
):
    read_dir = Path(read_dir)
    scaler_path = Path(scaler_path)
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    read_paths = list(read_dir.glob("*"))
    scaler = joblib.load(scaler_path)

    print(f"File nums: {len(read_paths)}")

    partial_func = partial(transform, scaler=scaler, write_dir=write_dir, inverse=False)
    p = Pool(processes=cpu_counts)
    p.map(partial_func, [read_path for read_path in read_paths[:]])


if __name__ == "__main__":
    fire.Fire(main)
