""""""

from pathlib import Path

import fire
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def main(
    read_dir: str,
    write_path: str,
):
    read_dir = Path(read_dir)
    read_paths = list(read_dir.glob("*"))
    print(read_dir)
    print(f"File nums: {len(read_paths)}")

    scaler = StandardScaler()
    for read_path in tqdm(read_paths):
        c = np.load(read_path)
        scaler.partial_fit(c)
    joblib.dump(scaler, write_path)


if __name__ == "__main__":
    fire.Fire(main)
