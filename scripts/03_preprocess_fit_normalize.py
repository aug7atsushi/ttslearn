"""継続長モデル、音響モデルの入出力となる特徴量の正規化を行う"""

from pathlib import Path

import fire
from _preprocess_fit import main as fit
from _preprocess_normalize import main as normalize


def main(org_dir_root, norm_dir_root, cpu_counts):
    for model in ["duration", "acoustic"]:
        for in_out in ["in", "out"]:
            read_dir = (
                Path(org_dir_root) / "train" / f"{in_out}_{model}"
            )  # 学習データの平均と標準偏差を用いて正規化
            write_path = Path(org_dir_root) / f"{in_out}_{model}_scaler.joblib"
            print(read_dir)
            print(write_path)
            fit(read_dir=read_dir, write_path=write_path)

    for dataset_set in ["train", "eval", "dev"]:
        for model in ["duration", "acoustic"]:
            for in_out in ["in", "out"]:
                read_dir = Path(org_dir_root) / f"{dataset_set}" / f"{in_out}_{model}"

                write_dir = Path(norm_dir_root) / f"{dataset_set}" / f"{in_out}_{model}"

                scaler_path = Path(org_dir_root) / f"{in_out}_{model}_scaler.joblib"

                print(read_dir)
                print(write_dir)
                print(scaler_path)

                normalize(
                    read_dir=read_dir,
                    write_dir=write_dir,
                    scaler_path=scaler_path,
                    cpu_counts=cpu_counts,
                )


if __name__ == "__main__":
    fire.Fire(main)
