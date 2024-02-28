"""継続長モデルのための1発話に対する前処理"""

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import fire
import numpy as np
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts

from ttslearn.utils.io import read_qst


def preprocess(
    lab_path: Path,
    binary_dict: dict,
    numeric_dict: dict,
    save_dir_lng_feat: Path,
    save_dir_dur_feat: Path,
) -> None:
    """1つのフルコンテキストラベルファイルを受け取り、言語特徴量と音素継続長をnpyで保存

    Args:
        lab_path (Path): _description_
        binary_dict (dict): _description_
        numeric_dict (dict): _description_
        save_dir_lng_feat (Path): _description_
        save_dir_dur_feat (Path): _description_
    """
    labels = hts.load(lab_path)
    # 継続長モデルの入力：言語特徴量
    lng_feats = fe.linguistic_features(labels, binary_dict, numeric_dict).astype(
        np.float32
    )
    # 継続長モデルの出力：音素継続長
    dur_feats = fe.duration_features(labels).astype(np.float32)

    np.save(
        save_dir_lng_feat / f"{lab_path.stem}.npy",
        lng_feats,
        allow_pickle=False,
    )
    np.save(
        save_dir_dur_feat / f"{lab_path.stem}.npy",
        dur_feats,
        allow_pickle=False,
    )


def main(
    lab_root_dir: str,
    utt_list_path: str,
    qst_path: str,
    save_dir_root: str,
    cpu_counts: int = os.cpu_count(),
):
    """フルコンテキストラベルのファイル群から、言語特徴量および音素継続長を抽出する

    Args:
        lab_root_dir (str): _description_
        utt_list_path (str): _description_
        qst_path (str): _description_
        save_dir_root (str): _description_
        cpu_counts (int, optional): _description_. Defaults to os.cpu_count().
    """
    utt_list_path = Path(utt_list_path)
    qst_path = Path(qst_path)

    dataset_set = utt_list_path.stem
    save_dir_lng_feat = Path(save_dir_root) / dataset_set / "in_duration"
    save_dir_dur_feat = Path(save_dir_root) / dataset_set / "out_duration"

    save_dir_lng_feat.mkdir(exist_ok=False, parents=True)
    save_dir_dur_feat.mkdir(exist_ok=False, parents=True)

    with open(utt_list_path) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    lab_paths = [Path(lab_root_dir) / f"{utt_id}.lab" for utt_id in utt_ids]

    binary_dict, numeric_dict = read_qst(qst_path)

    partial_func = partial(
        preprocess,
        binary_dict=binary_dict,
        numeric_dict=numeric_dict,
        save_dir_lng_feat=save_dir_lng_feat,
        save_dir_dur_feat=save_dir_dur_feat,
    )
    p = Pool(processes=cpu_counts)
    p.map(partial_func, [lab_path for lab_path in lab_paths[:]])


if __name__ == "__main__":
    fire.Fire(main)
