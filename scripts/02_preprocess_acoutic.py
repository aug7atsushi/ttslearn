"""音響モデルのための 1 発話に対する前処理"""

import os
from multiprocessing import Pool
from pathlib import Path

import fire
import numpy as np
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts

from ttslearn.utils.dsp import world_spss_params
from ttslearn.utils.io import read_qst, read_wav, resample

FRAME_SHIFT_MS = 0.005


def preprocess(
    wav_path: Path,
    lab_path: Path,
    binary_dict: dict,
    numeric_dict: dict,
    save_dir_lng_feat: Path,
    save_dir_acoustic_feat: Path,
    fs: int = 16000,
):
    """wavファイルとフルコンテキストラベルファイルを受け取り、言語特徴量(input)と音響特徴量(output)をnpyで保存

    Args:
        wav_path (Path): _description_
        lab_path (Path): _description_
        binary_dict (dict): _description_
        numeric_dict (dict): _description_
        save_dir_lng_feat (Path): _description_
        save_dir_acoustic_feat (Path): _description_
        fs (int, optional): _description_. Defaults to 16000.
    """
    assert wav_path.stem == lab_path.stem

    # 言語特徴量の計算
    labels = hts.load(lab_path)
    in_feats = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features="coarse_coding",
    )

    # 音響特徴量の計算
    x, _fs = read_wav(wav_path)
    x = resample(x, _fs, fs)
    # workaround for over resampling: add a small white noise
    if fs > _fs:
        x = x + np.random.randn(len(x)) * (1 / 2**15)
    out_feats = world_spss_params(x, fs)

    # フレーム数の調整
    min_frames = min(in_feats.shape[0], out_feats.shape[0])
    in_feats, out_feats = in_feats[:min_frames], out_feats[:min_frames]

    # 冒頭と末尾の非音声区間の長さを調整
    assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
    start_frame = int((labels.start_times[1] * 1e-7) / FRAME_SHIFT_MS)
    end_frame = int((labels.end_times[-2] * 1e-7) / FRAME_SHIFT_MS)

    # 冒頭：50 ミリ秒、末尾：100 ミリ秒
    start_frame = max(0, start_frame - int(0.050 / FRAME_SHIFT_MS))
    end_frame = min(min_frames, end_frame + int(0.100 / FRAME_SHIFT_MS))

    in_feats = in_feats[start_frame:end_frame]
    out_feats = out_feats[start_frame:end_frame]

    # print("入力特徴量のサイズ:", in_feats.shape)
    # print("出力特徴量のサイズ:", out_feats.shape)

    # NumPy 形式でファイルに保存
    utt_id = lab_path.stem
    np.save(
        save_dir_lng_feat / f"{utt_id}-feats.npy",
        in_feats.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        save_dir_acoustic_feat / f"{utt_id}-feats.npy",
        out_feats.astype(np.float32),
        allow_pickle=False,
    )


def preprocess_wrapper(args):
    return preprocess(*args)


def main(
    wav_root_dir: str,
    lab_root_dir: str,
    utt_list_path: str,
    qst_path: str,
    save_dir_root: str,
    fs: int = 16000,
    cpu_counts: int = os.cpu_count(),
):
    """wavファイルとフルコンテキストラベルのファイル群から、言語特徴量および音響特徴量を抽出する

    Args:
        wav_root_dir (str): _description_
        lab_root_dir (str): _description_
        utt_list_path (str): _description_
        qst_path (str): _description_
        save_dir_root (str): _description_
        fs (int, optional): _description_. Defaults to 16000.
        cpu_counts (int, optional): _description_. Defaults to os.cpu_count().
    """
    utt_list_path = Path(utt_list_path)
    qst_path = Path(qst_path)

    dataset_set = utt_list_path.stem
    save_dir_lng_feat = Path(save_dir_root) / dataset_set / "in_acoustic"
    save_dir_acoustic_feat = Path(save_dir_root) / dataset_set / "out_acoustic"

    save_dir_lng_feat.mkdir(exist_ok=True, parents=True)
    save_dir_acoustic_feat.mkdir(exist_ok=True, parents=True)

    with open(utt_list_path) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    wav_paths = [Path(wav_root_dir) / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_paths = [Path(lab_root_dir) / f"{utt_id}.lab" for utt_id in utt_ids]

    # wav_paths = wav_paths[:1]
    # lab_paths = lab_paths[:1]

    binary_dict, numeric_dict = read_qst(qst_path)

    args = (
        (
            wav_path,
            lab_path,
            binary_dict,
            numeric_dict,
            save_dir_lng_feat,
            save_dir_acoustic_feat,
            fs,
        )
        for wav_path, lab_path in zip(wav_paths, lab_paths)
    )

    p = Pool(processes=cpu_counts)
    p.map(preprocess_wrapper, args)


if __name__ == "__main__":
    fire.Fire(main)
