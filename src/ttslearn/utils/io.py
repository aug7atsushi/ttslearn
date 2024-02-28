from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
from nnmnkwii.io import hts
from scipy.io import wavfile


def read_wav(wav_path: str | Path) -> Tuple[np.ndarray, int]:
    fs, signal = wavfile.read(wav_path)
    # TODO: wavのbit数は16bitを仮定している
    signal = (signal / 32768).astype(np.float64)
    return signal, fs


def write_wav(wav_path: str | Path, fs: int, signal: np.ndarray) -> None:
    signal = signal * 32768
    # TODO: wavのbit数は16bitを仮定している
    signal = np.clip(signal, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
        np.int16
    )
    wavfile.write(wav_path, fs, signal)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)
    return signal


def read_qst(qst_path: str | Path) -> Union[dict, dict]:
    binary_dict, numeric_dict = hts.load_question_set(qst_path)
    return binary_dict, numeric_dict
