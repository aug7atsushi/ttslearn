import numpy as np
import pysptk
import pyworld
from nnmnkwii.preprocessing import delta_features
from nnmnkwii.preprocessing.f0 import interp1d


def f0_to_lf0(f0):
    """Convert F0 to log-F0

    Args:
        f0 (ndarray): F0 in Hz.

    Returns:
        ndarray: log-F0.
    """
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(f0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    return lf0


def lf0_to_f0(lf0, vuv):
    """Convert log-F0 (and V/UV) to F0

    Args:
        lf0 (ndarray): F0 in Hz.
        vuv (ndarray): V/UV.

    Returns:
        ndarray: F0 in Hz.
    """
    f0 = np.exp(lf0)
    f0[vuv < 0.5] = 0
    return f0


def world_spss_params(x, sr, mgc_order=None) -> np.ndarray:
    """WORLD-based acoustic feature extraction

    Args:
        x (ndarray): Waveform.
        sr (int): Sampling rate.
        mgc_order (int, optional): MGC order. Defaults to None.

    Returns:
        np.ndarray: WORLD features.
    """
    f0, timeaxis = pyworld.dio(x, sr)
    # (Optional) Stonemask によってF0の推定結果をrefineする
    f0 = pyworld.stonemask(x, f0, timeaxis, sr)

    sp = pyworld.cheaptrick(x, f0, timeaxis, sr)
    ap = pyworld.d4c(x, f0, timeaxis, sr)

    alpha = pysptk.util.mcepalpha(sr)
    # メルケプストラムの次元数（※過去の論文にならい、16kHzの際に
    # 次元数が40（mgc_order + 1）になるように設定する
    # ただし、上限を 60 (59 + 1) とします
    # [Zen 2013] Statistical parametric speech synthesis
    # using deep neural networks
    if mgc_order is None:
        mgc_order = min(int(sr / 16000.0 * 40) - 1, 59)
    mgc = pysptk.sp2mc(sp, mgc_order, alpha)

    # 有声/無声フラグ
    vuv = (f0 > 0).astype(np.float32)

    # 連続対数F0
    lf0 = f0_to_lf0(f0)
    lf0 = interp1d(lf0)
    # 帯域非周期性指標
    bap = pyworld.code_aperiodicity(ap, sr)

    # F0とvuvを二次元の行列の形にしておく
    lf0 = lf0[:, np.newaxis] if len(lf0.shape) == 1 else lf0
    vuv = vuv[:, np.newaxis] if len(vuv.shape) == 1 else vuv

    # 動的特徴量の計算
    windows = [
        [1.0],  # 静的特徴量に対する窓
        [-0.5, 0.0, 0.5],  # 1次動的特徴量に対する窓
        [1.0, -2.0, 1.0],  # 2次動的特徴量に対する窓
    ]
    mgc = delta_features(mgc, windows)
    lf0 = delta_features(lf0, windows)
    bap = delta_features(bap, windows)

    feats = np.hstack([mgc, lf0, vuv, bap]).astype(np.float32)
    return feats
