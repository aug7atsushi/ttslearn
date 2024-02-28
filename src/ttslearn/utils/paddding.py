from typing import List

import numpy as np
import torch


def pad_2d(
    x: np.ndarray,
    max_len: int,
    constant_values: int = 0,
) -> np.ndarray:
    """与えられた2Dの特徴量の末尾にパディングを行う"""
    x = np.pad(
        x,
        [(0, max_len - len(x)), (0, 0)],
        mode="constant",
        constant_values=constant_values,
    )
    return x


def make_pad_mask(lengths: List[int], maxlen: int = None):
    """Make mask for padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length.

    Returns:
        torch.ByteTensor: mask
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask


def make_non_pad_mask(lengths: List[int], maxlen: int = None):
    """Make mask for non-padding frames

    Args:
        lengths (list): list of lengths
        maxlen (int, optional): maximum length.

    Returns:
        torch.ByteTensor: mask
    """
    return ~make_pad_mask(lengths, maxlen)
