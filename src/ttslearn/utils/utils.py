import random

import numpy as np
import torch


def init_seed(seed: int = 42) -> None:
    """Initialize random seed.

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_utt_list(utt_list_path):
    """Load a list of utterances.

    Args:
        utt_list (str): path to a file containing a list of utterances

    Returns:
        List[str]: list of utterances
    """
    utt_ids = []
    with open(utt_list_path) as f:
        for utt_id in f:
            utt_id = utt_id.strip()
            if len(utt_id) > 0:
                utt_ids.append(utt_id)
    return utt_ids
