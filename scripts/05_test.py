import logging

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from ttslearn.criterions.distance import L2Loss
from ttslearn.data.collate_fns import collate_fn_dnntts
from ttslearn.data.dataset import TestDataLoader, TTSDataset
from ttslearn.train.tester import Tester
from ttslearn.utils.utils import init_seed


@hydra.main(version_base=None, config_path="../config/", config_name="sample")
def main(cfg: DictConfig):
    init_seed(cfg.seed)

    test_dataset = TTSDataset(datacfg=cfg.data.test)
    logging.info(f"Test dataset includes {len(test_dataset)} samples.")

    loader = TestDataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        collate_fn=collate_fn_dnntts,
        shuffle=False,
    )

    # モデルのインスタンス化
    model = hydra.utils.instantiate(cfg.model)
    logging.info(model)
    logging.info(f"# Parameters: {model.num_parameters}")

    # TODO: モデルパラメータのロード

    if cfg.train.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            logging.info("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        logging.info("Does NOT use CUDA")

    # Criterion
    if cfg.test.criterion == "l2loss":
        criterion = L2Loss()
    else:
        raise ValueError(f"Not support criterion {cfg.test.criterion}")

    tester = Tester(
        model=model,
        loader=loader,
        criterion=criterion,
        testcfg=cfg.test,
    )
    tester.run()


if __name__ == "__main__":
    main()
