import logging

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from torch import nn

from ttslearn.criterions.distance import L2Loss
from ttslearn.data.collate_fns import collate_fn_dnntts
from ttslearn.data.dataset import EvalDataLoader, TrainDataLoader, TTSDataset
from ttslearn.train.optimizer import get_optimizer
from ttslearn.train.trainer import Trainer
from ttslearn.utils.utils import init_seed


@hydra.main(version_base=None, config_path="../config/", config_name="sample")
def main(cfg: DictConfig):
    with mlflow.start_run():
        init_seed(cfg.seed)

        train_dataset = TTSDataset(datacfg=cfg.data.train)
        valid_dataset = TTSDataset(datacfg=cfg.data.dev)
        logging.info(f"Training dataset includes {len(train_dataset)} samples.")
        logging.info(f"Valid dataset includes {len(valid_dataset)} samples.")

        loader = {}
        loader["train"] = TrainDataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate_fn_dnntts,
            pin_memory=cfg.train.pin_memory,
            num_workers=cfg.train.num_workers,
            shuffle=True,
        )
        loader["valid"] = EvalDataLoader(
            valid_dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate_fn_dnntts,
            pin_memory=cfg.train.pin_memory,
            num_workers=cfg.train.num_workers,
            shuffle=False,
        )

        # モデルのインスタンス化
        model = hydra.utils.instantiate(cfg.model)
        logging.info(model)
        logging.info(f"# Parameters: {model.num_parameters}")

        if cfg.train.use_cuda:
            if torch.cuda.is_available():
                model.cuda()
                model = nn.DataParallel(model)
                logging.info("Use CUDA")
            else:
                raise ValueError("Cannot use CUDA.")
        else:
            logging.info("Does NOT use CUDA")

        # Optimizer
        optimizer = get_optimizer(model, cfg.train.optimizer)

        # Criterion
        if cfg.train.criterion == "l2loss":
            criterion = L2Loss()
        else:
            raise ValueError(f"Not support criterion {cfg.train.criterion}")

        trainer = Trainer(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            traincfg=cfg.train,
        )
        trainer.run()


if __name__ == "__main__":
    main()
