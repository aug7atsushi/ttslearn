import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import torch
from torch import nn

from ttslearn.utils.paddding import make_non_pad_mask


class ABCTrainer(ABC):
    @abstractmethod
    def run(self):
        return NotImplemented

    @abstractmethod
    def run_one_epoch(self):
        return NotImplemented

    @abstractmethod
    def run_one_epoch_train(self):
        return NotImplemented

    @abstractmethod
    def run_one_epoch_eval(self):
        return NotImplemented


class TrainerBase(ABCTrainer):
    def __init__(self, model, loader, criterion, optimizer, traincfg) -> None:
        self.model = model
        self.train_loader, self.valid_loader = loader["train"], loader["valid"]
        self.criterion = criterion
        self.optimizer = optimizer
        self.traincfg = traincfg
        self._set()

    def _set(self):
        self.epochs = self.traincfg.epochs
        self.use_cuda = self.traincfg.use_cuda
        self.overwrite = self.traincfg.overwrite
        self.save_dir = Path(self.traincfg.save_dir)
        self.continue_from = self.traincfg.continue_from

        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.train_loss = torch.empty(self.epochs)
        self.valid_loss = torch.empty(self.epochs)

        if self.continue_from:
            config = torch.load(
                self.continue_from, map_location=lambda storage, loc: storage
            )

            self.start_epoch = config["epoch"]

            self.train_loss[: self.start_epoch] = config["train_loss"][
                : self.start_epoch
            ]
            self.valid_loss[: self.start_epoch] = config["valid_loss"][
                : self.start_epoch
            ]
            self.best_loss = config["best_loss"]

            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(config["state_dict"])
            else:
                self.model.load_state_dict(config["state_dict"])

            self.optimizer.load_state_dict(config["optim_dict"])
        else:
            if (self.save_dir / "best.pth").exists():
                if self.overwrite:
                    print("Overwrite models.")
                else:
                    raise ValueError(
                        "Model already exists. If continue, set overwrite."
                    )

            self.start_epoch = 0
            self.best_loss = float("infinity")

    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            self.logging(
                train_loss=train_loss,
                valid_loss=valid_loss,
                epoch=epoch,
                exec_time=end - start,
            )

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.save_model(epoch, filename="best.pth")
            self.save_model(epoch, filename="last.pth")

    def run_one_epoch(self, epoch: int):
        train_loss = self.run_one_epoch_train(epoch)
        valid_loss = self.run_one_epoch_eval(epoch)
        return train_loss, valid_loss

    def run_one_epoch_train(self, epoch: int) -> float:
        self.model.train()
        train_loss = 0
        n_train_batch = len(self.train_loader)

        for idx, (in_feats, out_feats, lengths) in enumerate(self.train_loader):
            # NOTE: PackedSequence の仕様に合わせるため、系列長の降順にソート
            lengths, indices = torch.sort(lengths, dim=0, descending=True)

            if self.use_cuda:
                in_feats = in_feats[indices].cuda()
                out_feats = out_feats[indices].cuda()

            pred_feats = self.model(in_feats)

            # ゼロパディングされた部分を損失の計算に含めないように、マスクを作成
            mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
            pred_feats = pred_feats.masked_select(mask)
            out_feats = out_feats.masked_select(mask)

            loss = self.criterion(pred_feats, out_feats)

            self.optimizer.zero_grad()
            loss.backward()  # 自動微分
            self.optimizer.step()  # 重み更新
            train_loss += loss.item()

            if (idx + 1) % 100 == 0:
                logging.info(
                    "[Epoch {}/{}] iter {}/{} train loss: {:.5f}".format(
                        epoch + 1,
                        self.epochs,
                        idx + 1,
                        n_train_batch,
                        loss.item(),
                    )
                )

        train_loss /= n_train_batch
        return train_loss

    def run_one_epoch_eval(self, epoch: int) -> float:
        self.model.eval()
        valid_loss = 0
        n_eval_batch = len(self.valid_loader)

        with torch.no_grad():
            for in_feats, out_feats, lengths in self.valid_loader:
                # NOTE: PackedSequence の仕様に合わせるため、系列長の降順にソート
                lengths, indices = torch.sort(lengths, dim=0, descending=True)

                if self.use_cuda:
                    in_feats = in_feats[indices].cuda()
                    out_feats = out_feats[indices].cuda()

                pred_feats = self.model(in_feats)

                # ゼロパディングされた部分を損失の計算に含めないように、マスクを作成
                mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
                pred_feats = pred_feats.masked_select(mask)
                out_feats = out_feats.masked_select(mask)

                loss = self.criterion(pred_feats, out_feats)
                valid_loss += loss.item()

        valid_loss /= n_eval_batch
        return valid_loss

    def save_model(self, epoch: int, filename: str = "tmp.pth"):
        config = {}
        if isinstance(self.model, nn.DataParallel):
            config["state_dict"] = self.model.module.state_dict()
        else:
            config["state_dict"] = self.model.state_dict()

        config["optim_dict"] = self.optimizer.state_dict()
        config["best_loss"] = self.best_loss
        config["train_loss"] = self.train_loss
        config["valid_loss"] = self.valid_loss
        config["epoch"] = epoch + 1

        torch.save(config, self.save_dir / filename)
        mlflow.pytorch.log_state_dict(config, "model")

    def logging(
        self,
        train_loss: float,
        valid_loss: float,
        epoch: int,
        exec_time: float,
    ):
        self.train_loss[epoch] = train_loss
        self.valid_loss[epoch] = valid_loss

        if mlflow.active_run():
            mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
            mlflow.log_metric(key="valid_loss", value=valid_loss, step=epoch)
            mlflow.log_metric(key="best_loss", value=self.best_loss, step=epoch)
            mlflow.log_metric(key="exec_time", value=exec_time, step=epoch)

        logging.info(
            """[Epoch {}/{}] loss (train): {:.5f}, \
            loss (valid): {:.5f}, {:.3f} [sec]""".format(
                epoch + 1,
                self.epochs,
                train_loss,
                valid_loss,
                exec_time,
            )
        )
