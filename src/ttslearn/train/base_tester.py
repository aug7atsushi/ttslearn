from abc import ABC, abstractmethod

import torch
from torch import nn

from ttslearn.utils.paddding import make_non_pad_mask


class ABCTester(ABC):
    @abstractmethod
    def run(self):
        return NotImplemented


class TesterBase(ABCTester):
    def __init__(self, model, loader, criterion, testcfg):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.testcfg = self._set(testcfg)

    def _set(self):
        self.use_cuda = self.testcfg.use_cuda
        self.save_dir = self.testcfg.save_dir
        self.model_path = self.testcfg.model_path
        self.save_dir.mkdir(exist_ok=True, parents=True)

        config = torch.load(self.model_path, map_location=lambda storage, loc: storage)

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(config["state_dict"])
        else:
            self.model.load_state_dict(config["state_dict"])

    def run(self):
        self.model.eval()

        test_loss = 0
        n_test = len(self.loader.dataset)

        print("Loss", flush=True)

        with torch.no_grad():
            for idx, (in_feats, out_feats, lengths) in enumerate(self.loader):
                # NOTE: PackedSequence の仕様に合わせるため、系列長の降順にソート
                lengths, indices = torch.sort(lengths, dim=0, descending=True)

                if self.use_cuda:
                    in_feats = in_feats[indices].cuda()
                    out_feats = out_feats[indices].cuda()

                pred_feats = self.model(in_feats, lengths)

                # ゼロパディングされた部分を損失の計算に含めないように、マスクを作成
                mask = make_non_pad_mask(lengths).unsqueeze(-1).to(in_feats.device)
                pred_feats = pred_feats.masked_select(mask)
                out_feats = out_feats.masked_select(mask)

                loss = self.criterion(pred_feats, out_feats).item()

                print("{:.3f},".format(loss), flush=True)
                test_loss += loss
        test_loss /= n_test
        print("Loss: {:.3f}".format(test_loss))
