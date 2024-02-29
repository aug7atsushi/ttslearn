from abc import ABC, abstractmethod

import mlflow
from torch import nn


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        """順伝搬の関数"""
        return NotImplemented

    @property
    def num_parameters(self):
        _num_parameters = 0

        for p in self.parameters():
            if p.requires_grad:
                _num_parameters += p.numel()

        return _num_parameters

    def build_model(self, model_path) -> None:
        config = mlflow.pytorch.load_state_dict(model_path)
        self.load_state_dict(config["state_dict"])
