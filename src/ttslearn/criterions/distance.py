import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (batch_size, *)
            target: (batch_size, *)
        Returns:
            output: () or (batch_size, )
        """
        loss = torch.abs(input - target)
        loss = torch.sum(loss**2)
        loss = torch.sqrt(loss)
        return loss
