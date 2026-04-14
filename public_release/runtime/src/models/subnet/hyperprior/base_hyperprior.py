import torch
import torch.nn as nn


class BaseHyperEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseHyperDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError