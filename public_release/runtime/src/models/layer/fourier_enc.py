from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class FourierEncoding(object):
    def __init__(self, L: int, max_beta: float, use_pi: bool = True, include_x: bool=False) -> None:
        assert L > 0
        assert max_beta > 0
        self.L = L
        self.max_beta = max_beta
        self.freq = torch.pow(
            torch.Tensor([2]), torch.arange(L)
        ).unsqueeze(0)  # [2^0, 2^1, 2^2, ..., 2^(L-1)]
        if use_pi:
            self.freq = self.freq * np.pi
        self.include_x = include_x

    def embed(self, beta: Union[int, float, Tensor]) -> Tensor:
        if isinstance(beta, (int, float)):
            beta = torch.Tensor([beta]).float()
        assert isinstance(beta, Tensor)
        assert beta.ndim == 1
        assert 0 <= beta.min() <= self.max_beta
        assert 0 <= beta.max() <= self.max_beta

        beta = beta.cpu()
        norm_beta = beta / self.max_beta  # [0, 1]
        norm_beta = (norm_beta - 0.5) * 2  # [-1, 1]
        norm_beta = norm_beta.unsqueeze(1) # [B, 1]

        sin_tensor = torch.sin(norm_beta * self.freq)
        cos_tensor = torch.cos(norm_beta * self.freq)
        out = torch.cat([sin_tensor, cos_tensor], dim=-1) # [B, 2L]
        if self.include_x:
            out = torch.cat([norm_beta, out], dim=-1) # [B, 2L+1]
        return out.detach() # [B, 2L]