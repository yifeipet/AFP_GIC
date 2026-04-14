
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

class LightFuseSftBlock(nn.Module):
    def __init__(self, cond_ch: int, dec_ch: int, mid_ch: int):
        super().__init__()
        self.fuse_layer = nn.Sequential(
            nn.Conv2d(cond_ch+dec_ch, mid_ch, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.scale = nn.Conv2d(mid_ch, dec_ch, kernel_size=3, padding=1)
        self.shift = nn.Conv2d(mid_ch, dec_ch, kernel_size=3, padding=1)

    def forward(self, dec_feat: Tensor, cond_feat: Tensor, w: float=1.) -> Tensor:
        fuse_feat = self.fuse_layer(torch.cat([cond_feat, dec_feat], dim=1))
        scale = self.scale(fuse_feat)
        shift = self.shift(fuse_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out