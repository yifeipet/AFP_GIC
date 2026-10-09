"""
S.Zhou et al. "Towards Robust Blind Face Restoration with Codebook Lookup Transformer" (NeurIPS 2022)

From official implementation https://github.com/sczhou/CodeFormer/blob/master/basicsr/archs/codeformer_arch.py
"""

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class FuseSftBlock(nn.Module):
    def __init__(self, cond_ch: int, dec_ch: int, mid_ch: int):
        super().__init__()
        self.fuse_block = ResBlock(cond_ch+dec_ch, mid_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(mid_ch, dec_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(mid_ch, dec_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1))

    def forward(self, dec_feat: Tensor, cond_feat: Tensor, w: float=1.) -> Tensor:
        fuse_feat = self.fuse_block(torch.cat([cond_feat, dec_feat], dim=1))
        scale = self.scale(fuse_feat)
        shift = self.shift(fuse_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out