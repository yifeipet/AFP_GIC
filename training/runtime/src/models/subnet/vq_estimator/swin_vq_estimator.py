import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from torch import Tensor

from src.models.layer.swinir_layers import RSTB
from src.models.layer.femasr_layers import ResBlock

from src.utils.registry import VQ_ESTIMATOR_REGISTRY
    

@VQ_ESTIMATOR_REGISTRY.register()
class DualBlockSwinVqEstimator(nn.Module):
    """Swin-Transformer based VQ-Estimator
    from FeMaSR(ACMMM2022) https://github.com/chaofengc/FeMaSR/blob/main/basicsr/archs/femasr_arch.py
    """

    def __init__(
        self,
        input_resolution: tuple[int, int]=(32, 32),
        in_ch: int=192,
        main_ch: int=256,
        n_embed: int = 256,
        embed_dim: int=4,
        blk_depth: int=6,
        num_heads: int=8,
        window_size: int=8,
        num_swin_blocks: int=4,
        act_type: str='silu',
        norm_type: str='gn',
        use_upsample: bool=False,
        rstb_kwargs: dict={},
        proj_pos: str = 'before_rstb',
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.first_block = nn.Sequential(
            nn.Conv2d(in_ch, main_ch, kernel_size=3, padding=1, stride=1),
            nn.Upsample(scale_factor=2) if use_upsample else nn.Identity(),
            ResBlock(main_ch, main_ch, norm_type=norm_type, act_type=act_type),
            ResBlock(main_ch, main_ch, norm_type=norm_type, act_type=act_type),
            nn.Conv2d(main_ch, main_ch, 3, stride=1, padding=1),
        )
        self.embed_projection = nn.Conv2d(main_ch, embed_dim, 1, stride=1, padding=0)

        self.swin_blks = nn.ModuleList()
        for _ in range(num_swin_blocks):
            layer = RSTB(
                main_ch,
                input_resolution,
                blk_depth,
                num_heads,
                window_size,
                patch_size=1,
                **rstb_kwargs
            )
            self.swin_blks.append(layer)

        self.out_block = nn.Sequential(
                        ResBlock(main_ch, main_ch, norm_type=norm_type, act_type=act_type),
                        nn.Conv2d(main_ch, n_embed, 3, stride=1, padding=1),
                    )
        assert proj_pos in ['before_rstb', 'after_rstb']
        self.proj_pos = proj_pos

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.first_block(x)
        if self.proj_pos == 'before_rstb':
            pred_embed = self.embed_projection(x)

        b, c, h, w = x.shape

        need_padding = (h % self.window_size != 0) or (w % self.window_size != 0)
        if need_padding:
            assert not self.training
            pad_h = math.ceil(h / self.window_size) * self.window_size - h
            pad_w = math.ceil(w / self.window_size) * self.window_size - w
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = x.shape[2:]

        x = x.reshape(b, c, h * w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)

        if need_padding:
            x = x[:, :, :h - pad_h, :w - pad_w]

        if self.proj_pos == 'after_rstb':
            pred_embed = self.embed_projection(x)

        logits = self.out_block(x)

        return pred_embed, logits
