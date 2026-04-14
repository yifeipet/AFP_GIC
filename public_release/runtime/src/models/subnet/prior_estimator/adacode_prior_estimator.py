from copy import deepcopy
from typing import Dict

import torch.nn as nn
from torch import Tensor

from src.models.layer.femasr_layers import ResBlock


class AdaCodePriorEstimator(nn.Module):
    def __init__(
        self,
        in_ch: int = 192,
        out_ch: int = 128,
        hidden_ch: int = 256,
        num_res_blocks: int = 4,
        act_type: str = "silu",
        norm_type: str = "gn",
        use_output_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1)
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResBlock(
                    hidden_ch,
                    hidden_ch,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )
        self.trunk = nn.Sequential(*blocks)
        self.output_proj = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.use_output_skip = use_output_skip
        if self.use_output_skip:
            # Project the input-projected feature to output channels so the
            # estimator keeps a stable residual path even when hidden_ch != out_ch.
            self.skip_proj = (
                nn.Identity()
                if hidden_ch == out_ch
                else nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.skip_proj = None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.input_proj(x)
        out = self.trunk(feat)
        out = self.output_proj(out)
        if self.skip_proj is not None:
            out = out + self.skip_proj(feat)
        return out


def build_adacode_prior_estimator(opt: Dict) -> AdaCodePriorEstimator:
    subnet_opt = deepcopy(opt)
    subnet_opt.pop("type", None)
    return AdaCodePriorEstimator(**subnet_opt)
