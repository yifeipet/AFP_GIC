from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.layer.femasr_layers import ResBlock


class AdaCodeBranchPriorEstimator(nn.Module):
    def __init__(
        self,
        in_ch: int = 192,
        out_ch: int = 128,
        num_branches: int = 4,
        hidden_ch: int = 256,
        num_res_blocks: int = 4,
        act_type: str = "silu",
        norm_type: str = "gn",
        weight_norm_type: str = "softmax",
        use_output_skip: bool = True,
    ) -> None:
        super().__init__()
        if weight_norm_type not in {"softmax", "none"}:
            raise ValueError(f"Unsupported weight_norm_type: {weight_norm_type}")

        self.out_ch = out_ch
        self.num_branches = num_branches
        self.weight_norm_type = weight_norm_type

        self.input_proj = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1)
        trunk_blocks = []
        for _ in range(num_res_blocks):
            trunk_blocks.append(
                ResBlock(
                    hidden_ch,
                    hidden_ch,
                    norm_type=norm_type,
                    act_type=act_type,
                )
            )
        self.trunk = nn.Sequential(*trunk_blocks)
        self.branch_head = nn.Conv2d(
            hidden_ch,
            num_branches * out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.weight_head = nn.Conv2d(
            hidden_ch,
            num_branches,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.branch_skip = None
        if use_output_skip:
            self.branch_skip = nn.Conv2d(
                hidden_ch,
                num_branches * out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def normalize_weight_map(self, weight_map: Tensor) -> Tensor:
        if weight_map.ndim == 5:
            weight_map = weight_map.squeeze(2)
        if weight_map.ndim != 4:
            raise ValueError(f"Expected weight_map to be 4D/5D, got {tuple(weight_map.shape)}")
        if self.weight_norm_type == "softmax":
            return torch.softmax(weight_map, dim=1)
        return weight_map

    def resize_branch_feats(
        self,
        branch_feats: Tensor,
        target_hw: Tuple[int, int],
    ) -> Tensor:
        if branch_feats.shape[-2:] == target_hw:
            return branch_feats
        b, k, c, _, _ = branch_feats.shape
        resized = F.interpolate(
            branch_feats.reshape(b * k, c, *branch_feats.shape[-2:]),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(b, k, c, *target_hw)

    def resize_weight_map(
        self,
        weight_map: Tensor,
        target_hw: Tuple[int, int],
    ) -> Tensor:
        if weight_map.ndim == 5:
            weight_map = weight_map.squeeze(2)
        if weight_map.shape[-2:] == target_hw:
            return weight_map
        return F.interpolate(
            weight_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )

    def fuse_branch_feats(
        self,
        branch_feats: Tensor,
        weight_map: Tensor,
    ) -> Tensor:
        if weight_map.ndim == 4:
            weight_map = weight_map.unsqueeze(2)
        return torch.sum(branch_feats * weight_map, dim=1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feat = self.input_proj(x)
        trunk_out = self.trunk(feat)

        pred_branch_feats = self.branch_head(trunk_out)
        if self.branch_skip is not None:
            pred_branch_feats = pred_branch_feats + self.branch_skip(feat)

        b, _, h, w = pred_branch_feats.shape
        pred_branch_feats = pred_branch_feats.reshape(
            b,
            self.num_branches,
            self.out_ch,
            h,
            w,
        )
        pred_weight_logits = self.weight_head(trunk_out)
        pred_weight_map = self.normalize_weight_map(pred_weight_logits)
        pred_prior_fused = self.fuse_branch_feats(pred_branch_feats, pred_weight_map)

        return {
            "pred_branch_feats": pred_branch_feats,
            "pred_weight_logits": pred_weight_logits,
            "pred_weight_map": pred_weight_map,
            "pred_prior_fused": pred_prior_fused,
        }
