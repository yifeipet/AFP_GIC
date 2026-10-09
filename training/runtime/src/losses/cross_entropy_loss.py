from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight: float, ce_kwargs: dict={}) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """_summary_

        Args:
            input (Tensor): [B, C, H, W] FloatTensor (logit)
            target (Tensor): [B, H, W] LongTensor (index)

        Returns:
            Tensor: _description_
        """
        loss_val = self.ce(input, target)
        return self.loss_weight * loss_val
    

@LOSS_REGISTRY.register()
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight: float, gamma: float, reduction: str='mean', **kwargs) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(**kwargs, reduction='none')
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce(input, target) # [B, H, W]
        softmax_probs = F.softmax(input, dim=1) # [B, C, H, W]
        pt = softmax_probs.gather(1, target.unsqueeze(1)).squeeze(1) # [B, 1, H, W]
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            loss_val = focal_loss.mean()
        elif self.reduction == 'sum':
            loss_val = focal_loss.sum()
        else:
            loss_val = focal_loss
        return self.loss_weight * loss_val