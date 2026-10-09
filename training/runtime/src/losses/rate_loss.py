from typing import Optional, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class RateLoss(nn.Module):
    def __init__(self, loss_weight: float, target_rate: float=0.0, reduction: str='mean'):
        super().__init__()
        self.lamb_rate = loss_weight
        self.target_rate = target_rate
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction
    
    def forward(self, bpp, **kwargs):
        if self.reduction == 'mean':
            bpp = torch.mean(bpp)
        elif self.reduction == 'sum':
            bpp = torch.sum(bpp)
        return self.lamb_rate * bpp
