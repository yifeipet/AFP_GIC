from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class VanillaGANLoss(nn.Module):
    def __init__(self, loss_weight: float, real_label: float=1.0, fake_label: float=0.0, loss_reduction: str='mean'):
        super().__init__()
        self.lamb_gan = loss_weight
        self.loss = nn.BCEWithLogitsLoss(reduction=loss_reduction)
        self.real_label_val = real_label
        self.fake_label_val = fake_label

    def get_label_like_x(self, x: torch.Tensor, is_real: bool):
        if is_real:
            if not(hasattr(self, 'real_label')):
                self.real_label = torch.tensor([self.real_label_val], device=x.device, dtype=x.dtype)
            return self.real_label.expand_as(x)
        
        if not(hasattr(self, 'fake_label')):
            self.fake_label = torch.tensor([self.fake_label_val], device=x.device, dtype=x.dtype)
        return self.fake_label.expand_as(x)

    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        label = self.get_label_like_x(x, is_real)
        loss = self.loss(x, label)
        return loss if is_disc else self.lamb_gan * loss


@LOSS_REGISTRY.register()
class HingeGANLoss(nn.Module):
    def __init__(self, loss_weight: float) -> None:
        super().__init__()
        self.lamb_gan = loss_weight
        self.relu = nn.ReLU()

    def forward(self, x, is_real: bool, is_disc: bool=False, **kwargs):
        ## For D
        if is_disc:
            if is_real:
                loss = self.relu(1 - x)
            else: ## fake
                loss = self.relu(1 + x)
            return torch.mean(loss)

        ## For G
        assert is_real, 'For G loss `is_real` should be True'
        loss = -torch.mean(x)
        return self.lamb_gan * loss
