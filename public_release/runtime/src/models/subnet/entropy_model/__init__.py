from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from compressai.entropy_models import EntropyModel
from torch import Tensor

from src.utils.registry import ENTROPYMODEL_REGISTRY

from .entropy_bottleneck import *
from .gaussian_conditional import *
from .ste_gaussian_conditional import *

from .ste_round import ste_round


@ENTROPYMODEL_REGISTRY.register()
class VqCategoricalEntropyModel(EntropyModel):
    def __init__(self, likelihood_bound=1e-9, *args, **kwargs):
        super().__init__(likelihood_bound=likelihood_bound, *args, **kwargs)
        self.softmax = nn.Softmax(dim=1)
        # self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, indices: Tensor, pred: Tensor, is_train: bool=True) -> Tuple[Tensor, Tensor]:
        N, C, H, W = pred.shape
        indices = indices.reshape(N, 1, H, W)
        p = self.softmax(pred) # [N, C, H, W]
        likelihood = torch.gather(p, dim=1, index=indices)
        # likelihood[i, j, k, l] = p[i, indices[i][j][k][l], k, l]

        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        return indices, likelihood