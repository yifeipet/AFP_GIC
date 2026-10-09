"""
GAN loss in MS-ILLM
paper: "Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models", ICML2023
https://github.com/facebookresearch/NeuralCompression/blob/main/neuralcompression/loss_fn/_oasis_loss.py
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.registry import LOSS_REGISTRY


def _verify_logit_target_shape(logits: Tensor, target: Tensor):
    if logits.ndim != 4:
        raise ValueError("Only expect 4-dimensional logits.")

    expected_target_numel = logits.shape[0] * logits.shape[-2] * logits.shape[-1]
    actual_target_shape = target.numel()
    logits_shape = logits.shape
    if expected_target_numel != actual_target_shape:
        raise ValueError(
            f"Based on logits size {logits_shape}, expected target numel to be "
            f"{expected_target_numel}, but found {actual_target_shape}"
        )


@LOSS_REGISTRY.register()
class OasisGANLoss(nn.Module):
    def __init__(
        self,
        loss_weight: float,
    ):
        super().__init__()
        self.lamb_gan = loss_weight

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        is_disc: bool,
        is_real: bool,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            logits (Tensor): Discriminator's outputs
            target (Tensor): VQ Tokens index (LongTensor)
            is_disc (bool): D's loss if True, G's loss if False
            is_real (bool): if False, target is replaced with 0-tensor

        Returns:
            Tensor: adv loss
        """
        _verify_logit_target_shape(logits, target)

        if target.dtype != torch.long:
            raise ValueError("Expected target to have dtype torch.long.")

        batch_size, num_classes, _, _ = logits.shape

        ## 0 is a "Fake" label
        if is_real:
            target = target + 1 # +1 to avoid 0
        else:
            target = torch.zeros_like(target)

        target = target.view(batch_size, -1)
        logits = logits.view(batch_size, num_classes, -1)
        loss = F.cross_entropy(logits, target)

        # Generator loss
        if not is_disc:
            return self.lamb_gan * loss
        # Discriminator loss
        return loss
