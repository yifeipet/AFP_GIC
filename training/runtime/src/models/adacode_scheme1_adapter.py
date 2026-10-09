from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from src.models.adacode_adapter import AdaCodeAdapter


class AdaCodeScheme1Adapter(AdaCodeAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_codebooks = len(self.net.quantize_group)

    def _encode_branch_targets(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        enc_feats = self.net.multiscale_encoder(x.detach())
        if self.net.LQ_stage:
            enc_feats = enc_feats[-3:]
        else:
            enc_feats = enc_feats[::-1]

        feat_before_decoder = None
        branch_feats = None
        weight_map = None
        latent = enc_feats[0]

        for i in range(self.net.max_depth):
            cur_res = self.net.gt_res // 2**self.net.max_depth * 2**i
            if cur_res in self.net.codebook_scale:
                before_quant_feat = enc_feats[i]
                cur_branch_feats = []
                for codebook_idx in range(self.num_codebooks):
                    feat_to_quant = self.net.before_quant_group[codebook_idx](
                        before_quant_feat
                    )
                    z_quant, _, _ = self.net.quantize_group[codebook_idx](feat_to_quant)
                    if not self.net.use_quantize:
                        z_quant = feat_to_quant
                    after_quant_feat = self.net.after_quant_group[codebook_idx](z_quant)
                    cur_branch_feats.append(after_quant_feat)

                branch_feats = torch.stack(cur_branch_feats, dim=1)
                weight_map = self.net.weight_predictor(before_quant_feat)
                feat_before_decoder = torch.sum(
                    branch_feats * weight_map.unsqueeze(2),
                    dim=1,
                )
                latent = feat_before_decoder
            elif self.net.LQ_stage and self.net.use_residual:
                latent = latent + enc_feats[i]

            latent = self.net.decoder_group[i](latent)

        if feat_before_decoder is None or branch_feats is None or weight_map is None:
            raise RuntimeError("AdaCode Scheme 1 encoder targets were not produced.")

        return (
            feat_before_decoder.detach(),
            branch_feats.detach(),
            weight_map.detach(),
        )

    def encode_to_prior_targets(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.net.eval()
        with torch.no_grad():
            gt_prior_fused, gt_branch_feats, gt_weight_map_raw = self._encode_branch_targets(
                x
            )

        aux = {
            "gt_prior_fused": gt_prior_fused,
            "gt_branch_feats": gt_branch_feats,
            "gt_weight_map_raw": gt_weight_map_raw,
        }
        return gt_prior_fused, aux

    def encode_to_fused_prior(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.encode_to_prior_targets(x)
