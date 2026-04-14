from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.layer.codeformer_layers import FuseSftBlock
from src.models.layer.light_fuse_layer import LightFuseSftBlock
from src.utils.logger import get_root_logger

from external_libs.adacode.adacode_arch import AdaCodeSRNet


class AdaCodeFusionWrapper(nn.Module):
    def __init__(
        self,
        decoder_channels: Tuple[int, ...],
        cond_keys: Tuple[str, ...] = ("block_1_8", "block_1_4", "block_1_2"),
        cond_ch: int = 192,
        mid_ch_scale: float = 1.0,
        fuse_type: str = "sft",
    ) -> None:
        super().__init__()
        assert fuse_type in {"sft", "light_sft"}
        self.cond_keys = cond_keys
        self.blocks = nn.ModuleDict()
        block_cls = FuseSftBlock if fuse_type == "sft" else LightFuseSftBlock
        for idx, dec_ch in enumerate(decoder_channels[: len(cond_keys)]):
            key = cond_keys[idx]
            mid_ch = max(int(dec_ch * mid_ch_scale), dec_ch)
            self.blocks[key] = block_cls(cond_ch=cond_ch, dec_ch=dec_ch, mid_ch=mid_ch)

    def forward(
        self,
        x: Tensor,
        stage_idx: int,
        cond_feat_dict: Optional[Dict[str, Tensor]],
        w: float = 1.0,
    ) -> Tensor:
        if cond_feat_dict is None or stage_idx >= len(self.cond_keys):
            return x
        cond_key = self.cond_keys[stage_idx]
        if cond_key not in cond_feat_dict or cond_key not in self.blocks:
            return x
        cond_feat = cond_feat_dict[cond_key]
        if cond_feat.shape[2:] != x.shape[2:]:
            cond_feat = F.interpolate(
                cond_feat, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return self.blocks[cond_key](x, cond_feat, w=w)


class AdaCodeAdapter(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        device: str,
        gt_resolution: int = 256,
        codebook_params: Optional[list[list[int]]] = None,
        weight_softmax: bool = False,
        freeze_pretrained: bool = True,
        strict_load: bool = False,
        cond_ch: int = 192,
        cond_keys: Tuple[str, ...] = ("block_1_8", "block_1_4", "block_1_2"),
        fuse_type: str = "sft",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.ckpt_path = ckpt_path
        self.gt_resolution = gt_resolution
        self.weight_softmax = weight_softmax
        self.strict_load = strict_load

        if codebook_params is None:
            if not ckpt_path:
                raise ValueError(
                    "AdaCodeAdapter requires either a ckpt_path or explicit codebook_params."
                )
            codebook_params = self._infer_codebook_params(ckpt_path)

        model = AdaCodeSRNet(
            codebook_params=codebook_params,
            gt_resolution=gt_resolution,
            AdaCode_stage=True,
            LQ_stage=False,
            weight_softmax=weight_softmax,
        )
        if ckpt_path:
            self._load_pretrained(model, ckpt_path, strict_load)
        else:
            logger = get_root_logger()
            logger.info(
                "No external AdaCode checkpoint is provided. "
                "The adapter will be initialized from explicit codebook_params and "
                "later populated by the released AFP-GIC checkpoint."
            )
        self.net = model.to(self.device)

        self.prior_channels = int(self.net.decoder_group[0].block[1].in_channels)
        decoder_channels = tuple(
            int(block.block[1].out_channels) for block in self.net.decoder_group
        )
        self.fusion_wrapper = AdaCodeFusionWrapper(
            decoder_channels=decoder_channels,
            cond_keys=cond_keys,
            cond_ch=cond_ch,
            fuse_type=fuse_type,
        )

        if freeze_pretrained:
            self.freeze_pretrained()

    @staticmethod
    def _select_state_dict(ckpt: Dict) -> Dict[str, Tensor]:
        for key in ("params_ema", "params", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    @classmethod
    def _infer_codebook_params(cls, ckpt_path: str) -> list[list[int]]:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = cls._select_state_dict(ckpt)
        codebook_params = []
        idx = 0
        while True:
            key = f"quantize_group.{idx}.embedding.weight"
            if key not in state_dict:
                break
            weight = state_dict[key]
            codebook_params.append([32, int(weight.shape[0]), int(weight.shape[1])])
            idx += 1
        if not codebook_params:
            raise KeyError(f"No AdaCode codebook weights found in checkpoint: {ckpt_path}")
        return codebook_params

    @classmethod
    def _load_pretrained(cls, model: nn.Module, ckpt_path: str, strict: bool) -> None:
        logger = get_root_logger()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = cls._select_state_dict(ckpt)
        total_ckpt_keys = len(state_dict)
        model_state = model.state_dict()
        matched_keys = []
        skipped_shape_keys = []
        missing_keys = []
        filtered_state_dict = state_dict
        if not strict:
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k not in model_state:
                    continue
                if model_state[k].shape == v.shape:
                    filtered_state_dict[k] = v
                    matched_keys.append(k)
                else:
                    skipped_shape_keys.append(k)
            missing_keys = sorted([k for k in model_state.keys() if k not in filtered_state_dict])
        else:
            matched_keys = sorted([k for k in state_dict.keys() if k in model_state])
        load_msg = model.load_state_dict(filtered_state_dict, strict=strict)
        if strict:
            missing_keys = list(getattr(load_msg, "missing_keys", []))
            skipped_shape_keys = []

        matched_ratio = len(matched_keys) / max(total_ckpt_keys, 1)
        logger.info(
            "AdaCode checkpoint load summary: "
            f"total_ckpt_keys={total_ckpt_keys}, "
            f"matched_keys={len(matched_keys)}, "
            f"missing_keys={len(missing_keys)}, "
            f"shape_mismatch_skipped={len(skipped_shape_keys)}, "
            f"matched_ratio={matched_ratio:.4f}"
        )
        if skipped_shape_keys:
            logger.info(
                f"AdaCode skipped shape-mismatch keys (first 20): {skipped_shape_keys[:20]}"
            )
        if missing_keys:
            logger.info(f"AdaCode missing model keys (first 20): {missing_keys[:20]}")
        unexpected = list(getattr(load_msg, "unexpected_keys", []))
        if unexpected:
            logger.info(f"AdaCode unexpected checkpoint keys (first 20): {unexpected[:20]}")

    def freeze_pretrained(self) -> None:
        self.net.requires_grad_(False)
        self.net.eval()

    def encode_to_fused_prior(
        self, x: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self.net.eval()
        with torch.no_grad():
            _, _, aux = self.net.encode_and_decode(x)
        fused_prior = aux["feat_before_decoder"]
        assert fused_prior.ndim == 4
        return fused_prior, aux

    def decode_from_fused_prior(
        self,
        z: Tensor,
        cond_feat_dict: Optional[Dict[str, Tensor]] = None,
        w: float = 1.0,
        return_intermediates: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        assert z.ndim == 4, f"expected [B,C,H,W], got {tuple(z.shape)}"
        assert (
            z.shape[1] == self.prior_channels
        ), f"expected prior channels={self.prior_channels}, got {z.shape[1]}"

        feats = {}
        x = z
        for idx, block in enumerate(self.net.decoder_group):
            x = block(x)
            x = self.fusion_wrapper(x, idx, cond_feat_dict, w=w)
            if return_intermediates:
                feats[f"decoder_stage_{idx}"] = x

        out_img = self.net.out_conv(x)
        if return_intermediates:
            feats["decoder_out_feat"] = x
            return out_img, feats
        return out_img, None
