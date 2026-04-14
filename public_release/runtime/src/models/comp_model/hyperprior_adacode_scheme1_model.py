from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.adacode_scheme1_adapter import AdaCodeScheme1Adapter
from src.models.subnet import build_subnet
from src.models.subnet.prior_estimator.adacode_branch_prior_estimator import (
    AdaCodeBranchPriorEstimator,
)
from src.utils.img_utils import calc_ms_ssim, calc_psnr
from src.utils.registry import MODEL_REGISTRY

from .hyperprior_charm_dc_vic_model import HyperpriorCharmDualCondVicModel
from .hyperprior_charm_vic_model import HyperpriorCharmVicModel


class _AdaCodeScheme1Mixin:
    def __init__(self, *args, adacode_oracle_mode: str = "gt_all", **kwargs) -> None:
        if adacode_oracle_mode not in {
            "gt_all",
            "gt_branch_pred_weight",
            "pred_branch_gt_weight",
        }:
            raise ValueError(f"Unsupported adacode_oracle_mode: {adacode_oracle_mode}")
        self.adacode_oracle_mode = adacode_oracle_mode
        super().__init__(*args, **kwargs)

    def _build_scheme1_subnets(self) -> None:
        self.encoder = build_subnet(self.opt.subnet.encoder, subnet_type="encoder")
        self.decoder = build_subnet(self.opt.subnet.decoder, subnet_type="decoder")
        self.hyperencoder = build_subnet(
            self.opt.subnet.hyperencoder, subnet_type="hyperencoder"
        )
        self.hyperdecoder = build_subnet(
            self.opt.subnet.hyperdecoder, subnet_type="hyperdecoder"
        )
        self.entropy_model_z = build_subnet(
            self.opt.subnet.entropy_model_z, subnet_type="entropy_model"
        )
        self.entropy_model_y = build_subnet(
            self.opt.subnet.entropy_model_y, subnet_type="entropy_model"
        )
        self.context_model = build_subnet(
            self.opt.subnet.context_model, subnet_type="context_model"
        )

        self.vq_estimator = None
        self.vq_model = None
        self.fusion_module = None

        adacode_prior_opt = deepcopy(self.opt.subnet.adacode_prior)
        ckpt_path = adacode_prior_opt.pop("ckpt_path")
        self.adacode_adapter = AdaCodeScheme1Adapter(
            ckpt_path=ckpt_path,
            device=self.device,
            **adacode_prior_opt,
        )

        prior_estimator_opt = deepcopy(self.opt.subnet.adacode_scheme1_prior_estimator)
        prior_estimator_opt.pop("type", None)
        prior_estimator_opt["out_ch"] = self.adacode_adapter.prior_channels
        prior_estimator_opt["num_branches"] = self.adacode_adapter.num_codebooks
        self.prior_estimator = AdaCodeBranchPriorEstimator(**prior_estimator_opt)

        self.encoder_prior_in_ch = int(
            self.opt.subnet.encoder.get(
                "input_feat_ch",
                self.adacode_adapter.prior_channels,
            )
        )
        self.prior_to_encoder = nn.Conv2d(
            self.adacode_adapter.prior_channels,
            self.encoder_prior_in_ch,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def adacode_encode(
        self,
        real_images: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        return self.adacode_adapter.encode_to_prior_targets(real_images)

    def _prepare_scheme1_targets(
        self,
        gt_prior_fused_encoder: Tensor,
        prior_aux: Dict[str, Tensor],
    ) -> Dict[str, Optional[Tensor]]:
        gt_branch_feats = prior_aux.get("gt_branch_feats")
        gt_weight_map_raw = prior_aux.get("gt_weight_map_raw")

        if gt_branch_feats is None or gt_weight_map_raw is None:
            return {
                "gt_prior_fused": gt_prior_fused_encoder,
                "gt_branch_feats": None,
                "gt_weight_map": None,
                "gt_weight_map_raw": gt_weight_map_raw,
            }

        target_hw = gt_prior_fused_encoder.shape[-2:]
        gt_branch_feats = self.prior_estimator.resize_branch_feats(gt_branch_feats, target_hw)
        gt_weight_map_raw = self.prior_estimator.resize_weight_map(
            gt_weight_map_raw,
            target_hw,
        )
        gt_weight_map = self.prior_estimator.normalize_weight_map(gt_weight_map_raw)
        gt_prior_fused = self.prior_estimator.fuse_branch_feats(
            gt_branch_feats,
            gt_weight_map,
        )

        return {
            "gt_prior_fused": gt_prior_fused,
            "gt_branch_feats": gt_branch_feats,
            "gt_weight_map": gt_weight_map,
            "gt_weight_map_raw": gt_weight_map_raw,
        }

    def _decode_adacode(
        self,
        y_hat: Tensor,
        gt_prior_fused: Tensor,
        prior_aux: Dict[str, Tensor],
        is_train: bool = True,
        run_vq_decoder: bool = True,
        decoder_kwargs: Optional[Dict] = None,
    ) -> Dict:
        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        transformer_feat, cond_feat_dict = self.decoder.get_feats(y_hat, **decoder_kwargs)
        pred_dict = self.prior_estimator(transformer_feat)
        target_hw = gt_prior_fused.shape[-2:]

        pred_branch_feats = self.prior_estimator.resize_branch_feats(
            pred_dict["pred_branch_feats"],
            target_hw,
        )
        pred_weight_logits = self.prior_estimator.resize_weight_map(
            pred_dict["pred_weight_logits"],
            target_hw,
        )
        pred_weight_map = self.prior_estimator.normalize_weight_map(pred_weight_logits)
        pred_prior_fused = self.prior_estimator.fuse_branch_feats(
            pred_branch_feats,
            pred_weight_map,
        )

        gt_targets = self._prepare_scheme1_targets(gt_prior_fused, prior_aux)
        gt_branch_feats = gt_targets["gt_branch_feats"]
        gt_weight_map = gt_targets["gt_weight_map"]
        gt_weight_map_raw = gt_targets["gt_weight_map_raw"]
        gt_prior_fused_target = gt_targets["gt_prior_fused"]

        pred_fake_images = None
        fake_images = torch.zeros(
            y_hat.size(0),
            3,
            y_hat.size(2) * 16,
            y_hat.size(3) * 16,
            device=y_hat.device,
        )
        if run_vq_decoder:
            pred_fake_images, _ = self.adacode_adapter.decode_from_fused_prior(
                pred_prior_fused,
                cond_feat_dict=cond_feat_dict,
                w=1.0,
            )
            fake_images = pred_fake_images

        oracle_fake_images = None
        oracle_prior_fused = None
        if self.use_oracle_prior and run_vq_decoder and gt_branch_feats is not None:
            if self.adacode_oracle_mode == "gt_branch_pred_weight":
                oracle_prior_fused = self.prior_estimator.fuse_branch_feats(
                    gt_branch_feats,
                    pred_weight_map,
                )
            elif self.adacode_oracle_mode == "pred_branch_gt_weight":
                oracle_prior_fused = self.prior_estimator.fuse_branch_feats(
                    pred_branch_feats,
                    gt_weight_map,
                )
            else:
                oracle_prior_fused = gt_prior_fused_target

            oracle_fake_images, _ = self.adacode_adapter.decode_from_fused_prior(
                oracle_prior_fused,
                cond_feat_dict=cond_feat_dict,
                w=1.0,
            )
            fake_images = oracle_fake_images

        return {
            "fake_images": fake_images,
            "pred_fake_images": pred_fake_images,
            "oracle_fake_images": oracle_fake_images,
            "pred_prior_fused": pred_prior_fused,
            "gt_prior_fused": gt_prior_fused_target,
            "pred_branch_feats": pred_branch_feats,
            "gt_branch_feats": gt_branch_feats,
            "pred_weight_map": pred_weight_map,
            "gt_weight_map": gt_weight_map,
            "pred_weight_logits": pred_weight_logits,
            "gt_weight_map_raw": gt_weight_map_raw,
            "gt_prior_fused_encoder": gt_prior_fused,
            "oracle_prior_fused": oracle_prior_fused,
        }

    def _scheme1_validation_impl(
        self,
        dataloader,
        max_sample_size: int,
        **run_kwargs,
    ) -> pd.DataFrame:
        score_list = []
        sample_size = min(len(dataloader), max_sample_size)

        for idx, data in enumerate(dataloader):
            model_input = {
                k: v for k, v in data.items() if k not in {"img_path", "disc_img_path"}
            }
            with torch.no_grad():
                out_dict = self.run_model(
                    **model_input,
                    is_train=False,
                    **run_kwargs,
                )

            fake_key = "pred_fake_images" if out_dict.get("pred_fake_images") is not None else "fake_images"
            score_dict = {
                "idx": idx + 1,
                "bpp": out_dict["bpp"].item(),
                "psnr": calc_psnr(
                    out_dict["real_images"], out_dict[fake_key], 255
                ),
                "ms_ssim": calc_ms_ssim(
                    out_dict["real_images"], out_dict[fake_key]
                ),
                "vq_acc": out_dict["vq_accuracy"].item(),
            }
            if out_dict.get("pred_prior_fused") is not None and out_dict.get("gt_prior_fused") is not None:
                score_dict["prior_mse"] = F.mse_loss(
                    out_dict["pred_prior_fused"],
                    out_dict["gt_prior_fused"],
                ).item()
            if out_dict.get("pred_branch_feats") is not None and out_dict.get("gt_branch_feats") is not None:
                score_dict["branch_mse"] = F.mse_loss(
                    out_dict["pred_branch_feats"],
                    out_dict["gt_branch_feats"],
                ).item()
            if out_dict.get("pred_weight_map") is not None and out_dict.get("gt_weight_map") is not None:
                score_dict["weight_mse"] = F.mse_loss(
                    out_dict["pred_weight_map"],
                    out_dict["gt_weight_map"],
                ).item()
            if out_dict.get("oracle_fake_images") is not None:
                score_dict["oracle_psnr"] = calc_psnr(
                    out_dict["real_images"],
                    out_dict["oracle_fake_images"],
                    255,
                )
                score_dict["oracle_ms_ssim"] = calc_ms_ssim(
                    out_dict["real_images"],
                    out_dict["oracle_fake_images"],
                )

            score_list.append(score_dict)
            if idx + 1 == sample_size:
                break

        return pd.json_normalize(score_list)


@MODEL_REGISTRY.register()
class HyperpriorCharmAdaCodeScheme1Model(_AdaCodeScheme1Mixin, HyperpriorCharmVicModel):
    def _build_subnets(self) -> None:
        self._build_scheme1_subnets()

    def forward(
        self,
        real_images: Tensor,
        vq_indices: Optional[Tensor] = None,
        fusion_w: Optional[float] = None,
        is_train: bool = True,
        fix_entropy_models: bool = False,
        run_vq_decoder: bool = True,
        use_oracle_prior: Optional[bool] = None,
    ) -> Dict:
        if use_oracle_prior is not None:
            self.use_oracle_prior = use_oracle_prior

        with torch.no_grad():
            gt_prior_fused_encoder, prior_aux = self.adacode_encode(real_images)

        grad_enabled = not fix_entropy_models if is_train else False
        with torch.set_grad_enabled(grad_enabled):
            y, enc_prior_feat = self.comp_encode_adacode(
                real_images,
                gt_prior_fused_encoder,
            )
            entropy_dict = self.estimate_entropy(y, is_train=is_train)
            y_hat = entropy_dict["quantized_code"]["y"]

        decode_dict = self._decode_adacode(
            y_hat=y_hat,
            gt_prior_fused=gt_prior_fused_encoder,
            prior_aux=prior_aux,
            is_train=is_train,
            run_vq_decoder=run_vq_decoder,
        )

        return {
            **decode_dict,
            **self._empty_vq_outputs(real_images),
            "enc_prior_feat": enc_prior_feat,
            "gt_prior_aux": prior_aux,
            **entropy_dict,
        }

    def validation(
        self,
        dataloader,
        max_sample_size: int,
    ) -> pd.DataFrame:
        return self._scheme1_validation_impl(
            dataloader=dataloader,
            max_sample_size=max_sample_size,
        )


@MODEL_REGISTRY.register()
class HyperpriorCharmDualCondAdaCodeScheme1Model(
    _AdaCodeScheme1Mixin,
    HyperpriorCharmDualCondVicModel,
):
    def _build_subnets(self) -> None:
        self._build_scheme1_subnets()

    def forward(
        self,
        real_images: Tensor,
        beta_rate: float,
        beta_vq: float,
        vq_indices=None,
        fusion_w: Optional[float] = None,
        is_train: bool = True,
        fix_entropy_models: bool = False,
        use_oracle_prior: Optional[bool] = None,
    ) -> Dict:
        if use_oracle_prior is not None:
            self.use_oracle_prior = use_oracle_prior

        with torch.no_grad():
            gt_prior_fused_encoder, prior_aux = self.adacode_encode(real_images)

        grad_enabled = not fix_entropy_models if is_train else False
        with torch.set_grad_enabled(grad_enabled):
            y, enc_prior_feat = self.comp_encode_adacode(
                real_images=real_images,
                gt_prior_fused=gt_prior_fused_encoder,
                enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
            )
            entropy_dict = self.estimate_entropy(y, is_train=is_train)
            y_hat = entropy_dict["quantized_code"]["y"]

        decode_dict = self._decode_adacode(
            y_hat=y_hat,
            gt_prior_fused=gt_prior_fused_encoder,
            prior_aux=prior_aux,
            is_train=is_train,
            run_vq_decoder=True,
            decoder_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
        )

        return {
            **decode_dict,
            **self._empty_vq_outputs(real_images),
            "enc_prior_feat": enc_prior_feat,
            "gt_prior_aux": prior_aux,
            **entropy_dict,
        }

    def validation(
        self,
        dataloader,
        max_sample_size: int,
        fusion_w: Optional[float] = None,
        beta_rate: Optional[float] = None,
        beta_vq: Optional[float] = None,
    ) -> pd.DataFrame:
        beta_rate = beta_rate if beta_rate is not None else self.max_beta_rate / 2.0
        beta_vq = beta_vq if beta_vq is not None else self.max_beta_vq / 2.0
        return self._scheme1_validation_impl(
            dataloader=dataloader,
            max_sample_size=max_sample_size,
            beta_rate=beta_rate,
            beta_vq=beta_vq,
            fusion_w=fusion_w,
        )
