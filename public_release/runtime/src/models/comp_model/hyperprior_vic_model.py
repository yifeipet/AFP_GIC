"""
DC-VIC without dual-conditioning
"""

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from compressai.models import get_scale_table
from einops import rearrange
from torch import Tensor, nn
from tqdm import tqdm

from src.models.comp_model.base_model import BaseModel
from src.models.subnet import build_subnet
from src.models.subnet.vq_estimator import build_vq_estimator
from src.models.subnet.vq_fusion_module import build_vq_fusion_module
from src.models.subnet.prior_estimator.adacode_prior_estimator import (
    build_adacode_prior_estimator,
)
from src.models.vq_vae_builder import build_pretrained_prior_model
from src.utils.img_utils import calc_ms_ssim, calc_psnr, imwrite
from src.utils.logger import get_root_logger
from src.utils.registry import MODEL_REGISTRY

SPLIT_DECODE_RESOLUTION = 1024
SPLIT_WINDOW_SIZE = 512
SPLIT_STRIDE = 256

@MODEL_REGISTRY.register()
class HyperpriorVicModel(BaseModel):
    def __init__(
        self,
        opt,
        gumbel_sampling: bool = False,
        gumbel_kwargs: Dict = {},
        enc_vq_input: str = "norm_indices",
        enc_input_vq_recon: bool = False,
        prior_type: str = "vqgan",
        use_oracle_prior: bool = False,
        debug_prior_shapes: bool = False,
    ) -> None:
        assert prior_type in ["vqgan", "adacode"]
        self.prior_type = prior_type
        self.use_oracle_prior = use_oracle_prior
        self.debug_prior_shapes = debug_prior_shapes
        self._printed_prior_shape_debug = False
        self.gumbel_sampling = gumbel_sampling
        self.gumbel_kwargs = gumbel_kwargs
        super().__init__(opt)

        assert enc_vq_input in ["norm_indices", "onehot_indices", "long_indices"]
        self.enc_vq_input = enc_vq_input
        self.enc_input_vq_recon = enc_input_vq_recon

        self.n_embed = self.vq_model.n_embed if self.prior_type == "vqgan" else 0

    def _build_subnets(self) -> None:
        self.encoder = build_subnet(self.opt.subnet.encoder, subnet_type='encoder')
        self.decoder = build_subnet(self.opt.subnet.decoder, subnet_type='decoder')
        self.hyperencoder = build_subnet(self.opt.subnet.hyperencoder, subnet_type='hyperencoder')
        self.hyperdecoder = build_subnet(self.opt.subnet.hyperdecoder, subnet_type='hyperdecoder')
        self.entropy_model_z = build_subnet(self.opt.subnet.entropy_model_z, subnet_type='entropy_model')
        self.entropy_model_y = build_subnet(self.opt.subnet.entropy_model_y, subnet_type='entropy_model')
        if self.prior_type == "vqgan":
            self.vq_estimator = build_vq_estimator(self.opt.subnet.vq_estimator)
            self.vq_model = build_pretrained_prior_model(
                "vqgan", self.opt.subnet.vq_model, device=self.device
            )
            self.vq_model.quantize.sane_index_shape = True
            self.fusion_module = build_vq_fusion_module(self.opt.subnet.fusion_module)
            self.adacode_adapter = None
            self.prior_estimator = None
            self.prior_to_encoder = None
            self.encoder_prior_in_ch = None
        else:
            self.vq_estimator = None
            self.vq_model = None
            self.fusion_module = None
            self.adacode_adapter = build_pretrained_prior_model(
                "adacode", self.opt.subnet.adacode_prior, device=self.device
            )
            prior_estimator_opt = deepcopy(self.opt.subnet.adacode_prior_estimator)
            requested_out_ch = prior_estimator_opt.get("out_ch", None)
            prior_estimator_opt["out_ch"] = self.adacode_adapter.prior_channels
            if requested_out_ch is not None and requested_out_ch != self.adacode_adapter.prior_channels:
                logger = get_root_logger()
                logger.info(
                    "AdaCodePriorEstimator out_ch overridden: "
                    f"config_out_ch={requested_out_ch}, "
                    f"effective_out_ch={self.adacode_adapter.prior_channels}"
                )
            self.prior_estimator = build_adacode_prior_estimator(prior_estimator_opt)
            self.encoder_prior_in_ch = int(
                self.opt.subnet.encoder.get(
                    "input_feat_ch", self.adacode_adapter.prior_channels
                )
            )
            self.prior_to_encoder = nn.Conv2d(
                self.adacode_adapter.prior_channels,
                self.encoder_prior_in_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def get_rate_summary_dict(self, out_dict: Dict, num_pixel: int) -> Dict[str, Tensor]:
        _, y_noisy_bpp = self.likelihood_to_bit(out_dict['likelihoods']['y'], num_pixel)
        _, z_noisy_bpp = self.likelihood_to_bit(out_dict['likelihoods']['z'], num_pixel)
        _, y_quantized_bpp = self.likelihood_to_bit(out_dict['q_likelihoods']['y'], num_pixel)
        _, z_quantized_bpp = self.likelihood_to_bit(out_dict['q_likelihoods']['z'], num_pixel)
        return dict(
            y_likelihood=out_dict['likelihoods']['y'],
            z_likelihood=out_dict['likelihoods']['z'],
            bpp=y_noisy_bpp + z_noisy_bpp,
            y_q_likelihood=out_dict['q_likelihoods']['y'],
            z_q_likelihood=out_dict['q_likelihoods']['z'],
            qbpp=y_quantized_bpp + z_quantized_bpp,
        )
    
    def likelihood_to_bit(self, likelihood: Tensor, num_pixel: int) -> Tuple[Tensor, Tensor]:
        bitcost = -(torch.log(likelihood).sum()) / np.log(2)
        return bitcost, bitcost / num_pixel

    def run_model(self, real_images: Tensor, is_train: bool = True, **kwargs) -> Dict:
        img_shape = real_images.shape[2:]
        processed_input = self.data_preprocess(real_images, **kwargs, is_train=is_train)
        out_dict = self.forward(**processed_input, is_train=is_train)
        return self.data_postprocess(
            img_shape, processed_input, out_dict, is_train=is_train
        )

    def data_preprocess(
        self,
        real_images: Tensor,
        vq_indices: Optional[Tensor] = None,
        is_train: bool = True,
        fix_entropy_models: bool = False,
        fusion_w: Optional[float] = None,
        run_vq_decoder: bool = True,
        use_oracle_prior: Optional[bool] = None,
    ) -> dict:
        real_images = self.img_preprocess(real_images, is_train=is_train)
        if vq_indices is not None:
            vq_indices = vq_indices.to(self.device)

        return dict(
            real_images=real_images,
            vq_indices=vq_indices,
            fix_entropy_models=fix_entropy_models,
            fusion_w=fusion_w,
            run_vq_decoder=run_vq_decoder,
            use_oracle_prior=self.use_oracle_prior if use_oracle_prior is None else use_oracle_prior,
        )

    def data_postprocess(
        self, img_shape: Tuple, input_data: Dict, out_dict: Dict, is_train: bool = True,
    ) -> Dict:
        real_images = input_data["real_images"]

        N = real_images.size(0)
        H, W = img_shape
        num_pixel = N * H * W

        fake_images = out_dict.pop("fake_images")
        pred_fake_images = out_dict.pop("pred_fake_images", None)
        oracle_fake_images = out_dict.pop("oracle_fake_images", None)
        real_images, fake_images = self.img_postprocess(
            real_images, fake_images, size=(H, W), is_train=is_train
        )
        if pred_fake_images is not None:
            pred_fake_images = self.img_postprocess(
                pred_fake_images, size=(H, W), is_train=is_train
            )
        if oracle_fake_images is not None:
            oracle_fake_images = self.img_postprocess(
                oracle_fake_images, size=(H, W), is_train=is_train
            )
        rate_summary_dict = self.get_rate_summary_dict(out_dict, num_pixel)

        return dict(
            real_images=real_images,
            fake_images=fake_images,
            pred_fake_images=pred_fake_images,
            oracle_fake_images=oracle_fake_images,
            y_hat=out_dict['quantized_code']['y'],
            z_hat=out_dict['quantized_code']['z'],
            **out_dict,
            **rate_summary_dict,
        )
    
    def vq_encode(
        self, real_images: Tensor, vq_indices: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if vq_indices is None:
            N, _, H, W = real_images.size()
            if max(H, W) > 1024:
                _z = self._vq_encode_split(real_images)
            else:
                _z = self.vq_model.encode(real_images)
            # _z: [N, C_z, H_z, W_z]
            # gt_vq_latent: same shape as _z [N, C_z, H_z, W_z]
            # gt_vq_indices: [N, H_z, W_z]
            if max(H, W) > 1024 and self.n_embed > 1024:
                gt_vq_latent, gt_vq_indices = self._vq_quantize_split(_z)
            else:
                gt_vq_latent, _, (_, _, gt_vq_indices) = self.vq_model.quantize(_z)

            normalized_indices = gt_vq_indices.float() / (self.n_embed - 1)
            normalized_indices = normalized_indices.unsqueeze(1)
        else:
            gt_vq_indices = vq_indices
            normalized_indices = gt_vq_indices.unsqueeze(1).float() / (
                self.n_embed - 1
            )
            gt_vq_latent = self.vq_indices_to_latent(gt_vq_indices)

        return gt_vq_latent, gt_vq_indices
    
    def vq_indices_to_latent(self, indices):
        vq_latent = self.vq_model.quantize.embedding(indices)  # [N, H, W, D]
        vq_latent = rearrange(vq_latent, "b h w c -> b c h w").contiguous()
        return vq_latent
    
    def _vq_quantize_split(self, z):
        N, C, H, W = z.size()
        out_vq_latent = torch.zeros_like(z)
        out_vq_indices = torch.zeros(N, H, W, device=z.device, dtype=torch.long)
        out_vq_indices = out_vq_indices - 1
        stride = 64
        for h in range(0, H, stride):
            for w in range(0, W, stride):
                crop = z[:, :, h : h + stride, w : w + stride]
                crop_latent, _, (_, _, crop_indices) = self.vq_model.quantize(
                    crop
                )
                out_vq_latent[:, :, h : h + stride, w : w + stride] = crop_latent
                out_vq_indices[:, h : h + stride, w : w + stride] = crop_indices

        assert out_vq_indices.min() >= 0
        assert out_vq_indices.max() < self.n_embed

        return out_vq_latent, out_vq_indices
    
    def _vq_encode_split(self, real_images: Tensor) -> Tensor:
        N, _, H, W = real_images.size()
        stride = 256
        patch_size = 512
        df = 2 ** (self.vq_model.encoder.num_resolutions - 1)
        ndim = self.vq_model.embed_dim

        left_list = []
        for i in range(W // stride + 1):
            left = i * stride
            if left + patch_size < W:
                left_list.append(left)
            else:
                left_list.append(W - patch_size)
                break

        top_list = []
        for i in range(H // stride + 1):
            top = i * stride
            if top + patch_size < H:
                top_list.append(top)
            else:
                top_list.append(H - patch_size)
                break

        z_out = torch.zeros(N, ndim, H // df, W // df).to(real_images.device)

        for y0 in top_list:
            for x0 in left_list:
                crop = real_images[
                    :, :, y0 : y0 + patch_size, x0 : x0 + patch_size
                ]
                z = self.vq_model.encode(
                    crop
                )  # [N, C, patch_size//df, patch_size//df]

                offset = (stride // 2) // df
                _x0 = x0 // df
                _y0 = y0 // df
                l = _x0 + offset if x0 > 0 else 0
                t = _y0 + offset if y0 > 0 else 0
                r = (
                    _x0 + offset + stride // df
                    if x0 < left_list[-1]
                    else W // df
                )
                b = (
                    _y0 + offset + stride // df
                    if y0 < top_list[-1]
                    else H // df
                )

                z_out[:, :, t:b, l:r] = z[
                    :, :, t - _y0 : b - _y0, l - _x0 : r - _x0
                ]

        return z_out

    def comp_encode(
            self,
            real_images: Tensor,
            gt_vq_latent: Tensor,
            gt_vq_indices: Tensor,
            enc_kwargs: Dict = {}
        ) -> Tensor:
        """Run compression encoder and return latent code y
        """
        enc_image_input = real_images
        if self.enc_input_vq_recon:
            with torch.no_grad():
                vq_recon = self.vq_decode_from_indices(gt_vq_indices)
            enc_image_input = torch.cat([real_images, vq_recon], dim=1)

        if hasattr(self.encoder, "input_vq_latent"):
            if self.enc_vq_input == "norm_indices":
                normalized_indices = gt_vq_indices.float() / (self.n_embed - 1)
                normalized_indices = normalized_indices.unsqueeze(1).detach()
                vq_ind_feat = normalized_indices
            elif self.enc_vq_input == "onehot_indices":
                onehot_indices = F.one_hot(gt_vq_indices, num_classes=self.n_embed)
                onehot_indices = onehot_indices.permute(0, 3, 1, 2).float().detach()
                vq_ind_feat = onehot_indices
            elif self.enc_vq_input == "long_indices":
                vq_ind_feat = gt_vq_indices
            else:
                raise NotImplementedError()

            if self.enc_vq_input != "long_indices":
                enc_vq_feat = torch.cat([gt_vq_latent, vq_ind_feat], dim=1)
            else:
                enc_vq_feat = gt_vq_latent
                enc_kwargs['vq_indices'] = gt_vq_indices

            y = self.encoder(
                enc_image_input,
                enc_vq_feat,
                **enc_kwargs,
            )
        else:
            y = self.encoder(enc_image_input, **enc_kwargs)
        return y

    def adacode_encode(
        self, real_images: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        assert self.adacode_adapter is not None
        gt_prior_fused, prior_aux = self.adacode_adapter.encode_to_fused_prior(real_images)
        return gt_prior_fused, prior_aux

    def _align_prior_grid(self, prior_feat: Tensor, real_images: Tensor) -> Tensor:
        # Phase-1 keeps the original DC-VIC f=8 token grid interface.
        target_hw = (real_images.shape[2] // 8, real_images.shape[3] // 8)
        if prior_feat.shape[2:] != target_hw:
            prior_feat = F.interpolate(
                prior_feat,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return prior_feat

    def comp_encode_adacode(
        self,
        real_images: Tensor,
        gt_prior_fused: Tensor,
        enc_kwargs: Dict = {},
    ) -> Tuple[Tensor, Tensor]:
        prior_feat = self._align_prior_grid(gt_prior_fused, real_images)
        assert self.prior_to_encoder is not None
        enc_prior_feat = self.prior_to_encoder(prior_feat)
        if hasattr(self.encoder, "input_vq_latent"):
            y = self.encoder(real_images, enc_prior_feat, **enc_kwargs)
        else:
            y = self.encoder(real_images, **enc_kwargs)
        return y, enc_prior_feat

    def _empty_vq_outputs(self, ref_tensor: Tensor) -> Dict[str, Optional[Tensor]]:
        zero = torch.zeros(1, device=ref_tensor.device)
        return dict(
            out_vq_latent=None,
            gt_vq_latent=None,
            out_vq_logits=None,
            gt_vq_indices=None,
            vq_accuracy=zero,
        )

    def _decode_adacode(
        self,
        y_hat: Tensor,
        gt_prior_fused: Tensor,
        is_train: bool = True,
        run_vq_decoder: bool = True,
        decoder_kwargs: Optional[Dict] = None,
    ) -> Dict:
        assert self.prior_estimator is not None
        assert self.adacode_adapter is not None
        decoder_kwargs = {} if decoder_kwargs is None else decoder_kwargs

        transformer_feat, cond_feat_dict = self.decoder.get_feats(y_hat, **decoder_kwargs)
        estimator_raw = self.prior_estimator(transformer_feat)
        pred_prior_fused = estimator_raw
        if pred_prior_fused.shape[2:] != gt_prior_fused.shape[2:]:
            pred_prior_fused = F.interpolate(
                pred_prior_fused,
                size=gt_prior_fused.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        assert (
            pred_prior_fused.shape[1] == gt_prior_fused.shape[1]
        ), (
            "AdaCode prior channel mismatch after adaptation: "
            f"estimator_raw={tuple(estimator_raw.shape)}, "
            f"pred_prior_fused_after_adaptation={tuple(pred_prior_fused.shape)}, "
            f"gt_prior_fused={tuple(gt_prior_fused.shape)}"
        )
        if self.debug_prior_shapes and not self._printed_prior_shape_debug:
            logger = get_root_logger()
            logger.info(f"estimator_raw.shape: {tuple(estimator_raw.shape)}")
            logger.info(
                "pred_prior_fused_after_adaptation.shape: "
                f"{tuple(pred_prior_fused.shape)}"
            )
            logger.info(f"gt_prior_fused.shape: {tuple(gt_prior_fused.shape)}")
            self._printed_prior_shape_debug = True

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
                pred_prior_fused, cond_feat_dict=cond_feat_dict, w=1.0
            )
            fake_images = pred_fake_images

        oracle_fake_images = None
        if self.use_oracle_prior and run_vq_decoder:
            oracle_fake_images, _ = self.adacode_adapter.decode_from_fused_prior(
                gt_prior_fused, cond_feat_dict=cond_feat_dict, w=1.0
            )
            fake_images = oracle_fake_images

        return dict(
            fake_images=fake_images,
            pred_fake_images=pred_fake_images,
            oracle_fake_images=oracle_fake_images,
            pred_prior_fused=pred_prior_fused,
            gt_prior_fused=gt_prior_fused,
        )
    
    def estimate_entropy(self, y: Tensor, is_train: bool=True):
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood = self.entropy_model_y(
            y, hyper_out, is_train=is_train
        )

        with torch.no_grad():
            _, z_q_likelihood = self.entropy_model_z(z, is_train=False)
            _, y_q_likelihood = self.entropy_model_y(
                y, hyper_out, is_train=False
            )

        return {
            "quantized_code": {
                "y": y_hat,
                "z": z_hat,
            },
            "latent_code": {
                "y": y,
                "z": z,
            },
            "likelihoods": {
                "y": y_likelihood,
                "z": z_likelihood,
            },
            "q_likelihoods": {
                "y": y_q_likelihood,
                "z": z_q_likelihood,
            },
        }

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

        if self.prior_type == "vqgan":
            ##### Encode part #################################
            with torch.no_grad():
                gt_vq_latent, gt_vq_indices = self.vq_encode(real_images, vq_indices)

            grad_enabled = not (fix_entropy_models) if is_train else False
            with torch.set_grad_enabled(grad_enabled):
                y = self.comp_encode(
                    real_images,
                    gt_vq_latent,
                    gt_vq_indices,
                )
                entropy_dict = self.estimate_entropy(y, is_train=is_train)
                y_hat = entropy_dict["quantized_code"]["y"]

            ##### Decode part #################################
            w = 1.0
            do_split_decode = max(real_images.shape[2:]) > SPLIT_DECODE_RESOLUTION
            if do_split_decode:
                assert not is_train, 'split-decoding is only available in inference'
                fake_images = self.decode_split(y_hat, w)
                out_vq_latent = None
                out_vq_logits = None
                out_vq_indices = torch.zeros_like(gt_vq_indices)
                vq_accuracy = torch.zeros(1).to(fake_images.device)
            else:
                transformer_feat, cond_feat_dict = self.decoder.get_feats(y_hat)
                out_vq_latent, out_vq_logits = self.vq_estimator(transformer_feat)
                out_vq_indices = torch.argmax(out_vq_logits, dim=1)  # [N, H, W]
                vq_accuracy = (out_vq_indices == gt_vq_indices).float().mean()

                if is_train and self.gumbel_sampling:
                    vq_latent = self.gumbel_vq_latent_sample(
                        out_vq_logits, gumbel_kwargs=self.gumbel_kwargs
                    )
                else:
                    vq_latent = self.vq_indices_to_latent(out_vq_indices)
                vq_latent = self.vq_model.post_quant_conv(vq_latent)

                if run_vq_decoder:
                    fake_images = self.fusion_module(
                        vq_latent, cond_feat_dict, self.vq_model.decoder, w=w
                    )
                else:
                    fake_images = torch.zeros_like(real_images)

            return {
                "fake_images": fake_images,
                "pred_fake_images": None,
                "oracle_fake_images": None,
                "out_vq_latent": out_vq_latent,
                "gt_vq_latent": gt_vq_latent,
                "out_vq_logits": out_vq_logits,
                "gt_vq_indices": gt_vq_indices,
                "vq_accuracy": vq_accuracy,
                **entropy_dict,
            }

        with torch.no_grad():
            gt_prior_fused, prior_aux = self.adacode_encode(real_images)

        grad_enabled = not (fix_entropy_models) if is_train else False
        with torch.set_grad_enabled(grad_enabled):
            y, enc_prior_feat = self.comp_encode_adacode(real_images, gt_prior_fused)
            entropy_dict = self.estimate_entropy(y, is_train=is_train)
            y_hat = entropy_dict["quantized_code"]["y"]

        decode_dict = self._decode_adacode(
            y_hat=y_hat,
            gt_prior_fused=gt_prior_fused,
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
    
    def gumbel_vq_latent_sample(
        self, vq_logits: Tensor, gumbel_kwargs: Dict = {}
    ) -> Tensor:
        _, _, H, W = vq_logits.shape
        one_hot = F.gumbel_softmax(
            vq_logits, dim=1, hard=True, **gumbel_kwargs
        )  # [N, C, H, W] one-hot
        one_hot = rearrange(one_hot, "b c h w -> b (h w) c").contiguous()
        codebook = self.vq_model.quantize.embedding.weight.detach()  # [C, 4]
        vq_latent = torch.matmul(one_hot, codebook)  # [N, HW, 4]
        vq_latent = rearrange(
            vq_latent, "b (h w) c -> b c h w", h=H, w=W
        ).contiguous()
        return vq_latent
    
    def vq_decode_from_indices(self, indices):
        vq_latent = self.vq_indices_to_latent(indices)
        vq_recon = self.vq_model.decode(vq_latent, force_not_quantize=False)
        return vq_recon

    def decode_split(self, y_hat, fuse_w, **kwargs):
        N, _, yH, yW = y_hat.size()

        df = 16
        stride = SPLIT_STRIDE // df
        patch_size = SPLIT_WINDOW_SIZE // df

        left_list = []
        for l in range(0, yW, stride):
            r = l + patch_size
            if r < yW:
                left_list.append(l)
            else:
                left_list.append(yW - patch_size)
                break

        top_list = []
        for t in range(0, yH, stride):
            b = t + patch_size
            if b < yH:
                top_list.append(t)
            else:
                top_list.append(yH - patch_size)
                break

        fake_images = torch.zeros(
            (N, 3, yH * df, yW * df), device="cpu"
        ).fill_(-100.0)

        for y0 in top_list:
            for x0 in left_list:
                input_crop = y_hat[
                    :, :, y0 : y0 + patch_size, x0 : x0 + patch_size
                ]
                out_patch = self._decode(
                    input_crop, w=fuse_w, **kwargs
                )  # [N, 3, 512, 512]

                offset = (stride // 2) * df
                _x0 = x0 * df
                _y0 = y0 * df
                l = _x0 + offset if x0 > 0 else 0
                t = _y0 + offset if y0 > 0 else 0
                r = (
                    _x0 + offset + stride * df
                    if x0 < left_list[-1]
                    else yW * df
                )
                b = (
                    _y0 + offset + stride * df
                    if y0 < top_list[-1]
                    else yH * df
                )

                fake_images[:, :, t:b, l:r] = out_patch[
                    :, :, t - _y0 : b - _y0, l - _x0 : r - _x0
                ].cpu()

        fake_images = fake_images.to(y_hat.device)

        return fake_images

    def _decode(self, y_hat, w):
        transformer_feat, cond_feat_dict = self.decoder.get_feats(y_hat)
        out_vq_latent, out_vq_logits = self.vq_estimator(transformer_feat)
        out_vq_indices = torch.argmax(out_vq_logits, dim=1)  # [N, H, W]
        vq_latent = self.vq_indices_to_latent(out_vq_indices)
        vq_latent = self.vq_model.post_quant_conv(vq_latent)
        fake_images = self.fusion_module(
            vq_latent, cond_feat_dict, self.vq_model.decoder, w=w
        )
        return fake_images

    def validation(
        self,
        dataloader,
        max_sample_size: int,
    ) -> pd.DataFrame:
        score_list = []

        sample_size = min(len(dataloader), max_sample_size)

        for idx, data in enumerate(dataloader):
            model_input = {
                k: v for k, v in data.items() if k not in {"img_path", "disc_img_path"}
            }
            with torch.no_grad():
                out_dict = self.run_model(**model_input, is_train=False)

            score_dict = {
                "idx": idx + 1,
                "bpp": out_dict["bpp"].item(),
                "psnr": calc_psnr(
                    out_dict["real_images"], out_dict["fake_images"], 255
                ),
                "ms_ssim": calc_ms_ssim(
                    out_dict["real_images"], out_dict["fake_images"]
                ),
            }
            if self.prior_type == "vqgan":
                score_dict["vq_acc"] = out_dict["vq_accuracy"].item()
            else:
                if out_dict.get("pred_prior_fused") is not None and out_dict.get("gt_prior_fused") is not None:
                    score_dict["prior_mse"] = F.mse_loss(
                        out_dict["pred_prior_fused"], out_dict["gt_prior_fused"]
                    ).item()
                score_dict["vq_acc"] = out_dict["vq_accuracy"].item()
                if out_dict.get("oracle_fake_images") is not None:
                    score_dict["oracle_psnr"] = calc_psnr(
                        out_dict["real_images"], out_dict["oracle_fake_images"], 255
                    )
                    score_dict["oracle_ms_ssim"] = calc_ms_ssim(
                        out_dict["real_images"], out_dict["oracle_fake_images"]
                    )

            score_list.append(score_dict)

            if idx + 1 == sample_size:
                break

        return pd.json_normalize(score_list)

    def codec_setup(self):
        self.entropy_model_z.update(force=True)
        scale_table = get_scale_table()
        self.entropy_model_y.update_scale_table(scale_table, force=True)

        # use cpu in models involving entropy coding
        self.entropy_model_z.to('cpu')
        self.entropy_model_y.to('cpu')
        self.hyperdecoder.to('cpu')

        size = 256
        tmp = torch.rand((1, 3, size, size), device=self.device)
        with torch.no_grad():
            y = self.encoder(tmp)
            z = self.hyperencoder(y)
        _, self.yC, yH, _ = y.shape
        _, self.zC, zH, _ = z.shape
        self.model_stride = size // zH
        self.y_stride = size // yH
