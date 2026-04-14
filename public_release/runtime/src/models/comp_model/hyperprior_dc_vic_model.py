from typing import List, Dict, Tuple, Optional

import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from compressai.models import get_scale_table

from src.utils.registry import MODEL_REGISTRY
from src.utils.img_utils import calc_psnr, calc_ms_ssim, imwrite
from src.utils.logger import get_root_logger
from src.utils.codec_utils import HeaderHandler

from .hyperprior_vic_model import (
    HyperpriorVicModel,
)

SPLIT_DECODE_RESOLUTION = 1024


@MODEL_REGISTRY.register()
class HyperpriorDualCondVicModel(HyperpriorVicModel):
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
        num_beta_levels: int = 100,
        # instead of sampling beta from uniform distribution,
        # sample beta from pre-defined discrete beta_list
        use_selected_beta_pairs: bool = False,
        selected_beta_rate: Optional[List[float]] = None,
        selected_beta_vq: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            opt,
            gumbel_sampling=gumbel_sampling,
            gumbel_kwargs=gumbel_kwargs,
            enc_vq_input=enc_vq_input,
            enc_input_vq_recon=enc_input_vq_recon,
            prior_type=prior_type,
            use_oracle_prior=use_oracle_prior,
            debug_prior_shapes=debug_prior_shapes,
        )
        logger = get_root_logger()

        self.max_beta_rate: float = opt.subnet.decoder.max_beta_1
        self.max_beta_vq: float = opt.subnet.decoder.max_beta_2
        self.num_beta_levels = num_beta_levels

        self.use_selected_beta_pairs = use_selected_beta_pairs
        self.selected_beta_rate = selected_beta_rate
        self.selected_beta_vq = selected_beta_vq

        if self.use_selected_beta_pairs:
            assert isinstance(self.selected_beta_rate, list)
            assert isinstance(self.selected_beta_vq, list)
            assert len(self.selected_beta_rate) == len(self.selected_beta_vq)
            logger.info("use_selected_beta_pairs: True")
            logger.info(f"selected_beta_rate: {self.selected_beta_rate}")
            logger.info(f"selected_beta_vq: {self.selected_beta_vq}")

    def codec_setup(self):
        self.entropy_model_z.update(force=True)
        scale_table = get_scale_table()
        self.entropy_model_y.update_scale_table(scale_table, force=True)

        self.entropy_model_z.to("cpu")
        self.entropy_model_y.to("cpu")
        self.hyperdecoder.to("cpu")

        size = 256
        tmp = torch.rand((1, 3, size, size), device=self.device)
        with torch.no_grad():
            if self.prior_type == "vqgan":
                gt_vq_latent, gt_vq_indices = self.vq_encode(tmp, vq_indices=None)
                y = self.comp_encode(
                    real_images=tmp,
                    gt_vq_latent=gt_vq_latent,
                    gt_vq_indices=gt_vq_indices,
                    enc_kwargs=dict(beta_1=0.0, beta_2=0.0),
                )
            else:
                gt_prior_fused, _ = self.adacode_encode(tmp)
                y, _ = self.comp_encode_adacode(
                    real_images=tmp,
                    gt_prior_fused=gt_prior_fused,
                    enc_kwargs=dict(beta_1=0.0, beta_2=0.0),
                )
            z = self.hyperencoder(y)
        _, self.yC, yH, _ = y.shape
        _, self.zC, zH, _ = z.shape
        self.model_stride = size // zH
        self.y_stride = size // yH

    def _sample_beta(
        self, max_beta: float, num_levels: int = 100, num_samples: int = 1
    ) -> Tensor:
        i = np.random.randint(0, num_levels + 1, num_samples).astype(np.float32)
        beta = max_beta * (i / float(num_levels))
        beta = torch.Tensor(beta)
        return beta

    def _sample_selected_beta_pair(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        assert isinstance(self.selected_beta_rate, list)
        assert isinstance(self.selected_beta_vq, list)

        num_levels = len(self.selected_beta_rate)
        i = np.random.randint(0, num_levels, num_samples)

        _beta_rate = [self.selected_beta_rate[_i] for _i in i]
        _beta_vq = [self.selected_beta_vq[_i] for _i in i]
        beta_rate = torch.Tensor(_beta_rate).float()
        beta_vq = torch.Tensor(_beta_vq).float()
        return beta_rate, beta_vq

    def run_model(self, real_images: Tensor, is_train: bool = True, **input) -> Dict:
        img_shape = real_images.shape[2:]
        processed_input = self.data_preprocess(real_images, **input, is_train=is_train)
        out_dict = self.forward(**processed_input, is_train=is_train)
        return self.data_postprocess(
            img_shape, processed_input, out_dict, is_train=is_train
        )

    def data_preprocess(
        self,
        real_images: Tensor,
        vq_indices: Optional[Tensor] = None,
        beta_rate: Optional[Tensor] = None,
        beta_vq: Optional[Tensor] = None,
        is_train: bool = True,
        fix_entropy_models: bool = False,
        fusion_w: Optional[float] = None,
        sample_batch_beta: bool = False,
        use_oracle_prior: Optional[bool] = None,
    ) -> dict:
        real_images = self.img_preprocess(real_images, is_train=is_train)
        if vq_indices is not None:
            vq_indices = vq_indices.to(self.device)

        if self.use_selected_beta_pairs:
            if beta_rate is None or beta_vq is None:
                if not is_train:
                    raise ValueError(
                        '"beta_rate" and "beta_vq" must be specified if is_train=False'
                    )
                num_samples = real_images.size(0) if sample_batch_beta else 1
                beta_rate, beta_vq = self._sample_selected_beta_pair(num_samples)
        else:
            if beta_rate is None or beta_vq is None:
                if not is_train:
                    raise ValueError(
                        '"beta_rate" and "beta_vq" must be specified if is_train=False'
                    )
            num_samples = real_images.size(0) if sample_batch_beta else 1
            if beta_rate is None:
                beta_rate = self._sample_beta(
                    self.max_beta_rate,
                    num_levels=self.num_beta_levels,
                    num_samples=num_samples,
                )
            if beta_vq is None:
                beta_vq = self._sample_beta(
                    self.max_beta_vq,
                    num_levels=self.num_beta_levels,
                    num_samples=num_samples,
                )

        return dict(
            real_images=real_images,
            beta_rate=beta_rate,
            beta_vq=beta_vq,
            vq_indices=vq_indices,
            fix_entropy_models=fix_entropy_models,
            fusion_w=fusion_w,
            use_oracle_prior=self.use_oracle_prior if use_oracle_prior is None else use_oracle_prior,
        )

    def data_postprocess(
        self, img_shape: Tuple, input_data: Dict, out_dict: Dict, is_train: bool = True
    ) -> Dict:
        out_dict = super().data_postprocess(
            img_shape=img_shape, input_data=input_data, out_dict=out_dict, is_train=is_train
        )
        return dict(
            **out_dict,
            beta_rate=input_data["beta_rate"],
            beta_vq=input_data["beta_vq"],
        )
        # real_images = input_data["real_images"]
        # beta_rate = input_data["beta_rate"]
        # beta_vq = input_data["beta_vq"]

        # N = real_images.size(0)
        # H, W = img_shape
        # num_pixel = N * H * W

        # fake_images = out_dict.pop("fake_images")
        # real_images, fake_images = self.img_postprocess(
        #     real_images, fake_images, size=(H, W), is_train=is_train
        # )
        # rate_summary_dict = self.get_rate_summary_dict(out_dict, num_pixel)

        # return dict(
        #     real_images=real_images,
        #     fake_images=fake_images,
        #     beta_rate=beta_rate,
        #     beta_vq=beta_vq,
        #     y_hat=out_dict["quantized_code"]["y"],
        #     z_hat=out_dict["quantized_code"]["z"],
        #     **out_dict,
        #     **rate_summary_dict,
        # )

    def forward(
        self,
        real_images,
        beta_rate: float,
        beta_vq: float,
        vq_indices=None,
        fusion_w: Optional[float] = None,
        is_train=True,
        fix_entropy_models: bool = False,
        use_oracle_prior: Optional[bool] = None,
    ):
        if use_oracle_prior is not None:
            self.use_oracle_prior = use_oracle_prior

        if self.prior_type == "vqgan":
            with torch.no_grad():
                gt_vq_latent, gt_vq_indices = self.vq_encode(real_images, vq_indices)

            grad_enabled = not (fix_entropy_models) if is_train else False
            with torch.set_grad_enabled(grad_enabled):
                y = self.comp_encode(
                    real_images=real_images,
                    gt_vq_latent=gt_vq_latent,
                    gt_vq_indices=gt_vq_indices,
                    enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
                )
                entropy_dict = self.estimate_entropy(y, is_train=is_train)
                y_hat = entropy_dict["quantized_code"]["y"]

            w = 1.0
            do_split_decode = max(real_images.shape[2:]) > SPLIT_DECODE_RESOLUTION
            if do_split_decode:
                assert not is_train, "split-decoding is only available in inference"
                fake_images = self.decode_split(
                    y_hat, w, beta_rate=beta_rate, beta_vq=beta_vq
                )
                out_vq_latent, out_vq_logits = None, None
                out_vq_indices = torch.zeros_like(gt_vq_indices)
                vq_accuracy = torch.zeros(1).to(fake_images.device)
            else:
                transformer_feat, cond_feat_dict = self.decoder.get_feats(
                    y_hat, beta_1=beta_rate, beta_2=beta_vq
                )
                out_vq_latent, out_vq_logits = self.vq_estimator(transformer_feat)
                out_vq_indices = torch.argmax(out_vq_logits, dim=1)
                vq_accuracy = (out_vq_indices == gt_vq_indices).float().mean()

                if is_train and self.gumbel_sampling:
                    vq_latent = self.gumbel_vq_latent_sample(
                        out_vq_logits, gumbel_kwargs=self.gumbel_kwargs
                    )
                else:
                    vq_latent = self.vq_indices_to_latent(out_vq_indices)
                vq_latent = self.vq_model.post_quant_conv(vq_latent)

                fake_images = self.fusion_module(
                    vq_latent, cond_feat_dict, self.vq_model.decoder, w=w
                )

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
            y, enc_prior_feat = self.comp_encode_adacode(
                real_images=real_images,
                gt_prior_fused=gt_prior_fused,
                enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
            )
            entropy_dict = self.estimate_entropy(y, is_train=is_train)
            y_hat = entropy_dict["quantized_code"]["y"]

        decode_dict = self._decode_adacode(
            y_hat=y_hat,
            gt_prior_fused=gt_prior_fused,
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

    def _decode(self, y_hat, w, beta_rate, beta_vq):
        """used in decode_split"""
        transformer_feat, cond_feat_dict = self.decoder.get_feats(
            y_hat, beta_1=beta_rate, beta_2=beta_vq
        )
        if self.prior_type == "vqgan":
            out_vq_latent, out_vq_logits = self.vq_estimator(transformer_feat)
            out_vq_indices = torch.argmax(out_vq_logits, dim=1)  # [N, H, W]
            vq_latent = self.vq_indices_to_latent(out_vq_indices)
            vq_latent = self.vq_model.post_quant_conv(vq_latent)
            fake_images = self.fusion_module(
                vq_latent, cond_feat_dict, self.vq_model.decoder, w=w
            )
            return fake_images

        assert self.prior_estimator is not None
        assert self.adacode_adapter is not None
        pred_prior_fused = self.prior_estimator(transformer_feat)
        fake_images, _ = self.adacode_adapter.decode_from_fused_prior(
            pred_prior_fused, cond_feat_dict=cond_feat_dict, w=w
        )
        return fake_images

    def extract_y_hat(self, real_images, vq_indices, beta_rate, beta_vq):
        """Encode image and return only y_hat.
        This function is used if mc_sampling=True in Trainer
        and discriminator takes y_hat as a conditional input.
        """
        ## VQ Encode
        with torch.no_grad():
            gt_vq_latent, gt_vq_indices = self.vq_encode(real_images, vq_indices)
            y = self.comp_encode(
                real_images=real_images,
                gt_vq_latent=gt_vq_latent,
                gt_vq_indices=gt_vq_indices,
                enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
            )
            entropy_dict = self.estimate_entropy(y, is_train=False)
            y_hat = entropy_dict["quantized_code"]["y"]
        return y_hat

    def _compress_estimate_entropy(self, y: Tensor):
        z = self.hyperencoder(y)
        y = y.cpu()
        z = z.cpu()

        z_hat, z_likelihood = self.entropy_model_z(z, is_train=False)
        z_str = self.entropy_model_z.compress(z)

        hyper_out = self.hyperdecoder(z_hat)
        means_hat, scales_hat = hyper_out.chunk(2, 1)
        indexes = self.entropy_model_y.build_indexes(scales_hat)
        y_str = self.entropy_model_y.compress(y, indexes, means=means_hat)
        y_hat, y_likelihood = self.entropy_model_y(y, hyper_out, is_train=False)
        return {
            "y_hat": y_hat,
            "y_likelihood": y_likelihood,
            "y_str": y_str,
            "z_hat": z_hat,
            "z_likelihood": z_likelihood,
            "z_str": z_str,
        }

    @torch.no_grad()
    def compress(
        self,
        real_images: Tensor,
        quality_ind: Optional[int],
        vq_indices=None,
    ) -> Dict:
        beta_rate = self.selected_beta_rate[quality_ind]
        beta_vq = self.selected_beta_vq[quality_ind]

        N, _, H, W = real_images.shape
        assert N == 1, f"In compress mode, batch_size must be 1, but {N}"

        real_images = self.img_preprocess(real_images, is_train=False)

        if self.prior_type == "vqgan":
            gt_vq_latent, gt_vq_indices = self.vq_encode(real_images, vq_indices)
            y = self.comp_encode(
                real_images=real_images,
                gt_vq_latent=gt_vq_latent,
                gt_vq_indices=gt_vq_indices,
                enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
            )
        else:
            gt_prior_fused, _ = self.adacode_encode(real_images)
            y, _ = self.comp_encode_adacode(
                real_images=real_images,
                gt_prior_fused=gt_prior_fused,
                enc_kwargs=dict(beta_1=beta_rate, beta_2=beta_vq),
            )

        entropy_model_out = self._compress_estimate_entropy(y)
        y_hat = entropy_model_out["y_hat"]
        y_likelihood = entropy_model_out["y_likelihood"]
        z_likelihood = entropy_model_out["z_likelihood"]
        y_str = entropy_model_out["y_str"]
        z_str = entropy_model_out["z_str"]

        header_handler = HeaderHandler()
        header_str = header_handler.encode((H, W), y_hat, quality_ind)

        pred_y_bitcost, pred_y_bpp = self.likelihood_to_bit(y_likelihood, H * W)
        pred_z_bitcost, pred_z_bpp = self.likelihood_to_bit(z_likelihood, H * W)

        return {
            "string_list": [header_str, z_str[0], y_str[0]],
            "z_hat": entropy_model_out["z_hat"],
            "y_hat": y_hat,
            "z_likelihood": z_likelihood,
            "y_likelihood": y_likelihood,
            "pred_y_bit": pred_y_bitcost.item(),
            "pred_y_bpp": pred_y_bpp.item(),
            "pred_z_bit": pred_z_bitcost.item(),
            "pred_z_bpp": pred_z_bpp.item(),
        }

    def _decompress_estimate_entropy(self, z_str, y_str, zH: int, zW: int):
        z_symbol = self.entropy_model_z.decompress([z_str], (zH, zW))
        z_hat = self.entropy_model_z.dequantize(z_symbol)
        hyper_out = self.hyperdecoder(z_hat)

        means_hat, scales_hat = hyper_out.chunk(2, 1)
        indexes = self.entropy_model_y.build_indexes(scales_hat)
        y_hat = self.entropy_model_y.decompress([y_str], indexes, means=means_hat)

        return y_hat, z_hat

    @torch.no_grad()
    def decompress(self, string_list: List) -> Tuple[Tensor, Tensor, Tensor]:
        assert (
            len(string_list) == 3
        ), f"String list length should be 3 (header, z, and y),\
                                             but got {len(string_list)}"
        header_str = string_list[0]
        latent_z_str = string_list[1]
        latent_y_str = string_list[2]

        header_handler = HeaderHandler()
        header_dict = header_handler.decode(header_str)
        H, W = header_dict["img_size"]
        padH = int(np.ceil(H / self.model_stride)) * self.model_stride
        padW = int(np.ceil(W / self.model_stride)) * self.model_stride
        zH, zW = padH // self.model_stride, padW // self.model_stride

        quality_ind = header_dict["quality_ind"]
        beta_rate = self.selected_beta_rate[quality_ind]
        beta_vq = self.selected_beta_vq[quality_ind]

        beta_vq = torch.Tensor([beta_vq]).to(self.device)
        beta_rate = torch.Tensor([beta_rate]).to(self.device)

        y_hat, z_hat = self._decompress_estimate_entropy(
            latent_z_str, latent_y_str, zH, zW
        )

        w = 1.0
        from .hyperprior_vic_model import SPLIT_DECODE_RESOLUTION

        do_split_decode = max(H, W) > SPLIT_DECODE_RESOLUTION
        if do_split_decode:
            fake_images = self.decode_split(
                y_hat.to(self.device), w, beta_rate=beta_rate, beta_vq=beta_vq
            )
        elif self.prior_type == "vqgan":
            transformer_feat, cond_feat_dict = self.decoder.get_feats(
                y_hat.to(self.device), beta_1=beta_rate, beta_2=beta_vq
            )
            out_vq_latent, out_vq_logits = self.vq_estimator(transformer_feat)
            out_vq_indices = torch.argmax(out_vq_logits, dim=1)  # [N, H, W]

            vq_latent = self.vq_indices_to_latent(out_vq_indices)
            vq_latent = self.vq_model.post_quant_conv(vq_latent)

            fake_images = self.fusion_module(
                vq_latent, cond_feat_dict, self.vq_model.decoder, w=w
            )
        else:
            assert self.prior_estimator is not None
            assert self.adacode_adapter is not None
            transformer_feat, cond_feat_dict = self.decoder.get_feats(
                y_hat.to(self.device), beta_1=beta_rate, beta_2=beta_vq
            )
            pred_prior_fused = self.prior_estimator(transformer_feat)
            fake_images, _ = self.adacode_adapter.decode_from_fused_prior(
                pred_prior_fused, cond_feat_dict=cond_feat_dict, w=w
            )

        fake_images = self.img_postprocess(fake_images, size=(H, W), is_train=False)
        return fake_images, z_hat, y_hat

    def validation(
        self,
        dataloader,
        max_sample_size: int,
        fusion_w: Optional[float] = None,
        beta_rate: Optional[float] = None,
        beta_vq: Optional[float] = None,
    ) -> pd.DataFrame:
        score_list = []

        sample_size = min(len(dataloader), max_sample_size)

        beta_rate = beta_rate if beta_rate is not None else self.max_beta_rate / 2.0
        beta_vq = beta_vq if beta_vq is not None else self.max_beta_vq / 2.0

        for idx, data in enumerate(dataloader):
            model_input = {
                k: v for k, v in data.items() if k not in {"img_path", "disc_img_path"}
            }
            with torch.no_grad():
                out_dict = self.run_model(
                    **model_input,
                    beta_rate=beta_rate,
                    beta_vq=beta_vq,
                    is_train=False,
                    fusion_w=fusion_w,
                )

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
                score_dict["vq_mse"] = F.mse_loss(
                    out_dict["out_vq_latent"], out_dict["gt_vq_latent"]
                ).item()
            else:
                score_dict["vq_acc"] = out_dict["vq_accuracy"].item()
                if out_dict.get("pred_prior_fused") is not None and out_dict.get("gt_prior_fused") is not None:
                    score_dict["prior_mse"] = F.mse_loss(
                        out_dict["pred_prior_fused"], out_dict["gt_prior_fused"]
                    ).item()
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
