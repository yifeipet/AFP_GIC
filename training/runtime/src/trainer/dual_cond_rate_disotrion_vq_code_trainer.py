from typing import Dict, Tuple, Optional, Union, List

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor

from src.utils.registry import TRAINER_REGISTRY

from .rate_distortion_vq_code_trainer import RateDistortionVqCodeTrainer


@dataclass
class DualBetaModelOutput:
    real_images: Tensor
    fake_images: Tensor
    bpp: Tensor
    qbpp: Tensor
    out_vq_latent: Tensor
    gt_vq_latent: Tensor
    out_vq_logits: Tensor
    gt_vq_indices: Tensor
    beta_rate: Tensor
    beta_vq: Tensor
    y_hat: Tensor
    vq_accuracy: Tensor
    pred_prior_fused: Optional[Tensor]
    gt_prior_fused: Optional[Tensor]
    pred_fake_images: Optional[Tensor]
    oracle_fake_images: Optional[Tensor]
    other_outputs: Dict

    def __init__(self, output_dict: Dict) -> None:
        pop_keys = [
            'real_images',
            'fake_images',
            'bpp',
            'qbpp',
            'out_vq_latent',
            'gt_vq_latent',
            'out_vq_logits',
            'gt_vq_indices',
            'beta_rate',
            'beta_vq',
            'y_hat',
            "vq_accuracy",
        ]
        for k in pop_keys:
            v = output_dict.pop(k)
            setattr(self, k, v)

        optional_keys = [
            "pred_prior_fused",
            "gt_prior_fused",
            "pred_fake_images",
            "oracle_fake_images",
        ]
        for k in optional_keys:
            setattr(self, k, output_dict.pop(k, None))

        self.other_outputs = output_dict



@TRAINER_REGISTRY.register()
class DualBetaCondRateDistortionVqCodeTrainer(RateDistortionVqCodeTrainer):
    def __init__(
        self,
        opt,
        beta_policy: str = "linear",
        beta_offset: float = 1.0,
        sample_beta_batch: bool = False,
    ) -> None:
        super().__init__(opt)
        assert beta_policy in ["linear", "exp"]
        self.beta_policy = beta_policy
        self.beta_offset = beta_offset
        self.sample_beta_batch = sample_beta_batch

    def calc_vq_rate_loss_weight(self, beta_vq: Tensor, beta_rate: Tensor):
        beta_vq = beta_vq.to(self.device)
        beta_rate = beta_rate.to(self.device)
        if self.beta_policy == "linear":
            return beta_vq + self.beta_offset, beta_rate + self.beta_offset
        if self.beta_policy == "exp":
            return torch.exp(beta_vq), torch.exp(beta_rate)
        raise ValueError()

    def run_comp_model(
        self,
        data_dict: Dict,
    ) -> DualBetaModelOutput:
        model_input = {
            k: v for k, v in data_dict.items() if k not in {"img_path", "disc_img_path"}
        }
        out_dict = self.comp_model.run_model(
            **model_input,
            is_train=True,
            fix_entropy_models=False,
            sample_batch_beta=self.sample_beta_batch,
        )
        return DualBetaModelOutput(out_dict)

    def apply_loss_weight(self, loss: Tensor, weight: Tensor):
        assert weight.ndim == 1
        # different weight for each sample in batch
        if loss.ndim > 1:
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        loss = (loss * weight).mean()
        return loss

    def _calc_batch_bpp(
        self, y_likelihood: Tensor, z_likelihood: Tensor, num_pixel: int
    ) -> Tensor:
        bit_y = -torch.log(y_likelihood) / np.log(2)  # [N, C, H, W]
        bit_z = -torch.log(z_likelihood) / np.log(2)  # [N, C, H, W]
        bit_y = torch.sum(bit_y, dim=tuple(range(1, bit_y.ndim)))  # [N]
        bit_z = torch.sum(bit_z, dim=tuple(range(1, bit_z.ndim)))  # [N]
        bpp = (bit_y + bit_z) / num_pixel  # [N]
        return bpp
    
    def optimize_parameters(
        self, current_iter: int, data_dict: dict
    ) -> Union[dict, None]:
        log_dict = {}

        ###################################################################
        #                             Train G
        ###################################################################
        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        # run model
        model_outputs = self.run_comp_model(data_dict)
        l_total, log_dict = self.calc_g_loss(
            model_outputs,
            current_iter,
            calc_img_loss=True,
        )

        # For stability
        if loss_anomaly := self.check_loss_nan_inf(l_total):
            self.log_loss_anomaly(
                current_iter,
                loss_anomaly,
                l_total,
                log_dict=log_dict,
                data_dict=data_dict,
            )
            return None  # skip back-propagation part

        # back prop & update parameters
        l_total.backward()
        if self.opt.optim.get("clip_max_norm", None):
            nn.utils.clip_grad_norm_(
                self.comp_model.parameters(), self.opt.optim.clip_max_norm
            )

        self.g_optimizer.step()
        log_dict["lr"] = self.g_optimizer.param_groups[0]["lr"]
        self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict["aux"] = self.optimize_aux_parameters()

        return log_dict

    def calc_g_loss(
        self, outputs: DualBetaModelOutput, current_iter: int, calc_img_loss: bool = True
    ) -> Tuple[Tensor, Dict]:
        other_outputs = outputs.other_outputs

        vq_weight, rate_weight = self.calc_vq_rate_loss_weight(
            outputs.beta_vq, outputs.beta_rate
        )

        log_dict = {}
        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        # calculate losses
        if self.sample_beta_batch:
            num_pixel = outputs.real_images.size(2) * outputs.real_images.size(3)
            batch_bpp = self._calc_batch_bpp(
                other_outputs["y_likelihood"], other_outputs["z_likelihood"], num_pixel
            )
            rate = self.rate_loss(batch_bpp)
        else:
            rate = self.rate_loss(outputs.bpp)
        rate = self.apply_loss_weight(rate, rate_weight)
        log_dict["rate"] = rate
        l_total = rate

        ## Image Loss
        dist_weight = 1 if calc_img_loss else 0
        distortion = dist_weight * self.distortion_loss(outputs.real_images, outputs.fake_images)
        log_dict["distortion"] = distortion
        l_total += distortion

        perceptual = dist_weight * self.perceptual_loss(outputs.real_images, outputs.fake_images)
        log_dict["perceptual"] = perceptual
        l_total += perceptual

        ## VQ Code Loss
        if getattr(self.comp_model, "prior_type", "vqgan") == "adacode":
            if outputs.pred_prior_fused is None or outputs.gt_prior_fused is None:
                raise ValueError("AdaCode mode requires pred_prior_fused and gt_prior_fused.")
            if self.enable_prior_mse_loss:
                if self.prior_mse_loss is not None:
                    prior_mse = self.prior_mse_loss(
                        outputs.gt_prior_fused.detach(), outputs.pred_prior_fused
                    )
                else:
                    prior_mse = torch.square(
                        outputs.pred_prior_fused - outputs.gt_prior_fused.detach()
                    )
                prior_mse = self.apply_loss_weight(prior_mse, vq_weight)
                log_dict["prior_mse"] = prior_mse
                l_total += prior_mse
            if self.enable_prior_cosine_loss:
                prior_cosine = self._calc_prior_cosine_loss(
                    outputs.pred_prior_fused, outputs.gt_prior_fused
                )
                prior_cosine = self.apply_loss_weight(prior_cosine, vq_weight)
                prior_cosine = self.prior_cosine_loss_weight * prior_cosine
                log_dict["prior_cosine"] = prior_cosine
                l_total += prior_cosine
        else:
            code_dist = self.code_distortion_loss(outputs.gt_vq_latent, outputs.out_vq_latent)
            code_dist = self.apply_loss_weight(code_dist, vq_weight)
            log_dict["code_distortion"] = code_dist
            l_total += code_dist

            code_ce = self.code_ce_loss(outputs.out_vq_logits, outputs.gt_vq_indices)
            code_ce = self.apply_loss_weight(code_ce, vq_weight)
            log_dict["code_ce"] = code_ce
            l_total += code_ce

            group_nll, group_prob_l1 = self._calc_group_ce_nll(
                outputs.out_vq_logits, outputs.gt_vq_indices
            )
            if group_nll is not None:
                group_ce = self.apply_loss_weight(group_nll, vq_weight)
                group_ce = self.lambda_group_ce * group_ce
                log_dict["group_ce"] = group_ce
                l_total += group_ce
                if group_prob_l1 is not None:
                    log_dict["group_prob_l1"] = group_prob_l1

        return l_total, log_dict

    def _validation(self) -> Dict:
        max_beta_vq: float = self.comp_model.max_beta_vq  # type: ignore
        max_beta_rate: float = self.comp_model.max_beta_rate  # type: ignore

        if getattr(self.comp_model, "use_selected_beta_pairs", False):
            beta_rate_list: List = self.comp_model.selected_beta_rate  # type: ignore
            beta_vq_list: List = self.comp_model.selected_beta_vq  # type: ignore
            beta_rate_vq_label = [
                (beta_rate_list[0], beta_vq_list[0], "idx0"),
            ]
        else:
            beta_rate_vq_label = [
                (max_beta_rate, 0, "rate_max_vq_000"),
                (max_beta_rate, max_beta_vq, "rate_max_vq_max"),
                (0, 0, "rate_000_vq_000"),
                (0, max_beta_vq, "rate_000_vq_max"),
            ]

        eval_dict = {}

        for beta_rate, beta_vq, label in beta_rate_vq_label:
            eval_df = self.comp_model.validation(
                self.eval_loader,
                max_sample_size=100,
                beta_rate=beta_rate,
                beta_vq=beta_vq,
            )
            _eval_dict = eval_df.drop("idx", axis=1).mean().to_dict()
            _eval_dict = {f"{label}_{k}": v for k, v in _eval_dict.items()}
            eval_dict.update(_eval_dict)

        return eval_dict
