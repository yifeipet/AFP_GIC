from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.registry import TRAINER_REGISTRY

from .dual_cond_gan_distortion_vq_code_trainer import (
    DualBetaCondGanDistortionVqCodeTrainer,
)
from .dual_cond_rate_disotrion_vq_code_trainer import (
    DualBetaCondRateDistortionVqCodeTrainer,
)
from .rate_distortion_vq_code_trainer import RateDistortionVqCodeTrainer


class _AdaCodeScheme1TrainerMixin:
    def _init_scheme1_loss_cfg(self) -> None:
        loss_opt = deepcopy(self.opt.loss)

        branch_opt = loss_opt.get("prior_branch_loss", None)
        self.enable_prior_branch_loss = bool(
            branch_opt is not None and branch_opt.get("enabled", True)
        )
        self.prior_branch_loss_weight = (
            float(branch_opt.get("loss_weight", 0.0)) if branch_opt is not None else 0.0
        )

        weight_opt = loss_opt.get("prior_weight_loss", None)
        self.enable_prior_weight_loss = bool(
            weight_opt is not None and weight_opt.get("enabled", True)
        )
        self.prior_weight_loss_weight = (
            float(weight_opt.get("loss_weight", 0.0)) if weight_opt is not None else 0.0
        )

    @staticmethod
    def _get_scheme1_tensor(outputs, key: str) -> Optional[Tensor]:
        return outputs.other_outputs.get(key, None)

    def _calc_branch_loss(
        self,
        pred_branch_feats: Optional[Tensor],
        gt_branch_feats: Optional[Tensor],
    ) -> Optional[Tensor]:
        if (
            not self.enable_prior_branch_loss
            or pred_branch_feats is None
            or gt_branch_feats is None
        ):
            return None
        return self.prior_branch_loss_weight * torch.square(
            pred_branch_feats - gt_branch_feats.detach()
        )

    def _calc_weight_loss(
        self,
        pred_weight_map: Optional[Tensor],
        gt_weight_map: Optional[Tensor],
    ) -> Optional[Tensor]:
        if (
            not self.enable_prior_weight_loss
            or pred_weight_map is None
            or gt_weight_map is None
        ):
            return None
        return self.prior_weight_loss_weight * torch.square(
            pred_weight_map - gt_weight_map.detach()
        )


@TRAINER_REGISTRY.register()
class AdaCodeScheme1RateDistortionVqCodeTrainer(
    _AdaCodeScheme1TrainerMixin,
    RateDistortionVqCodeTrainer,
):
    def _set_losses(self) -> None:
        super()._set_losses()
        self._init_scheme1_loss_cfg()

    def optimize_parameters(
        self,
        current_iter: int,
        data_dict: dict,
    ) -> Union[dict, None]:
        log_dict = {}

        self.g_optimizer.zero_grad()
        if self.aux_optimizer:
            self.aux_optimizer.zero_grad()

        outputs = self.run_comp_model(data_dict)

        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        g_loss_dict = {}
        g_metric_dict = {}

        g_loss_dict["rate"] = self.rate_loss(outputs.bpp)
        g_loss_dict["distortion"] = self.distortion_loss(
            outputs.real_images,
            outputs.fake_images,
        )
        g_loss_dict["perceptual"] = self.perceptual_loss(
            outputs.real_images,
            outputs.fake_images,
        )

        if getattr(self.comp_model, "prior_type", "vqgan") != "adacode":
            raise ValueError("AdaCode Scheme 1 trainer expects prior_type=adacode.")

        if outputs.pred_prior_fused is None or outputs.gt_prior_fused is None:
            raise ValueError("AdaCode mode requires pred_prior_fused and gt_prior_fused.")

        if self.enable_prior_mse_loss:
            if self.prior_mse_loss is not None:
                g_loss_dict["prior_mse"] = self.prior_mse_loss(
                    outputs.gt_prior_fused.detach(),
                    outputs.pred_prior_fused,
                )
            else:
                g_loss_dict["prior_mse"] = F.mse_loss(
                    outputs.pred_prior_fused,
                    outputs.gt_prior_fused.detach(),
                )

        if self.enable_prior_cosine_loss:
            g_loss_dict["prior_cosine"] = (
                self._calc_prior_cosine_loss(
                    outputs.pred_prior_fused,
                    outputs.gt_prior_fused,
                ).mean()
                * self.prior_cosine_loss_weight
            )

        branch_loss = self._calc_branch_loss(
            self._get_scheme1_tensor(outputs, "pred_branch_feats"),
            self._get_scheme1_tensor(outputs, "gt_branch_feats"),
        )
        if branch_loss is not None:
            g_loss_dict["prior_branch_mse"] = branch_loss.mean()

        weight_loss = self._calc_weight_loss(
            self._get_scheme1_tensor(outputs, "pred_weight_map"),
            self._get_scheme1_tensor(outputs, "gt_weight_map"),
        )
        if weight_loss is not None:
            g_loss_dict["prior_weight_mse"] = weight_loss.mean()

        l_total: Tensor = sum(_v for _v in g_loss_dict.values())  # type: ignore

        if loss_anomaly := self.check_loss_nan_inf(l_total):
            self.log_loss_anomaly(
                current_iter,
                loss_anomaly,
                l_total,
                log_dict={**log_dict, **g_loss_dict, **g_metric_dict},
                data_dict=data_dict,
            )
            return None

        l_total.backward()
        if self.opt.optim.get("clip_max_norm", None):
            nn.utils.clip_grad_norm_(
                self.comp_model.parameters(),
                self.opt.optim.clip_max_norm,
            )

        self.g_optimizer.step()

        log_dict.update(g_loss_dict)
        log_dict.update(g_metric_dict)
        log_dict["lr"] = self.g_optimizer.param_groups[0]["lr"]
        self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict["aux"] = self.optimize_aux_parameters()

        return log_dict


@TRAINER_REGISTRY.register()
class AdaCodeScheme1DualBetaCondRateDistortionVqCodeTrainer(
    _AdaCodeScheme1TrainerMixin,
    DualBetaCondRateDistortionVqCodeTrainer,
):
    def _set_losses(self) -> None:
        super()._set_losses()
        self._init_scheme1_loss_cfg()

    def calc_g_loss(
        self,
        outputs,
        current_iter: int,
        calc_img_loss: bool = True,
    ) -> Tuple[Tensor, Dict]:
        l_total, log_dict = super().calc_g_loss(
            outputs,
            current_iter,
            calc_img_loss=calc_img_loss,
        )

        vq_weight, _ = self.calc_vq_rate_loss_weight(outputs.beta_vq, outputs.beta_rate)

        branch_loss = self._calc_branch_loss(
            self._get_scheme1_tensor(outputs, "pred_branch_feats"),
            self._get_scheme1_tensor(outputs, "gt_branch_feats"),
        )
        if branch_loss is not None:
            branch_loss = self.apply_loss_weight(branch_loss, vq_weight)
            log_dict["prior_branch_mse"] = branch_loss
            l_total += branch_loss

        weight_loss = self._calc_weight_loss(
            self._get_scheme1_tensor(outputs, "pred_weight_map"),
            self._get_scheme1_tensor(outputs, "gt_weight_map"),
        )
        if weight_loss is not None:
            weight_loss = self.apply_loss_weight(weight_loss, vq_weight)
            log_dict["prior_weight_mse"] = weight_loss
            l_total += weight_loss

        return l_total, log_dict


@TRAINER_REGISTRY.register()
class AdaCodeScheme1DualBetaCondGanDistortionVqCodeTrainer(
    _AdaCodeScheme1TrainerMixin,
    DualBetaCondGanDistortionVqCodeTrainer,
):
    def _set_losses(self) -> None:
        super()._set_losses()
        self._init_scheme1_loss_cfg()

    def calc_g_loss(
        self,
        outputs,
    ) -> Tuple[Tensor, Dict]:
        loss_total, log_dict = super().calc_g_loss(outputs)

        branch_loss = self._calc_branch_loss(
            self._get_scheme1_tensor(outputs, "pred_branch_feats"),
            self._get_scheme1_tensor(outputs, "gt_branch_feats"),
        )
        if branch_loss is not None:
            branch_loss = branch_loss.mean()
            log_dict["prior_branch_mse"] = branch_loss
            loss_total += branch_loss

        weight_loss = self._calc_weight_loss(
            self._get_scheme1_tensor(outputs, "pred_weight_map"),
            self._get_scheme1_tensor(outputs, "gt_weight_map"),
        )
        if weight_loss is not None:
            weight_loss = weight_loss.mean()
            log_dict["prior_weight_mse"] = weight_loss
            loss_total += weight_loss

        return loss_total, log_dict
