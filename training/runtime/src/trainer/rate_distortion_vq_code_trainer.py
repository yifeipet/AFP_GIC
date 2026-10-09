import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses import build_loss
from src.trainer.base_trainer import BaseTrainer
from src.utils.logger import IndentedLog, log_dict_items
from src.utils.path import PathHandler
from src.utils.registry import TRAINER_REGISTRY

from .optimizer import build_optimizer, build_scheduler


@dataclass
class ModelOutput:
    real_images: Tensor
    fake_images: Tensor
    bpp: Tensor
    qbpp: Tensor
    out_vq_latent: Tensor
    gt_vq_latent: Tensor
    out_vq_logits: Tensor
    gt_vq_indices: Tensor
    y_hat: Tensor
    vq_accuracy: Tensor
    pred_prior_fused: Optional[Tensor]
    gt_prior_fused: Optional[Tensor]
    pred_fake_images: Optional[Tensor]
    oracle_fake_images: Optional[Tensor]
    other_outputs: Dict

    def __init__(self, output_dict: Dict) -> None:
        pop_keys = [
            "real_images",
            "fake_images",
            "bpp",
            "qbpp",
            "out_vq_latent",
            "gt_vq_latent",
            "out_vq_logits",
            "gt_vq_indices",
            "y_hat",
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
class RateDistortionVqCodeTrainer(BaseTrainer):
    def __init__(
        self,
        opt,
    ) -> None:
        super().__init__(opt)

    def _set_optimizer_scheduler(self) -> None:
        if getattr(self.comp_model, "vq_model", None) is not None:
            self.comp_model.vq_model.requires_grad_(False)
        parameters_dict, aux_parameters_dict = self.comp_model.separete_aux_parameters()

        # set g_optimizer
        optim_opt = deepcopy(self.opt.optim)
        with IndentedLog(level="INFO", msg="building g_optimizer"):
            self.g_optimizer = build_optimizer(
                parameters_dict, optim_opt.g_optimizer
            )
            self.g_scheduler = build_scheduler(
                self.g_optimizer, optim_opt.g_scheduler
            )

        # set aux_optimizer
        if len(aux_parameters_dict) > 0:
            with IndentedLog(level="INFO", msg="building aux_optimizer"):
                self.aux_optimizer = build_optimizer(
                    aux_parameters_dict, optim_opt.aux_optimizer
                )
        else:
            self.logger.warn("aux_optimizer is NOT build.")
            self.aux_optimizer = None

        _clip_max_norm = self.opt.optim.get("clip_max_norm", None)
        log_dict_items({"clip_max_norm": _clip_max_norm}, level="INFO", indent=False)

    def _set_losses(self) -> None:
        loss_opt = deepcopy(self.opt.loss)

        self.distortion_loss = build_loss(
            loss_opt.distortion_loss, loss_name="distortion_loss"
        )
        self.rate_loss = build_loss(loss_opt.rate_loss, loss_name="rate_loss")
        self.perceptual_loss = build_loss(
            loss_opt.perceptual_loss, loss_name="perceptual_loss"
        ).to(self.device)

        ### VQ Code Loss
        self.code_distortion_loss = build_loss(
            loss_opt.code_distortion_loss, loss_name="code_distortion_loss"
        ).to(self.device)
        self.code_ce_loss = build_loss(
            loss_opt.code_ce_loss, loss_name="code_ce_loss"
        ).to(self.device)
        prior_mse_opt = loss_opt.get("prior_mse_loss", None)
        self.enable_prior_mse_loss = bool(
            prior_mse_opt is not None and prior_mse_opt.get("enabled", True)
        )
        self.prior_mse_loss = None
        if self.enable_prior_mse_loss and prior_mse_opt is not None:
            self.prior_mse_loss = build_loss(
                prior_mse_opt, loss_name="prior_mse_loss"
            ).to(self.device)
        prior_cosine_opt = loss_opt.get("prior_cosine_loss", None)
        self.enable_prior_cosine_loss = bool(
            prior_cosine_opt is not None and prior_cosine_opt.get("enabled", False)
        )
        self.prior_cosine_loss_weight = (
            float(prior_cosine_opt.get("loss_weight", 0.1))
            if prior_cosine_opt is not None
            else 0.0
        )
        self._init_group_ce(loss_opt)

    def _calc_prior_cosine_loss(
        self,
        pred_prior_fused: Tensor,
        gt_prior_fused: Tensor,
    ) -> Tensor:
        pred_norm = F.normalize(pred_prior_fused, dim=1, eps=1e-8)
        gt_norm = F.normalize(gt_prior_fused.detach(), dim=1, eps=1e-8)
        cosine_map = torch.sum(pred_norm * gt_norm, dim=1)
        return 1.0 - cosine_map

    def _resolve_group_map_path(self, group_map_path: str) -> Optional[str]:
        if not group_map_path:
            return None
        expanded = os.path.expanduser(group_map_path)
        if os.path.isabs(expanded) and os.path.exists(expanded):
            return expanded

        candidates = [os.path.abspath(expanded)]
        cfg_path = getattr(self.opt, "filename", None)
        if cfg_path:
            cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
            candidates.append(os.path.abspath(os.path.join(cfg_dir, expanded)))

        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _init_group_ce(self, loss_opt) -> None:
        self.enable_group_ce = bool(loss_opt.get("enable_group_ce", False))
        self.group_k = int(loss_opt.get("group_k", 4))
        self.lambda_group_ce = float(loss_opt.get("lambda_group_ce", 0.05))
        self.group_map_path = loss_opt.get("group_map_path", None)
        self._group_ce_enabled = False

        if not self.enable_group_ce:
            self.logger.info("group-ce is disabled.")
            return

        if not self.group_map_path:
            self.logger.warning(
                "enable_group_ce=True but group_map_path is empty. group-ce is disabled."
            )
            return

        resolved_map_path = self._resolve_group_map_path(self.group_map_path)
        if resolved_map_path is None:
            self.logger.warning(
                f'group_map not found: "{self.group_map_path}". group-ce is disabled.'
            )
            return

        group_map = np.load(resolved_map_path)
        if group_map.ndim != 1:
            raise ValueError(
                f"group_map must be 1D, but got shape={group_map.shape}"
            )

        gid = torch.from_numpy(group_map).long().to(self.device)
        n_embed = int(getattr(self.comp_model, "n_embed", gid.numel()))
        if gid.numel() != n_embed:
            raise ValueError(
                f"group_map length mismatch: len(gid)={gid.numel()} vs n_embed={n_embed}"
            )

        if gid.min().item() < 0 or gid.max().item() >= self.group_k:
            raise ValueError(
                f"group_map values must be in [0, {self.group_k - 1}], "
                f"but got min={gid.min().item()}, max={gid.max().item()}"
            )

        if "gid" in self.comp_model._buffers:
            self.comp_model._buffers["gid"] = gid
        else:
            self.comp_model.register_buffer("gid", gid, persistent=True)

        self._group_ce_enabled = True
        counts = torch.bincount(gid, minlength=self.group_k).cpu().tolist()
        self.logger.info(
            f'group-ce enabled: k={self.group_k}, lambda={self.lambda_group_ce}, '
            f'map="{resolved_map_path}", cluster_sizes={counts}'
        )

    def _calc_group_ce_nll(
        self,
        out_vq_logits: Optional[Tensor],
        gt_vq_indices: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if not self._group_ce_enabled:
            return None, None
        if out_vq_logits is None:
            return None, None

        gid: Tensor = self.comp_model.gid  # [C]
        p_tok = torch.softmax(out_vq_logits, dim=1)  # [B, C, H, W]
        _, c, h, w = p_tok.shape
        if gid.numel() != c:
            raise ValueError(
                f"gid size mismatch: gid={gid.numel()} vs logits channels={c}"
            )

        p_group = torch.zeros(
            (p_tok.size(0), self.group_k, h, w),
            dtype=p_tok.dtype,
            device=p_tok.device,
        )
        for k in range(self.group_k):
            mask_k = gid == k
            if mask_k.any():
                p_group[:, k] = p_tok[:, mask_k].sum(dim=1)

        log_p_group = torch.log(p_group.clamp_min(1e-9))
        k_gt = gid[gt_vq_indices.long()]
        group_nll = F.nll_loss(log_p_group, k_gt, reduction="none")  # [B, H, W]
        group_prob_l1 = (p_group.sum(dim=1) - 1.0).abs().mean().detach()
        return group_nll, group_prob_l1

    def run_comp_model(self, data_dict: Dict) -> ModelOutput:
        model_input = {
            k: v for k, v in data_dict.items() if k not in {"img_path", "disc_img_path"}
        }
        out_dict = self.comp_model.run_model(
            **model_input,
            is_train=True,
            fix_entropy_models=False,
            run_vq_decoder=True,
        )
        return ModelOutput(out_dict)

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
        outputs = self.run_comp_model(data_dict)

        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        # calculate losses
        g_loss_dict = {}
        g_metric_dict = {}

        g_loss_dict["rate"] = self.rate_loss(outputs.bpp)
        ## Image Loss
        g_loss_dict["distortion"] = self.distortion_loss(
            outputs.real_images, outputs.fake_images
        )
        g_loss_dict["perceptual"] = self.perceptual_loss(
            outputs.real_images, outputs.fake_images
        )

        if getattr(self.comp_model, "prior_type", "vqgan") == "adacode":
            if outputs.pred_prior_fused is None or outputs.gt_prior_fused is None:
                raise ValueError("AdaCode mode requires pred_prior_fused and gt_prior_fused.")
            if self.enable_prior_mse_loss:
                if self.prior_mse_loss is not None:
                    g_loss_dict["prior_mse"] = self.prior_mse_loss(
                        outputs.gt_prior_fused.detach(), outputs.pred_prior_fused
                    )
                else:
                    g_loss_dict["prior_mse"] = F.mse_loss(
                        outputs.pred_prior_fused, outputs.gt_prior_fused.detach()
                    )
            if self.enable_prior_cosine_loss:
                g_loss_dict["prior_cosine"] = (
                    self._calc_prior_cosine_loss(
                        outputs.pred_prior_fused, outputs.gt_prior_fused
                    ).mean()
                    * self.prior_cosine_loss_weight
                )
        else:
            g_loss_dict["code_distortion"] = self.code_distortion_loss(
                outputs.gt_vq_latent, outputs.out_vq_latent
            )
            g_loss_dict["code_ce"] = self.code_ce_loss(
                outputs.out_vq_logits, outputs.gt_vq_indices
            )
            group_nll, group_prob_l1 = self._calc_group_ce_nll(
                outputs.out_vq_logits, outputs.gt_vq_indices
            )
            if group_nll is not None:
                g_loss_dict["group_ce"] = self.lambda_group_ce * group_nll.mean()
                if group_prob_l1 is not None:
                    g_metric_dict["group_prob_l1"] = group_prob_l1

        l_total: Tensor = sum(_v for _v in g_loss_dict.values())  # type: ignore

        # For stability
        if loss_anomaly := self.check_loss_nan_inf(l_total):
            self.log_loss_anomaly(
                current_iter,
                loss_anomaly,
                l_total,
                log_dict={**log_dict, **g_loss_dict, **g_metric_dict},
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

        log_dict.update(g_loss_dict)
        log_dict.update(g_metric_dict)
        log_dict["lr"] = self.g_optimizer.param_groups[0]["lr"]

        self.g_scheduler.step()

        if self.aux_optimizer:
            log_dict["aux"] = self.optimize_aux_parameters()

        return log_dict
    
    def _validation(self) -> Dict:
        eval_df = self.comp_model.validation(self.eval_loader, max_sample_size=100)
        eval_dict = eval_df.drop('idx', axis=1).mean().to_dict()
        return eval_dict

    def optimize_aux_parameters(self):
        if self.aux_optimizer is None:
            return
        aux_loss = self.comp_model.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()
        return aux_loss

    def save(self, current_iter: int):
        # save model
        self.model_saver.save(
            {"comp_model": self.comp_model},
            "comp_model",
            current_iter,
            keep=True,
        )
        # save training_state
        optimizer_scheduler_dict = {"g_optimizer": self.g_optimizer}
        optimizer_scheduler_dict["g_scheduler"] = self.g_scheduler
        if self.aux_optimizer:
            optimizer_scheduler_dict["aux_optimizer"] = self.aux_optimizer
        self.model_saver.save(
            optimizer_scheduler_dict,
            "training_state",
            current_iter,
            keep=self.opt.get("keep_training_state", False),
        )

    def _get_best_checkpoint_payloads(self) -> Dict[str, Dict]:
        payloads = super()._get_best_checkpoint_payloads()
        best_cfg = self.opt.get("best_checkpoint", {})
        if best_cfg.get("save_training_state", False):
            optimizer_scheduler_dict = {"g_optimizer": self.g_optimizer}
            optimizer_scheduler_dict["g_scheduler"] = self.g_scheduler
            if self.aux_optimizer:
                optimizer_scheduler_dict["aux_optimizer"] = self.aux_optimizer
            payloads["training_state"] = optimizer_scheduler_dict
        return payloads

    def _load_checkpoint(
        self,
        exp_name: str,
        itr: int,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        new_g_lr: Optional[float] = None,
        strict: bool = True,
        **kwargs,
    ) -> None:
        ## get checkpoint path
        path_handler = PathHandler(self.opt.path.ckpt_root, exp_name)
        exp_path_dict = path_handler.get_exp_path_dict()
        model_ckpt_path = path_handler.get_ckpt_path("comp_model", itr)
        assert os.path.exists(model_ckpt_path)

        if load_optimizer:
            optim_ckpt_path = path_handler.get_ckpt_path("training_state", itr)
            assert os.path.exists(optim_ckpt_path)
        else:
            optim_ckpt_path = None
            self.logger.warn("optimizer is not loaded")

        log_dict_items(
            {
                "model_ckpt_path": model_ckpt_path,
                "optim_ckpt_path": optim_ckpt_path,
            },
            level="INFO",
            indent=False,
        )

        model_ckpt = torch.load(model_ckpt_path, map_location=self.device)
        out = self.comp_model.load_state_dict(
            model_ckpt["comp_model"], strict=strict
        )
        self.logger.debug(f'comp_model.load_state_dict: "{out}"')

        if not load_optimizer:
            return

        training_state_ckpt = torch.load(
            optim_ckpt_path, map_location=self.device
        )
        self.g_optimizer.load_state_dict(training_state_ckpt["g_optimizer"])
        self.logger.debug(f"load checkpoint: g_optimizer")

        if new_g_lr is not None:
            self.update_learning_rate(self.g_optimizer, new_g_lr)
            self.logger.info(f"g_optimizer: lr is changed to {new_g_lr}")

        if load_scheduler:
            self.g_scheduler.load_state_dict(training_state_ckpt["g_scheduler"])
            self.logger.debug(f"load checkpoint: g_scheduler")
        else:
            self.logger.warn("g_scheduler is not loaded")

        if self.aux_optimizer:
            self.aux_optimizer.load_state_dict(
                training_state_ckpt["aux_optimizer"]
            )
            self.logger.debug(f"load checkpoint: aux_optimizer")
