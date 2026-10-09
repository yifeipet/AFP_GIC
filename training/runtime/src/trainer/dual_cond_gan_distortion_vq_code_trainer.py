from typing import Dict, Tuple, Optional, Union, List

from copy import deepcopy

import torch
import torch.nn as nn

from torch import Tensor

from src.losses import build_loss
from src.utils.logger import IndentedLog, log_dict_items
from src.utils.registry import TRAINER_REGISTRY
from src.models.discriminator import build_discriminator
from src.utils.path import PathHandler

from .optimizer import build_optimizer, build_scheduler

from .dual_cond_rate_disotrion_vq_code_trainer import (
    DualBetaModelOutput,
    DualBetaCondRateDistortionVqCodeTrainer,
)


@TRAINER_REGISTRY.register()
class DualBetaCondGanDistortionVqCodeTrainer(DualBetaCondRateDistortionVqCodeTrainer):
    def __init__(
        self,
        opt,
        beta_policy: str = "linear",
        beta_offset: float = 1.0,
        sample_beta_batch: bool = False,
        mc_sampling: bool = False,
    ) -> None:
        super().__init__(
            opt,
            beta_policy=beta_policy,
            beta_offset=beta_offset,
            sample_beta_batch=sample_beta_batch,
        )
        self.mc_sampling = mc_sampling

    def _set_models(self) -> None:
        super()._set_models()
        self.discriminator = build_discriminator(self.opt.discriminator).to(self.device)
        self.discriminator.train()

    def _set_optimizer_scheduler(self) -> None:
        self.comp_model.requires_grad_(False)
        self.comp_model.decoder.requires_grad_(True)
        if getattr(self.comp_model, "prior_type", "vqgan") == "adacode":
            if getattr(self.comp_model, "prior_estimator", None) is not None:
                self.comp_model.prior_estimator.requires_grad_(True)
            if getattr(self.comp_model, "adacode_adapter", None) is not None:
                self.comp_model.adacode_adapter.fusion_wrapper.requires_grad_(True)
            self.logger.warn(
                "Only decoder, prior_estimator, adacode fusion_wrapper will be trained!"
            )
        else:
            self.comp_model.vq_estimator.requires_grad_(True)
            self.comp_model.fusion_module.requires_grad_(True)
            self.logger.warn("Only decoder, vq_estimator, fusion_module will be trained!")

        # Set g_optimizer and g_scheduler
        super()._set_optimizer_scheduler()

        # Set d_optimizer and d_scheduler
        optim_opt = deepcopy(self.opt.optim)
        with IndentedLog(level="INFO", msg="building d_optimizer"):
            self.d_optimizer = build_optimizer(
                {k: v for k, v in self.discriminator.named_parameters()},
                optim_opt.d_optimizer,
            )
            self.d_scheduler = build_scheduler(
                self.d_optimizer, optim_opt.d_scheduler
            )

    def _set_losses(self) -> None:
        loss_opt = deepcopy(self.opt.loss)

        self.distortion_loss = build_loss(
            loss_opt.distortion_loss, loss_name="distortion_loss"
        ).to(self.device)
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

        ### GAN Loss
        self.gan_loss = build_loss(loss_opt.gan_loss, loss_name="gan_loss").to(
            self.device
        )

    def _divide_samples_for_mc_sampling(self, data_dict: Dict) -> Tuple[Dict, Dict]:
        """Divide real_images into two groups for G and D, respectively

        NOTE:
        for now, pre-processing for D's data is not implemented except `.to(device)`
        if specific pre-processing is needed, override this method
        """
        disc_data_dict = {}
        real_images = data_dict["real_images"]
        batch_size = real_images.size(0)
        assert batch_size % 2 == 0
        data_dict["real_images"] = real_images[: batch_size // 2]
        disc_data_dict["disc_real_images"] = real_images[batch_size // 2 :].to(
            self.device
        )

        assert "vq_indices" in data_dict
        vq_indices = data_dict["vq_indices"]
        data_dict["vq_indices"] = vq_indices[: batch_size // 2]
        disc_data_dict["disc_vq_indices"] = vq_indices[batch_size // 2 :].to(
            self.device
        )
        if "img_path" in data_dict:
            img_paths = data_dict["img_path"]
            if isinstance(img_paths, (list, tuple)):
                data_dict["img_path"] = list(img_paths[: batch_size // 2])
                disc_data_dict["disc_img_path"] = list(img_paths[batch_size // 2 :])
        # print('mc sampling', real_images.shape, data_dict["real_images"].shape, disc_data_dict["disc_vq_indices"].shape)
        return data_dict, disc_data_dict

    def run_comp_model(
        self,
        data_dict: Dict,
    ) -> DualBetaModelOutput:
        self.comp_model: nn.Module
        disc_data_dict = {}
        if self.mc_sampling:
            # Divide input datas into two groups for G and D, respectively
            data_dict, disc_data_dict = self._divide_samples_for_mc_sampling(data_dict)

        model_input = {
            k: v for k, v in data_dict.items() if k not in {"img_path", "disc_img_path"}
        }
        out_dict = self.comp_model.run_model(
            **model_input,
            is_train=True,
            fix_entropy_models=True,
            sample_batch_beta=self.sample_beta_batch,
        )
        out_dict.update(disc_data_dict)
        return DualBetaModelOutput(out_dict)

    def optimize_parameters(
        self, current_iter: int, data_dict: dict
    ) -> Union[dict, None]:
        log_dict = {}

        ###################################################################
        #                             Train G
        ###################################################################
        self.g_optimizer.zero_grad()
        self.discriminator.requires_grad_(False)

        # run model
        model_outputs = self.run_comp_model(data_dict)

        l_total, log_dict = self.calc_g_loss(model_outputs)

        # For stability
        loss_anomaly = self.check_loss_nan_inf(l_total)
        if loss_anomaly:
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

        ###################################################################
        #                             Train D
        ###################################################################

        self.discriminator.requires_grad_(True)
        self.d_optimizer.zero_grad()
        d_real, d_fake = self.run_discriminator(model_outputs)

        d_loss, d_log_dict = self.calc_d_loss(
            d_real,
            d_fake,
            # for OASIS GAN loss
            gt_vq_indices=model_outputs.gt_vq_indices,
            disc_gt_vq_indices=model_outputs.other_outputs.get("disc_vq_indices", None),
        )
        d_loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()
        log_dict.update(d_log_dict)

        return log_dict
    
    def calc_g_loss(
        self,
        outputs: DualBetaModelOutput,
    ) -> Tuple[Tensor, Dict]:
        other_outputs = outputs.other_outputs

        log_dict = {}
        log_dict["qbpp"] = outputs.qbpp
        log_dict["vq_acc"] = outputs.vq_accuracy

        ## Image Loss
        distortion = self.distortion_loss(outputs.real_images, outputs.fake_images)
        log_dict["distortion"] = distortion
        loss_total = distortion

        perceptual = self.perceptual_loss(outputs.real_images, outputs.fake_images)
        log_dict["perceptual"] = perceptual
        loss_total += perceptual

        ## Adv Loss
        g_fake = self.discriminator(
            outputs.fake_images,
            beta_1=outputs.beta_rate,
            beta_2=outputs.beta_vq,
            y_hat=outputs.y_hat,
            **other_outputs,
        )
        adv_loss = self.gan_loss(g_fake, is_real=True, is_disc=False)
        log_dict["adv"] = adv_loss
        loss_total += adv_loss

        ## Prior / Code Loss
        if getattr(self.comp_model, "prior_type", "vqgan") == "adacode":
            if outputs.pred_prior_fused is None or outputs.gt_prior_fused is None:
                raise ValueError(
                    "AdaCode mode requires pred_prior_fused and gt_prior_fused."
                )
            if self.enable_prior_mse_loss:
                if self.prior_mse_loss is not None:
                    prior_mse = self.prior_mse_loss(
                        outputs.gt_prior_fused.detach(), outputs.pred_prior_fused
                    )
                else:
                    prior_mse = torch.square(
                        outputs.pred_prior_fused - outputs.gt_prior_fused.detach()
                    ).mean()
                log_dict["prior_mse"] = prior_mse
                loss_total += prior_mse

            if self.enable_prior_cosine_loss:
                prior_cosine = (
                    self._calc_prior_cosine_loss(
                        outputs.pred_prior_fused, outputs.gt_prior_fused
                    ).mean()
                    * self.prior_cosine_loss_weight
                )
                log_dict["prior_cosine"] = prior_cosine
                loss_total += prior_cosine
        else:
            code_distortion = self.code_distortion_loss(
                outputs.gt_vq_latent, outputs.out_vq_latent
            )
            log_dict["code_distortion"] = code_distortion
            loss_total += code_distortion

            code_ce = self.code_ce_loss(outputs.out_vq_logits, outputs.gt_vq_indices)
            log_dict["code_ce"] = code_ce
            loss_total += code_ce

            group_nll, group_prob_l1 = self._calc_group_ce_nll(
                outputs.out_vq_logits, outputs.gt_vq_indices
            )
            if group_nll is not None:
                group_ce = self.lambda_group_ce * group_nll.mean()
                log_dict["group_ce"] = group_ce
                loss_total += group_ce
                if group_prob_l1 is not None:
                    log_dict["group_prob_l1"] = group_prob_l1

        return loss_total, log_dict

    def run_discriminator(self, outputs: DualBetaModelOutput) -> Tuple[Tensor, Tensor]:
        other_outputs = outputs.other_outputs

        # Real samples
        if self.mc_sampling:
            disc_real_images = other_outputs["disc_real_images"]
            disc_vq_indices = other_outputs["disc_vq_indices"]

            if getattr(self.discriminator, "y_hat_cond"):
                # get y_hat of disc_real_images to use as a condition input
                disc_y_hat = self.comp_model.extract_y_hat(
                    disc_real_images,
                    disc_vq_indices,
                    outputs.beta_rate,
                    outputs.beta_vq,
                )
            else:
                disc_y_hat = None

            d_real = self.discriminator(
                disc_real_images,
                beta_1=outputs.beta_rate,
                beta_2=outputs.beta_vq,
                y_hat=disc_y_hat,
                **other_outputs,
            )
        else:
            d_real = self.discriminator(
                outputs.real_images,
                beta_1=outputs.beta_rate,
                beta_2=outputs.beta_vq,
                y_hat=outputs.y_hat,
                **other_outputs,
            )

        # Fake samples
        d_fake = self.discriminator(
            outputs.fake_images.detach(),
            beta_1=outputs.beta_rate,
            beta_2=outputs.beta_vq,
            y_hat=outputs.y_hat,
            **other_outputs,
        )

        return d_real, d_fake

    def calc_d_loss(
        self,
        d_real: Tensor,
        d_fake: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Dict]:
        l_d_real = self.gan_loss(d_real, is_real=True, is_disc=True) * 0.5
        l_d_fake = self.gan_loss(d_fake, is_real=False, is_disc=True) * 0.5
        d_loss = l_d_real + l_d_fake

        out_d_real, out_d_fake = self.calc_avg_d_score_for_log(d_real, d_fake)
        log_dict = {
            "d_real": l_d_real,
            "d_fake": l_d_fake,
            "d_total": d_loss,
            "out_d_real": out_d_real,
            "out_d_fake": out_d_fake,
        }
        return d_loss, log_dict

    @staticmethod
    def calc_avg_d_score_for_log(d_real, d_fake):
        if isinstance(d_real, list):
            out_d_real = torch.mean(
                torch.Tensor([torch.mean(rp.detach()) for rp in d_real])
            )
            out_d_fake = torch.mean(
                torch.Tensor([torch.mean(fp.detach()) for fp in d_fake])
            )
        elif isinstance(d_real, torch.Tensor):
            out_d_real = torch.mean(d_real.detach())
            out_d_fake = torch.mean(d_fake.detach())
        else:
            raise TypeError(
                f"d_real should be list or torch.Tensor, but {type(d_real)}"
            )
        return out_d_real, out_d_fake

    def save(self, current_iter: int):
        # save model
        self.model_saver.save(
            {"comp_model": self.comp_model}, "comp_model", current_iter, keep=True
        )
        self.model_saver.save(
            {"discriminator": self.discriminator},
            "discriminator",
            current_iter,
            keep=self.opt.get("keep_discriminator", False),
        )

        # save training_state
        optimizer_scheduler_dict = {
            "g_optimizer": self.g_optimizer,
            "d_optimizer": self.d_optimizer,
        }
        if self.aux_optimizer:
            optimizer_scheduler_dict["aux_optimizer"] = self.aux_optimizer
        optimizer_scheduler_dict["g_scheduler"] = self.g_scheduler
        optimizer_scheduler_dict["d_scheduler"] = self.d_scheduler
        self.model_saver.save(
            optimizer_scheduler_dict,
            "training_state",
            current_iter,
            keep=self.opt.get("keep_training_state", False),
        )

    def _get_best_checkpoint_payloads(self) -> Dict[str, Dict]:
        payloads = super()._get_best_checkpoint_payloads()
        best_cfg = self.opt.get("best_checkpoint", {})
        if best_cfg.get("save_discriminator", False):
            payloads["discriminator"] = {"discriminator": self.discriminator}
        if best_cfg.get("save_training_state", False):
            optimizer_scheduler_dict = {
                "g_optimizer": self.g_optimizer,
                "d_optimizer": self.d_optimizer,
            }
            if self.aux_optimizer:
                optimizer_scheduler_dict["aux_optimizer"] = self.aux_optimizer
            optimizer_scheduler_dict["g_scheduler"] = self.g_scheduler
            optimizer_scheduler_dict["d_scheduler"] = self.d_scheduler
            payloads["training_state"] = optimizer_scheduler_dict
        return payloads

    def _load_checkpoint(
        self,
        exp_name: str,
        itr: int,
        load_optimizer: bool = True,
        load_discriminator: bool = True,
        load_scheduler: bool = True,
        new_g_lr: Optional[float] = None,
        new_d_lr: Optional[float] = None,
        **kwargs,
    ) -> None:
        ## get checkpoint path
        path_handler = PathHandler(self.opt.path.ckpt_root, exp_name)
        model_ckpt_path = path_handler.get_ckpt_path("comp_model", itr)
        optim_ckpt_path = path_handler.get_ckpt_path("training_state", itr)
        discriminator_ckpt_path = path_handler.get_ckpt_path("discriminator", itr)
        log_dict_items(
            {
                "model_ckpt_path": model_ckpt_path,
                "optim_ckpt_path": optim_ckpt_path,
                "discriminator_ckpt_path": discriminator_ckpt_path,
            },
            level="INFO",
            indent=False,
        )

        # Load G & g_optimizer #######################################################################
        load_device = "cpu"
        self.logger.warn('load_device is set to "cpu"')
        comp_model_ckpt = torch.load(model_ckpt_path, map_location=load_device)
        out = self.comp_model.load_state_dict(comp_model_ckpt["comp_model"])
        self.logger.debug(f'comp_model.load_state_dict: "{out}"')

        if load_optimizer:
            optim_ckpt = torch.load(optim_ckpt_path, map_location=load_device)

            self.g_optimizer.load_state_dict(optim_ckpt["g_optimizer"])
            self.logger.debug(f"g_optimizer is loaded")

            if new_g_lr is not None:
                self.update_learning_rate(self.g_optimizer, new_g_lr)
                self.logger.info(f"g_optimizer lr is changed to {new_g_lr}")

            if load_scheduler:
                self.g_scheduler.load_state_dict(optim_ckpt["g_scheduler"])
                self.logger.debug(f"g_scheduler is loaded")
            else:
                self.logger.warn("g_scheduler is NOT loaded")

            if self.aux_optimizer:
                self.aux_optimizer.load_state_dict(optim_ckpt["aux_optimizer"])
                self.logger.debug(f"aux_optimizer is loaded")
        else:
            self.logger.warn(
                "g_optimizer, g_scheduler, and aux_optimizer are NOT loaded"
            )

        # Load D & d_optimizer #######################################################################
        if not load_discriminator:
            self.logger.warn("discriminator, d_optimizer, d_scheduler are NOT loaded")
            return

        discriminator_ckpt = torch.load(
            discriminator_ckpt_path, map_location=load_device
        )
        out = self.discriminator.load_state_dict(discriminator_ckpt["discriminator"])
        self.logger.debug(f'discriminator.load_state_dict: "{out}"')

        if load_optimizer:
            self.d_optimizer.load_state_dict(optim_ckpt["d_optimizer"])
            self.logger.debug(f"d_optimizer is loaded")

            if new_d_lr is not None:
                self.update_learning_rate(self.d_optimizer, new_d_lr)
                self.logger.info(f"d_optimizer lr is changed to {new_d_lr}")

            if load_scheduler:
                self.d_scheduler.load_state_dict(optim_ckpt["d_scheduler"])
                self.logger.debug(f"d_scheduler is loaded")
            else:
                self.logger.warn("d_scheduler is NOT loaded")
        else:
            self.logger.warn("d_optimizer and d_scheduler are NOT loaded")
