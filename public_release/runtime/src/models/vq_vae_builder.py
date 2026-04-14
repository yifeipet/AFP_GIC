import os
from copy import deepcopy

import torch

from src.utils.logger import get_root_logger
from src.models.adacode_adapter import AdaCodeAdapter


def build_pretrained_vq_model(vq_model_opt: dict, device: str):
    from ldm.models.autoencoder import VQModelInterface

    logger = get_root_logger()
    vq_model_opt = deepcopy(vq_model_opt)
    ckpt_path = vq_model_opt.pop('ckpt_path')

    model = VQModelInterface(**vq_model_opt)
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
        load_msg = model.load_state_dict(state_dict)
        logger.info(f'load_msg: {load_msg}')
    else:
        logger.info('No ckpt_path is provided. Skip loading VQGAN checkpoint.')
    return model.to(device)


def build_pretrained_prior_model(
    prior_type: str,
    prior_opt: dict,
    device: str,
):
    if prior_type == "vqgan":
        return build_pretrained_vq_model(prior_opt, device)

    if prior_type == "adacode":
        prior_opt = deepcopy(prior_opt)
        ckpt_path = prior_opt.pop("ckpt_path", "")
        return AdaCodeAdapter(
            ckpt_path=ckpt_path,
            device=device,
            **prior_opt,
        )

    raise ValueError(f"Unsupported prior_type: {prior_type}")
