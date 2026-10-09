from typing import List, Dict, Tuple, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.utils.registry import MODEL_REGISTRY
from src.models.subnet import build_subnet

from .hyperprior_vic_model import HyperpriorVicModel


@MODEL_REGISTRY.register()
class HyperpriorCharmVicModel(HyperpriorVicModel):

    def _build_subnets(self) -> None:
        super()._build_subnets()
        self.context_model = build_subnet(self.opt.subnet.context_model, subnet_type='context_model')
        if self.opt.subnet.get('residual_predictor'):
            raise NotImplementedError('Residual Predictor is not available yet in HypepriorCharmModel.')

    def estimate_entropy(self, y: Tensor, is_train: bool=True):
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood, y_q_likelihood = self.context_model(
            y, hyper_out, self.entropy_model_y, is_train=is_train, calc_q_likelihood=True)

        with torch.no_grad():
            _, z_q_likelihood = self.entropy_model_z(z, is_train=False)

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