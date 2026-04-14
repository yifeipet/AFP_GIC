
from typing import List, Dict, Tuple

import torch
from torch import Tensor

from src.utils.registry import MODEL_REGISTRY
from src.models.subnet import build_subnet

from .hyperprior_dc_vic_model import (
    HyperpriorDualCondVicModel,
)


@MODEL_REGISTRY.register()
class HyperpriorCharmDualCondVicModel(HyperpriorDualCondVicModel):

    def _build_subnets(self) -> None:
        super()._build_subnets()
        self.context_model = build_subnet(
            self.opt.subnet.context_model, subnet_type="context_model"
        )

    def estimate_entropy(self, y: Tensor, is_train: bool = True):
        z = self.hyperencoder(y)
        z_hat, z_likelihood = self.entropy_model_z(z, is_train=is_train)
        hyper_out = self.hyperdecoder(z_hat)
        y_hat, y_likelihood, y_q_likelihood = self.context_model(
            y,
            hyper_out,
            self.entropy_model_y,
            is_train=is_train,
            calc_q_likelihood=True,
        )

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

    def codec_setup(self):
        super().codec_setup()
        self.context_model.to("cpu")

    def _compress_estimate_entropy(self, y: Tensor):
        z = self.hyperencoder(y)
        y = y.cpu()
        z = z.cpu()

        z_hat, z_likelihood = self.entropy_model_z(z, is_train=False)
        z_str = self.entropy_model_z.compress(z)

        hyper_out = self.hyperdecoder(z_hat)
        y_str, y_hat, y_likelihood = self.context_model.forward_compress(
            y, hyper_out, self.entropy_model_y
        )
        return {
            'y_hat': y_hat,
            'y_likelihood': y_likelihood,
            'y_str': y_str,
            'z_hat': z_hat,
            'z_likelihood': z_likelihood,
            'z_str': z_str,
        }

    def _decompress_estimate_entropy(self, z_str: bytes, y_str: bytes, zH: int, zW: int) -> Tuple[Tensor, Tensor]:
        z_symbol = self.entropy_model_z.decompress([z_str], (zH, zW))
        z_hat = self.entropy_model_z.dequantize(z_symbol)
        hyper_out = self.hyperdecoder(z_hat)

        y_hat, y_symbol = self.context_model.forward_decompress(
            y_str, hyper_out, self.entropy_model_y
        )
        return y_hat, z_hat
