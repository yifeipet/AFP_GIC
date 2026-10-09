from typing import Optional

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from taming.modules.discriminator.model import weights_init
from src.utils.registry import DISCRIMINATOR_REGISTRY

from src.models.layer.fourier_enc import FourierEncoding
from .taming_nlayer_discriminator import NLayerDiscriminator, get_norm_layer


@DISCRIMINATOR_REGISTRY.register()
class DualBetaCondTamingNLayerDiscriminator(NLayerDiscriminator):
    def __init__(
        self,
        *args,
        max_beta_1: float = -1.0,  # must be specified
        max_beta_2: float = -1.0,  # must be specified
        L: int = 10,
        cond_ch: int = 8,
        use_pi: bool = False,
        include_x: bool = True,
        y_hat_cond: bool = False,
        y_hat_in_ch: Optional[int] = None,
        y_hat_out_ch: Optional[int] = None,
        weight_init: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embed_1 = FourierEncoding(
            L=L, max_beta=max_beta_1, use_pi=use_pi, include_x=include_x
        )
        self.embed_2 = FourierEncoding(
            L=L, max_beta=max_beta_2, use_pi=use_pi, include_x=include_x
        )
        mlp_in_ch = 2 * (2 * L + 1) if include_x else 2 * 2 * L
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_ch, cond_ch),
            nn.ReLU(inplace=True),
            nn.Linear(cond_ch, cond_ch),
        )

        self.y_hat_cond = y_hat_cond
        if y_hat_cond:
            assert y_hat_in_ch is not None
            assert y_hat_out_ch is not None
            self.y_hat_emb = nn.Sequential(
                nn.Conv2d(
                    y_hat_in_ch,
                    y_hat_out_ch,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        if weight_init:
            self.apply(weights_init)

    def forward(
        self,
        x,
        beta_1: Tensor,
        beta_2: Tensor,
        y_hat: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        _, _, H, W = x.size()

        cond_1 = self.embed_1.embed(beta_1).detach().to(x.device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(x.device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]
        cond = cond.expand(x.shape[0], -1, H, W)
        x = torch.cat([x, cond], dim=1)

        if self.y_hat_cond:
            y_hat = y_hat.detach()
            y_hat = self.y_hat_emb(y_hat)
            y_hat = F.interpolate(y_hat, size=(H, W), mode="nearest")
            x = torch.cat([x, y_hat], dim=1)

        return super().forward(x)


class BetaScaleShiftModule(nn.Module):
    def __init__(self, cond_ch: int, feat_ch: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(cond_ch, cond_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.scale = nn.Conv2d(cond_ch, feat_ch, kernel_size=1, stride=1, padding=0)
        self.shift = nn.Conv2d(cond_ch, feat_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, feat: Tensor, cond_feat: Tensor) -> Tensor:
        cond_feat = self.shared(cond_feat)
        scale = self.scale(cond_feat)
        shift = self.shift(cond_feat)
        return feat * (1 + scale) + shift


@DISCRIMINATOR_REGISTRY.register()
class DualBetaFtTamingNLayerDiscriminator(NLayerDiscriminator):
    def __init__(
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_type: str = "batchnorm",
        norm_kwargs: dict = {},
        max_beta_1: float = -1.0,  # must be specified
        max_beta_2: float = -1.0,  # must be specified
        L: int = 10,
        cond_ch: int = 8,
        use_pi: bool = False,
        include_x: bool = True,
        weight_init: bool = True,
        **kwargs,
    ):
        super(NLayerDiscriminator, self).__init__()
        assert max_beta_1 > 0
        assert max_beta_2 > 0

        use_bias = norm_type != "batchnorm"

        kw = 4
        padw = 1
        self.main = nn.ModuleList()
        self.beta_ft_list = nn.ModuleList()

        self.main.append(
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        )
        self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, ndf))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            block = nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                get_norm_layer(ndf * nf_mult, norm_type, **norm_kwargs),
            )
            self.main.append(block)
            self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, ndf * nf_mult))

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        block = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            get_norm_layer(ndf * nf_mult, norm_type, **norm_kwargs),
        )
        self.main.append(block)
        self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, ndf * nf_mult))

        self.final_layer = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        )  # output 1 channel prediction map

        assert len(self.main) == len(self.beta_ft_list)

        ####### Beta cond #############################################################
        self.embed_1 = FourierEncoding(
            L=L, max_beta=max_beta_1, use_pi=use_pi, include_x=include_x
        )
        self.embed_2 = FourierEncoding(
            L=L, max_beta=max_beta_2, use_pi=use_pi, include_x=include_x
        )
        mlp_in_ch = 2 * (2 * L + 1) if include_x else 2 * 2 * L
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_ch, cond_ch),
            nn.ReLU(inplace=True),
            nn.Linear(cond_ch, cond_ch),
        )

        if weight_init:
            self.apply(weights_init)

    def forward(self, x, beta_1: Tensor, beta_2: Tensor, *args, **kwargs) -> Tensor:
        cond_1 = self.embed_1.embed(beta_1).detach().to(x.device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(x.device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]

        for layer, beta_ft in zip(self.main, self.beta_ft_list):
            x = layer(x)
            x = beta_ft(x, cond)
        x = self.final_layer(x)

        return x
