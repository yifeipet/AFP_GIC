import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from taming.modules.discriminator.model import weights_init
from src.utils.registry import DISCRIMINATOR_REGISTRY

from src.models.layer.fourier_enc import FourierEncoding
from .taming_nlayer_discriminator import NLayerDiscriminator, get_norm_layer

from typing import Optional

class BetaEmbedding(nn.Module):
    def __init__(
        self,
        max_beta_1: float = -1,
        max_beta_2: float = -1,
        L: int = 10,
        cond_ch: int = 8,
        use_pi: bool = False,
        include_x: bool = True,
    ) -> None:
        super().__init__()
        ## Beta Embedding Module
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

    def forward(self, beta_1: Tensor, beta_2: Tensor, device) -> Tensor:
        cond_1 = self.embed_1.embed(beta_1).detach().to(device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        cond = self.mlp(cond) # [1, cond_ch, 1, 1]
        return cond


class LatentYEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_sequence = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, y_hat: Tensor) -> Tensor:
        return self.input_sequence(y_hat)


@DISCRIMINATOR_REGISTRY.register()
class OasisDualBetaCondTamingNLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        num_upsample: int = 1,
        out_nc: int = 128,
        use_actnorm=False,
        norm_type: str = "batchnorm",
        norm_kwargs: dict = {},
        y_hat_cond: bool = False,
        y_hat_in_ch: Optional[int] = None,
        y_hat_out_ch: Optional[int] = None,
        max_beta_1: float = -1,
        max_beta_2: float = -1,
        L: int = 10,
        cond_ch: int = 8,
        use_pi: bool = False,
        include_x: bool = True,
        weight_init: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert num_upsample > 0

        ## Main body
        kw = 4
        padw = 1
        use_bias = norm_type != 'batchnorm'

        self.body = nn.ModuleList()

        channels = [
            ndf * min(2 ** i, 8) for i in range(n_layers)
        ]

        layers = [nn.Conv2d(input_nc, channels[0], kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        self.body.append(nn.Sequential(*layers))

        for n in range(1, n_layers):  # gradually increase the number of filters
            _ch_in = channels[n - 1]
            _ch_out = channels[n]
            layers = [
                nn.Conv2d(_ch_in, _ch_out, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                get_norm_layer(_ch_out, norm_type, **norm_kwargs),
                nn.LeakyReLU(0.2, True)
            ]
            self.body.append(nn.Sequential(*layers))

        bottleneck_ch = channels[n_layers - 1]
        layers = [
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, bias=use_bias),
            get_norm_layer(bottleneck_ch, norm_type, **norm_kwargs),
            nn.LeakyReLU(0.2, True)
        ]
        self.bottleneck = nn.Sequential(*layers)

        self.up_blocks = nn.ModuleList()
        for i in range(num_upsample):
            _ch_in = channels[n_layers - 1 - i]
            _ch_out = channels[n_layers - 1 - i - 1]
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(_ch_in, _ch_out, kernel_size=3, stride=1, padding=1),
                    get_norm_layer(_ch_out, norm_type, **norm_kwargs),
                    nn.LeakyReLU(0.2, True)
                )
            )

        head_in_ch = channels[n_layers - 1 - num_upsample]
        self.head = nn.Sequential(
            nn.Conv2d(head_in_ch, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, out_nc, kernel_size=1, stride=1, padding=0),
        )

        ## Beta Embedding Module
        self.beta_emb = BetaEmbedding(
            max_beta_1=max_beta_1,
            max_beta_2=max_beta_2,
            L=L,
            cond_ch=cond_ch,
            use_pi=use_pi,
            include_x=include_x,
        )

        ## y_hat embedding
        if y_hat_cond:
            assert isinstance(y_hat_in_ch, int)
            assert isinstance(y_hat_out_ch, int)
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

    def forward(self, x, y_hat=None, beta_1=None, beta_2=None):
        _, _,H, W = x.size()
        inputs = [x]

        if beta_1 is not None and beta_2 is not None:
            beta_cond = self.beta_emb(beta_1, beta_2, x.device)
            beta_cond = beta_cond.unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]
            beta_cond = beta_cond.expand_as(x)  # [B, cond_ch, H, W]
            inputs.append(beta_cond)

        if y_hat is not None:
            y_hat = self.y_hat_emb(y_hat)
            y_hat = F.interpolate(y_hat, size=(H, W), mode="nearest")
            inputs.append(y_hat)

        x = torch.cat(inputs, dim=1)
        
        shortcut_list = []
        for layer in self.body:
            x = layer(x)
            shortcut_list.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.up_blocks):
            x = layer(x)
            x = x + shortcut_list[-i - 2]

        x = self.head(x)
        return x
