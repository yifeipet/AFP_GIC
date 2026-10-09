import functools
import torch.nn as nn

# from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.discriminator.model import weights_init
from taming.modules.util import ActNorm

from src.utils.registry import DISCRIMINATOR_REGISTRY


def get_norm_layer(num_channels: int, norm_type: str, **kwargs):
    if norm_type == "batchnorm":
        norm_layer = nn.BatchNorm2d(num_channels, **kwargs)
    elif norm_type == "instancenorm":
        norm_layer = nn.InstanceNorm2d(num_channels, **kwargs)
    elif norm_type == "none":
        norm_layer = nn.Identity()
    elif norm_type == "layernorm":
        norm_layer = nn.LayerNorm(num_channels, **kwargs)
    elif norm_type == "groupnorm":
        norm_layer = nn.GroupNorm(num_channels=num_channels, **kwargs)
    elif norm_type == "actnorm":
        norm_layer = ActNorm(num_channels, **kwargs)
    else:
        raise NotImplementedError(f"norm layer [{norm_type}] is not implemented")
    return norm_layer


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        input_nc=3,
        ndf=64,
        out_nc=1,
        n_layers=3,
        keep_shape: bool = False,
        use_actnorm=False,
        norm_type: str = "batchnorm",
        norm_kwargs: dict = {},
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if use_actnorm:
            import warnings

            warnings.warn(
                f'use_actnorm is deprecated. Use norm_type="actnorm" instead. norm_type="{norm_type}" is changed to "actnorm"'
            )
            norm_type = "actnorm"
        # if use_actnorm:
        #     norm_layer = ActNorm
        # else:
        #     norm_layer = get_norm_layer(norm_type)

        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        # use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        # use_bias = norm_layer != nn.BatchNorm2d
        use_bias = norm_type != "batchnorm"

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                get_norm_layer(ndf * nf_mult, norm_type, **norm_kwargs),
                nn.LeakyReLU(0.2, True),
            ]

        kw = 3 if keep_shape else kw # to keep the shape of the output

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            get_norm_layer(ndf * nf_mult, norm_type, **norm_kwargs),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, out_nc, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


@DISCRIMINATOR_REGISTRY.register()
class TamingNLayerDiscriminator(NLayerDiscriminator):
    def __init__(self, *args, weight_init: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if weight_init:
            self.apply(weights_init)

    def forward(self, x, *args, **kwargs):
        return super().forward(x)
