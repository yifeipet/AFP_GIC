from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY
from src.models.subnet.autoencoder.elic_autoencoder import (
    ElicEncoder,
    ElicDecoder,
    up_conv,
)
from src.models.layer.elic_layers import ResidualBottleneckBlocks
from src.models.subnet.autoencoder.base_autoencoder import (
    BaseEncoder,
    BaseDecoder,
)
from src.models.layer.cheng_nlam import ChengNLAM

from timm.models.layers import trunc_normal_
from src.models.layer.fourier_enc import FourierEncoding
from .elic_insert_encoder import ElicVqEmbCatEncoder


class BetaScaleShiftModule(nn.Module):
    def __init__(self, cond_ch: int, feat_ch: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(cond_ch, cond_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.scale = nn.Conv2d(
            cond_ch, feat_ch, kernel_size=1, stride=1, padding=0
        )
        self.shift = nn.Conv2d(
            cond_ch, feat_ch, kernel_size=1, stride=1, padding=0
        )

    def forward(self, feat: Tensor, cond_feat: Tensor) -> Tensor:
        cond_feat = self.shared(cond_feat)
        scale = self.scale(cond_feat)
        shift = self.shift(cond_feat)
        return feat * (1 + scale) + shift


@ENCODER_REGISTRY.register()
class ElicDualBetaFtVqScEncoder(ElicEncoder):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 192,
        main_ch: int = 192,
        block_mid_ch: int = 192,
        num_blocks: int = 3,
        ## Beta cond
        max_beta_1: float = 5.12,
        max_beta_2: float = 5.12,
        cond_ch: int = 512,
        L: int = 10,
        use_pi: bool = True,
        include_x: bool = False,
        ## Feat
        input_feat_ch: int = 5,
        proj_init: bool = True,
        proj_init_std: float = 0.02,
    ):
        super().__init__(in_ch=in_ch, out_ch=out_ch, main_ch=main_ch, 
                         block_mid_ch=block_mid_ch, num_blocks=num_blocks)
        self.layer_out_ch_list = [
            ('conv1', main_ch),
            ('block1', main_ch),
            ('conv2', main_ch),
            ('block2', main_ch),
            ('attn2', main_ch),
            ('conv3', main_ch),
            ('block3', main_ch),
            ('conv4', out_ch),
            ('attn4', out_ch),
        ]
        self.beta_ft_list = nn.ModuleList()
        for _, _ch in self.layer_out_ch_list:
            self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, _ch))

        ## Beta Conditioning
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

        self.projection = nn.Conv2d(
            main_ch + input_feat_ch, main_ch, kernel_size=3, padding=1
        )
        if proj_init:
            nn.init.constant(self.projection.bias, 0.0)
            trunc_normal_(self.projection.weight, std=proj_init_std)
        self.input_vq_latent = True

    def forward(self, x, feat, beta_1: float, beta_2: float):
        cond_1 = self.embed_1.embed(beta_1).detach().to(x.device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(x.device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        # breakpoint()
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]
        # breakpoint()

        x = self.conv1(x)  # [H, W] -> [H/2, W/2]
        x = self.beta_ft_list[0](x, cond)
        x = self.block1(x)
        x = self.beta_ft_list[1](x, cond)

        x = self.conv2(x)  # [H/2, W/2] -> [H/4, W/4]
        x = self.beta_ft_list[2](x, cond)

        x = self.block2(x)
        x = self.beta_ft_list[3](x, cond)
        x = self.attn2(x)
        x = self.beta_ft_list[4](x, cond)

        x = self.conv3(x)  # [H/4, W/4] -> [H/8, W/8]
        x = self.beta_ft_list[5](x, cond)
        proj = self.projection(torch.cat([feat, x], dim=1))
        x = x + proj
        x = self.block3(x)
        x = self.beta_ft_list[6](x, cond)

        x = self.conv4(x)  # [H/8, W/8] -> [H/16, W/16]
        x = self.beta_ft_list[7](x, cond)
        x = self.attn4(x)
        x = self.beta_ft_list[8](x, cond)

        return x


@ENCODER_REGISTRY.register()
class ElicDualBetaFtVqEmbCatEncoder(ElicVqEmbCatEncoder):
    def __init__(self,
                 main_ch: int = 192,
                 out_ch: int = 192,
                 ## Beta cond
                 max_beta_2: float = 5.12,
                 max_beta_1: float = 5.12,
                 cond_ch: int = 512,
                 L: int = 10,
                 use_pi: bool = True,
                 include_x: bool = False,
                 **kwargs
                ):
        super().__init__(main_ch=main_ch, out_ch=out_ch, **kwargs)
        self.layer_out_ch_list = [
            ('conv1', main_ch),
            ('block1', main_ch),
            ('conv2', main_ch),
            ('block2', main_ch),
            ('attn2', main_ch),
            ('conv3', main_ch),
            ('block3', main_ch),
            ('conv4', out_ch),
            ('attn4', out_ch),
        ]
        self.beta_ft_list = nn.ModuleList()
        for _, _ch in self.layer_out_ch_list:
            self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, _ch))

        ## Beta Conditioning
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
        self.input_vq_latent = True

    def forward(self, x, feat, beta_1: float, beta_2: float, vq_indices: Tensor):
        cond_1 = self.embed_1.embed(beta_1).detach().to(x.device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(x.device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        # breakpoint()
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]

        x = self.conv1(x)  # [H, W] -> [H/2, W/2]
        x = self.beta_ft_list[0](x, cond)
        x = self.block1(x)
        x = self.beta_ft_list[1](x, cond)

        x = self.conv2(x)  # [H/2, W/2] -> [H/4, W/4]
        x = self.beta_ft_list[2](x, cond)

        x = self.block2(x)
        x = self.beta_ft_list[3](x, cond)
        x = self.attn2(x)
        x = self.beta_ft_list[4](x, cond)

        x = self.conv3(x)  # [H/4, W/4] -> [H/8, W/8]
        if self.proj_pos == 'conv3':
            x = self.run_projection(x, feat, vq_indices)
        x = self.block3(x)
        x = self.beta_ft_list[6](x, cond)

        x = self.conv4(x)  # [H/8, W/8] -> [H/16, W/16]
        if self.proj_pos == 'conv4':
            x = self.run_projection(x, feat, vq_indices)
        x = self.beta_ft_list[7](x, cond)
        x = self.attn4(x)
        x = self.beta_ft_list[8](x, cond)

        return x



@DECODER_REGISTRY.register()
class ElicDualBetaFtFeatFusionDecoder(BaseDecoder):
    def __init__(
        self,
        fusion_layer_dict: dict[str, str],
        feat_layer_name: str,
        in_ch: int = 192,
        out_ch: int = 3,
        main_ch: int = 192,
        block_mid_ch: int = 192,
        num_blocks: int = 3,
        use_tanh: bool = True,
        pixel_shuffle: bool = False,
        res_in_res: bool = False,
        ## Beta cond
        max_beta_1: float = 5.12,
        max_beta_2: float = 5.12,
        cond_ch: int = 512,
        L: int = 10,
        use_pi: bool = True,
        include_x: bool = False,
        beta_weight_init: bool = False,
        beta_weight_init_std: float = 0.02,
    ):
        super().__init__()
        blk_kwargs = dict(
            ch=main_ch,
            mid_ch=block_mid_ch,
            num_blocks=num_blocks,
            res_in_res=res_in_res,
        )

        self.use_tanh = use_tanh

        self.attn1 = ChengNLAM(in_ch)
        self.conv1 = up_conv(
            in_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle
        )
        self.block1 = ResidualBottleneckBlocks(**blk_kwargs)

        self.conv2 = up_conv(
            main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle
        )
        self.attn2 = ChengNLAM(main_ch)
        self.block2 = ResidualBottleneckBlocks(**blk_kwargs)

        self.conv3 = up_conv(
            main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle
        )
        self.block3 = ResidualBottleneckBlocks(**blk_kwargs)

        self.conv4 = up_conv(
            main_ch, out_ch, kernel_size=5, pixel_shuffle=pixel_shuffle
        )

        self.layer_names = [
            "attn1",
            "conv1",
            "block1",
            "conv2",
            "attn2",
            "block2",
            "conv3",
            "block3",
            "conv4",
        ]
        self.feat_layer = feat_layer_name
        assert self.feat_layer in self.layer_names
        self.fusion_layer_dict = fusion_layer_dict
        for k, v in self.fusion_layer_dict.items():
            assert k in self.layer_names

        self.beta_ft_list = nn.ModuleList()
        ch_list = [in_ch, in_ch] + [main_ch] * 7
        for _ch in ch_list:
            self.beta_ft_list.append(BetaScaleShiftModule(cond_ch, _ch))

        ## Beta Conditioning
        self.max_beta_1 = max_beta_1
        self.max_beta_2 = max_beta_2
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
        self.init_fuse = BetaScaleShiftModule(cond_ch, main_ch)
        if beta_weight_init:
            for m in self.init_fuse.modules():
                if isinstance(m, nn.Conv2d):
                    print('BETA INIT WEIGHT INIT')
                    nn.init.constant_(m.bias, 0.0)
                    trunc_normal_(m.weight, std=beta_weight_init_std)

    def get_interm_feat(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def get_feats(self, x: Tensor, beta_1: float, beta_2: float):
        cond_1 = self.embed_1.embed(beta_1).detach().to(x.device)  # [1, 2L]
        cond_2 = self.embed_2.embed(beta_2).detach().to(x.device)  # [1, 2L]
        cond = torch.cat([cond_1, cond_2], dim=1)  # [1, 4L]
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1)  # [1, cond_ch, 1, 1]

        fusion_feat_dict = {}
        query_layer_list = list(self.fusion_layer_dict.keys())
        feat_1 = None

        x = self.init_fuse(x, cond) + x

        for layer_name, beta_ft_layer in zip(self.layer_names, self.beta_ft_list):
            layer = getattr(self, layer_name)
            x = beta_ft_layer(x, cond)
            x = layer(x)

            if layer_name == self.feat_layer:
                feat_1 = x

            if layer_name in query_layer_list:
                k = self.fusion_layer_dict[layer_name]
                fusion_feat_dict[k] = x

            if len(fusion_feat_dict) == len(query_layer_list):
                break

        return feat_1, fusion_feat_dict
