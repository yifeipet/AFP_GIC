import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY
from src.models.subnet.autoencoder.elic_autoencoder import ElicEncoder

from timm.models.layers import trunc_normal_

@ENCODER_REGISTRY.register()
class ElicVqScEncoder(ElicEncoder):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 192,
        main_ch: int = 192,
        block_mid_ch: int = 192,
        num_blocks: int = 3,
        res_in_res: bool = False,
        input_feat_ch: int = 5,
        proj_init: bool = True,
        proj_init_std: float = 0.02,
    ):
        super().__init__(
            in_ch, out_ch, main_ch, block_mid_ch, num_blocks, res_in_res
        )
        self.projection = nn.Conv2d(input_feat_ch, main_ch, 1)
        if proj_init:
            nn.init.constant(self.projection.bias, 0.0)
            trunc_normal_(self.projection.weight, std=proj_init_std)
        self.input_vq_latent = True

    def forward(self, x, feat):
        x = self.conv1(x) # [H, W] -> [H/2, W/2]
        x = self.block1(x)

        x = self.conv2(x) # [H/2, W/2] -> [H/4, W/4]
        x = self.block2(x)
        x = self.attn2(x)

        x = self.conv3(x) # [H/4, W/4] -> [H/8, W/8]
        proj = self.projection(feat)
        x = x + proj
        x = self.block3(x)

        x = self.conv4(x) # [H/8, W/8] -> [H/16, W/16]
        x = self.attn4(x)

        return x
    

@ENCODER_REGISTRY.register()
class ElicVqCatScEncoder(ElicEncoder):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 192,
        main_ch: int = 192,
        block_mid_ch: int = 192,
        num_blocks: int = 3,
        res_in_res: bool = False,
        input_feat_ch: int = 5,
        proj_init: bool = True,
        proj_init_std: float = 0.02,
        proj_pos: str = 'conv3',
    ):
        super().__init__(
            in_ch, out_ch, main_ch, block_mid_ch, num_blocks, res_in_res
        )
        
        self.projection = nn.Conv2d(main_ch+input_feat_ch, main_ch, kernel_size=3, padding=1)
        if proj_init:
            nn.init.constant(self.projection.bias, 0.0)
            trunc_normal_(self.projection.weight, std=proj_init_std)
        self.input_vq_latent = True
        assert proj_pos in ['conv3', 'conv4']
        self.proj_pos = proj_pos
    
    def run_projection(self, x, feat):
        proj = self.projection(torch.cat([feat, x], dim=1))
        x = x + proj
        return x

    def forward(self, x, feat):
        x = self.conv1(x) # [H, W] -> [H/2, W/2]
        x = self.block1(x)

        x = self.conv2(x) # [H/2, W/2] -> [H/4, W/4]
        x = self.block2(x)
        x = self.attn2(x)

        x = self.conv3(x) # [H/4, W/4] -> [H/8, W/8]
        if self.proj_pos == 'conv3':
            x = self.run_projection(x, feat)
        x = self.block3(x)

        x = self.conv4(x) # [H/8, W/8] -> [H/16, W/16]
        if self.proj_pos == 'conv4':
            x = self.run_projection(x, feat)
        x = self.attn4(x)

        return x
    

@ENCODER_REGISTRY.register()
class ElicVqEmbCatEncoder(ElicVqCatScEncoder):
    def __init__(
        self,
        vq_n_embed: int,
        vq_ind_embed_dim: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self.vq_ind_emb = nn.Embedding(vq_n_embed, vq_ind_embed_dim)
    
    def run_projection(self, x, feat, vq_indices):
        emb = self.vq_ind_emb(vq_indices)
        emb = emb.permute(0, 3, 1, 2).contiguous()
        proj = self.projection(torch.cat([feat, x, emb], dim=1))
        x = x + proj
        return x

    def forward(self, x: Tensor, feat: Tensor, vq_indices: Tensor):
        x = self.conv1(x) # [H, W] -> [H/2, W/2]
        x = self.block1(x)

        x = self.conv2(x) # [H/2, W/2] -> [H/4, W/4]
        x = self.block2(x)
        x = self.attn2(x)

        x = self.conv3(x) # [H/4, W/4] -> [H/8, W/8]
        if self.proj_pos == 'conv3':
            x = self.run_projection(x, feat, vq_indices)
        x = self.block3(x)

        x = self.conv4(x) # [H/8, W/8] -> [H/16, W/16]
        if self.proj_pos == 'conv4':
            x = self.run_projection(x, feat, vq_indices)
        x = self.attn4(x)

        return x