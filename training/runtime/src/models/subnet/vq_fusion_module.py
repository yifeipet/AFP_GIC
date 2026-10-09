from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, List, Dict, Any

from src.utils.logger import log_dict_items

from src.models.layer.codeformer_layers import FuseSftBlock
from src.models.layer.light_fuse_layer import LightFuseSftBlock
from src.utils.registry import VQ_FUSION_REGISTRY

from ldm.modules.diffusionmodules.model import nonlinearity, Normalize
from ldm.modules.diffusionmodules.model import Decoder as VqDecoder

from timm.models.layers import trunc_normal_

def build_vq_fusion_module(vq_fusion_opt: Dict) -> nn.Module:
    subnet_opt = deepcopy(vq_fusion_opt)
    network_type = subnet_opt.pop('type', 'VqDecFusionModule')
    subnet = VQ_FUSION_REGISTRY.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet


@VQ_FUSION_REGISTRY.register()
class VqDecFusionModule(nn.Module):
    def __init__(self, 
                 fuse_scedule_dict: dict[str, dict],
                 fuse_type: str='sft',
                 weight_init: bool = False,
                 weight_init_std: float = 0.02,) -> None:
        super().__init__()
        assert fuse_type in ['sft', 'light_sft'], 'For now, only "sft" fusion is supported'

        self.check_fusion_schedule(fuse_scedule_dict)
        self.fusion_modules = nn.ModuleDict()

        for k, v in fuse_scedule_dict.items():
            if fuse_type == 'sft':
                self.fusion_modules[k] = FuseSftBlock(
                    cond_ch=v['cond_ch'],
                    dec_ch=v['dec_ch'],
                    mid_ch=v['mid_ch'],
                )
            elif fuse_type == 'light_sft':
                self.fusion_modules[k] = LightFuseSftBlock(
                    cond_ch=v['cond_ch'],
                    dec_ch=v['dec_ch'],
                    mid_ch=v['mid_ch'],
                )
            
        self.fusion_keys = list(fuse_scedule_dict.keys())

        if weight_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=weight_init_std)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=weight_init_std)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def check_fusion_schedule(fuse_scedule_dict: dict[str, dict]) -> None:
        assert isinstance(fuse_scedule_dict, dict)
        for k, v in fuse_scedule_dict.items():
            assert isinstance(v, dict)
            assert 'cond_ch' in v
            assert 'dec_ch' in v

    def forward(self, z: Tensor, cond_feats: dict[str, Tensor], vq_dec: VqDecoder, w: float=1.) -> Tensor:
        N, _, H, W = z.shape
        if min(H*8, W*8) > 1024:
            return self.forward_split(z, cond_feats, vq_dec, w)

        vq_dec.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = vq_dec.conv_in(z)

        if 'before_mid' in self.fusion_keys:
            h = self.fusion_modules['before_mid'](h, cond_feats['before_mid'], w)

        # middle
        h = vq_dec.mid.block_1(h, temb) # type: ignore
        h = vq_dec.mid.attn_1(h) # type: ignore
        h = vq_dec.mid.block_2(h, temb) # type: ignore

        if 'after_mid' in self.fusion_keys:
            h = self.fusion_modules['after_mid'](h, cond_feats['after_mid'], w)

        # upsampling
        for i_level in reversed(range(vq_dec.num_resolutions)):
            for i_block in range(vq_dec.num_res_blocks+1):
                h = vq_dec.up[i_level].block[i_block](h, temb) # type: ignore
                if len(vq_dec.up[i_level].attn) > 0: # type: ignore
                    h = vq_dec.up[i_level].attn[i_block](h) # type: ignore
            
            scale = 2 ** i_level # 8, 4, 2, 1
            key = f'block_1_{scale}'
            if key in self.fusion_keys:
                h = self.fusion_modules[key](h, cond_feats[key], w)

            if i_level != 0:
                h = vq_dec.up[i_level].upsample(h) # type: ignore

        # end
        if vq_dec.give_pre_end:
            return h

        h = vq_dec.norm_out(h)
        h = nonlinearity(h)
        h = vq_dec.conv_out(h)
        if vq_dec.tanh_out:
            h = torch.tanh(h)
        return h


    def forward_split(self, z: Tensor, cond_feats: dict[str, Tensor], vq_dec: VqDecoder, w: float=1.) -> Tensor:
        N, _, H, W = z.size()
        ks = (32, 32)
        stride = (8, 8)
        vqf = 4  #
        self.split_input_params = {"ks": ks, "stride": stride,
                                        "vqf": vqf,
                                        "patch_distributed_vq": True,
                                        "tie_braker": False,
                                        "clip_max_weight": 0.5,
                                        "clip_min_weight": 0.01,
                                        "clip_max_tie_weight": 0.5,
                                        "clip_min_tie_weight": 0.01}
        fold, unfold, normalization, weighting = self.get_fold_unfold(
                z, kernel_size=ks, stride=stride, uf=1, df=1)
        
        vq_dec.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = vq_dec.conv_in(z)

        if 'before_mid' in self.fusion_keys:
            h = self.fusion_modules['before_mid'](h, cond_feats['before_mid'], w)

        # middle
        h = vq_dec.mid.block_1(h, temb) # type: ignore

        crops = unfold(h)  # (bn, nc * prod(**ks), L)
        # Reshape to img shape
        crops = crops.view((N, -1, ks[0], ks[1], crops.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

        output_list = [vq_dec.mid.attn_1(crops[:, :, :, :, i]) for i in range(crops.shape[-1])]  # type: ignore

        o = torch.stack(output_list, axis=-1)
        o = o * weighting

        # Reverse reshape to img shape
        o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
        # stitch crops together
        h = fold(o)
        h = h / normalization

        h = vq_dec.mid.block_2(h, temb) # type: ignore

        if 'after_mid' in self.fusion_keys:
            h = self.fusion_modules['after_mid'](h, cond_feats['after_mid'], w)

        # upsampling
        for i_level in reversed(range(vq_dec.num_resolutions)):
            for i_block in range(vq_dec.num_res_blocks+1):
                h = vq_dec.up[i_level].block[i_block](h, temb) # type: ignore
                if len(vq_dec.up[i_level].attn) > 0: # type: ignore

                    fold, unfold, normalization, weighting = self.get_fold_unfold(
                        h, kernel_size=ks, stride=stride, uf=1, df=1)
                
                    crops = unfold(h)  # (bn, nc * prod(**ks), L)
                    # Reshape to img shape
                    crops = crops.view((N, -1, ks[0], ks[1], crops.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                    output_list = [vq_dec.up[i_level].attn[i_block](crops[:, :, :, :, i]) for i in range(crops.shape[-1])]  # type: ignore

                    o = torch.stack(output_list, axis=-1)
                    o = o * weighting

                    # Reverse reshape to img shape
                    o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                    h = fold(o) / normalization # stitch crops together

                    # h = vq_dec.up[i_level].attn[i_block](h) # type: ignore
            
            scale = 2 ** i_level # 8, 4, 2, 1
            key = f'block_1_{scale}'
            if key in self.fusion_keys:
                h = self.fusion_modules[key](h, cond_feats[key], w)

            if i_level != 0:
                h = vq_dec.up[i_level].upsample(h) # type: ignore

        # end
        if vq_dec.give_pre_end:
            return h

        h = vq_dec.norm_out(h)
        h = nonlinearity(h)
        h = vq_dec.conv_out(h)
        if vq_dec.tanh_out:
            h = torch.tanh(h)
        return h
    

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr
    
    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting
    
    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    
