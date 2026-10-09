import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY
from src.models.subnet.autoencoder.elic_autoencoder import ElicDecoder

@DECODER_REGISTRY.register()
class ElicFeatDecoder(ElicDecoder):
    def __init__(
        self,
        feat_layer_name: str,
        in_ch: int = 192,
        out_ch: int = 3,
        main_ch: int = 192,
        block_mid_ch: int = 192,
        num_blocks: int = 3,
        use_tanh: bool = True,
        pixel_shuffle: bool = False,
        res_in_res: bool = False,
    ):
        super().__init__(
            in_ch,
            out_ch,
            main_ch,
            block_mid_ch,
            num_blocks,
            use_tanh,
            pixel_shuffle,
            res_in_res,
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

    def get_interm_feat(self, x: Tensor) -> Tensor:
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)

            if layer_name == self.feat_layer:
                return x
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = None
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)

            if layer_name == self.feat_layer:
                feat = x
        assert feat is not None

        if self.use_tanh:
            x = torch.tanh(x)

        return x, feat
    


@DECODER_REGISTRY.register()
class ElicFeatFusionDecoder(ElicFeatDecoder):
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
    ):
        super().__init__(
            feat_layer_name,
            in_ch=in_ch,
            out_ch=out_ch,
            main_ch=main_ch,
            use_tanh=use_tanh,
            block_mid_ch=block_mid_ch,
            num_blocks=num_blocks,
            pixel_shuffle=pixel_shuffle,
            res_in_res=res_in_res
        )
        self.fusion_layer_dict = fusion_layer_dict
        for k, v in self.fusion_layer_dict.items():
            assert k in self.layer_names

    def get_feats(self, x):
        fusion_feat_dict = {}
        query_layer_list = list(self.fusion_layer_dict.keys())
        feat_1 = None

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)

            if layer_name == self.feat_layer:
                feat_1 = x

            if layer_name in query_layer_list:
                k = self.fusion_layer_dict[layer_name]
                fusion_feat_dict[k] = x

            if len(fusion_feat_dict) == len(query_layer_list):
                break

        return feat_1, fusion_feat_dict