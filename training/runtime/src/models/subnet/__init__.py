from typing import Dict

from copy import deepcopy

from src.utils.logger import IndentedLog, log_dict_items
from src.utils.registry import (ENCODER_REGISTRY, DECODER_REGISTRY, HYPERENCODER_REGISTRY, 
            HYPERDECODER_REGISTRY, CONTEXTMODEL_REGISTRY, ENTROPYMODEL_REGISTRY, LRP_REGISTRY)


from .vq_estimator import *
from .autoencoder import *
from .entropy_model import *
from .context_model import *
from .vq_fusion_module import *
from .hyperprior import *


def build_subnet(subnet_opt: Dict, subnet_type: str):
    subnet_opt = deepcopy(subnet_opt)
    network_type = subnet_opt.pop('type')
    registry = {
        'encoder': ENCODER_REGISTRY,
        'decoder': DECODER_REGISTRY,
        'hyperencoder': HYPERENCODER_REGISTRY,
        'hyperdecoder': HYPERDECODER_REGISTRY,
        'context_model': CONTEXTMODEL_REGISTRY,
        'entropy_model': ENTROPYMODEL_REGISTRY,
        'residual_predictor': LRP_REGISTRY,
    }[subnet_type]
    subnet = registry.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet

