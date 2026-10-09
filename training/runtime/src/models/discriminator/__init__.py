# from .taming_nlayer_discriminator import *
# from .dual_beta_taming_nlayer_discriminator import *
# from .clic21_gvae_discriminator import *
# from .independent_dual_beta_taming_discriminator import *
# from .oasis_discriminator import *
import os.path as osp
from src.utils.misc import import_modules

from typing import Dict

from os import path as osp
from copy import deepcopy

import torch.nn as nn

from src.utils.logger import log_dict_items
from src.utils.misc import import_modules

import_modules('src.models.discriminator', osp.dirname(osp.abspath(__file__)), suffix='_discriminator.py')

from src.utils.registry import DISCRIMINATOR_REGISTRY

def build_discriminator(discriminator_opt: Dict) -> nn.Module:
    subnet_opt = deepcopy(discriminator_opt)
    network_type = subnet_opt.pop('type')
    subnet = DISCRIMINATOR_REGISTRY.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet


