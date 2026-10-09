from typing import Dict

from os import path as osp
from copy import deepcopy

import torch.nn as nn

from src.utils.logger import log_dict_items
from src.utils.misc import import_modules

from src.utils.registry import VQ_ESTIMATOR_REGISTRY

import_modules('src.models.subnet.vq_estimator', osp.dirname(osp.abspath(__file__)), suffix='estimator.py')


def build_vq_estimator(vq_estimator_opt: Dict) -> nn.Module:
    subnet_opt = deepcopy(vq_estimator_opt)
    network_type = subnet_opt.pop('type')
    subnet = VQ_ESTIMATOR_REGISTRY.get(network_type)(**subnet_opt)
    log_dict_items(subnet_opt, level='DEBUG', indent=True)
    return subnet
