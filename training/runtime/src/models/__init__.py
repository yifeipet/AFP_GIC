import importlib
from copy import deepcopy
from os import path as osp

from src.utils.registry import MODEL_REGISTRY
from src.utils.logger import get_root_logger
from .comp_model import *
from .subnet import *
from .discriminator import *

__all__ = ['build_comp_model']


def build_comp_model(opt):
    """Build model from options.
    Args:
        opt (Config): Configuration. It must contain model.type
    """
    opt = deepcopy(opt)
    if opt.get('model'):
        model_opt = deepcopy(opt['model'])
        model_type = model_opt.pop('type')
        model = MODEL_REGISTRY.get(model_type)(opt, **model_opt)
        return model

    raise ValueError('"model_type" key is not supported. Please use trainer.type')


def build_trained_comp_model(opt, ckpt_path: str):
    model = build_comp_model(opt)
    model.load_learned_weight(ckpt_path=ckpt_path)
    return model


