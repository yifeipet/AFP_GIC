# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
from typing import Any, Optional

import inspect
import os
import os.path as osp

from src.utils.misc import Color
from src.utils.logger import get_root_logger

class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name: str, obj: Any, filename: str):
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = {
            'obj': obj,
            'filename': filename
        }

    def register(self):
        def deco(func_or_class):
            filename = osp.basename(inspect.stack()[1].filename)
            name = func_or_class.__name__
            self._do_register(name, func_or_class, filename)
            return func_or_class

        return deco

    def get(self, class_name: str, display_name: Optional[str]=None):
        ret = self._obj_map.get(class_name)
        if ret is None:
            raise KeyError(f"No object named '{class_name}' found in '{self._name}' registry!")
        obj, filename = ret['obj'], ret['filename']
        logger = get_root_logger()
        display_name = self._name if display_name is None else display_name
        logger.info(f'{display_name} [{Color.BLUE}{class_name}{Color.RESET}] (from {Color.GREEN}{filename}{Color.RESET}) is build')
        return obj

    def __contains__(self, name: str):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


TRAINER_REGISTRY = Registry('trainer')
OPTIMIZER_REGISTRY = Registry('optimizer')
SCHEDULER_REGISTRY = Registry('scheduler')

MODEL_REGISTRY = Registry('comp_model')
ENCODER_REGISTRY = Registry('encoder')
DECODER_REGISTRY = Registry('decoder')
HYPERENCODER_REGISTRY = Registry('hyperencoder')
HYPERDECODER_REGISTRY = Registry('hyperdecoder')
CONTEXTMODEL_REGISTRY = Registry('context_model')
ENTROPYMODEL_REGISTRY = Registry('entropy_model')
DISCRIMINATOR_REGISTRY = Registry('discriminator')
LRP_REGISTRY = Registry('residual_predictor')

DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')

VQ_ESTIMATOR_REGISTRY = Registry('vq_estimator')
VQ_FUSION_REGISTRY = Registry('vq_fusion')