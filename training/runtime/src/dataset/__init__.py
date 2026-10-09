from typing import Dict

from copy import deepcopy

from torch.utils.data import Dataset

from .kodak_dataset import *
from .openimage_dataset import *
# from .clic_dataset import *
# from .dataset_path import DatasetPath
# from .openimage_pre_vq_encode_dataset import *

from src.utils.logger import IndentedLog, get_root_logger, log_dict_items

dataset_list = ['Kodak', 'OpenImage']

def cvt_dataset_name(dataset_name: str) -> str:
    _dataset_name = dataset_name.lower()
    for name in dataset_list:
        if name.lower() == _dataset_name:
            return name
    raise ValueError(f'Invalid dataset_name: "{dataset_name}".')

def build_dataset(dataset_opt: Dict, is_train: bool=True) -> Dataset:
    dataset_opt = deepcopy(dataset_opt)
    dataset_name = cvt_dataset_name(dataset_opt.pop('name')) # ex) COCO
    dataset_type = dataset_opt.pop('type') # ex) ImageDataset
    registry_key = dataset_name + dataset_type

    dataset_opt['is_train'] = is_train
    assert 'root_dir' in dataset_opt
    # if 'root_dir' not in dataset_opt:
        # dataset_opt['root_dir'] = DatasetPath.db_root_dir(dataset_name.lower())

    dataset = DATASET_REGISTRY.get(registry_key)(**dataset_opt)
    log_dict_items(dataset_opt, level='DEBUG', indent=True)
    log_dict_items({'len(dataset)': len(dataset)}, level='INFO', indent=True)
    assert len(dataset) > 0, 'len(dataset) should be >0.'
    return dataset