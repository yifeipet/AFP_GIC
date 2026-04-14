import os.path as osp
from src.utils.misc import import_modules

import_modules('src.models.comp_model', osp.dirname(osp.abspath(__file__)), suffix='_model.py')
