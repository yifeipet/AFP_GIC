from typing import Dict, List

from copy import deepcopy

import torch

from src.utils.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from src.utils.logger import get_root_logger, IndentedLog, log_dict_items

from torch.optim.lr_scheduler import _LRScheduler, ChainedScheduler, SequentialLR


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, warmup_factor, last_epoch=-1) -> None:
        self.total_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_iters:
            alpha = float(self.last_epoch) / self.total_iters
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        return self.base_lrs


class LinearWarmupMultiStepLR(SequentialLR):
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        warmup_factor: float,
        milestones: List[int],
        gamma: float,
        last_epoch: int=-1,
    ) -> None:
        """Apply MultiStepLR after linear warmup

        Args:
            optimizer (_type_): _description_
            warmup_iters (int): _description_
            warmup_factor (float): _description_
            milestones (List[int]): _description_
            gamma (float): _description_
            last_epoch (int, optional): _description_. Defaults to -1.
        """
        assert warmup_iters < milestones[0], f"warmup_iters={warmup_iters} must be smaller than milestones[0]={milestones[0]}"
        milestones = [m - warmup_iters for m in milestones]
        warmup_scheduler = LinearWarmupScheduler(
            optimizer, warmup_iters, warmup_factor, last_epoch
        )
        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma, last_epoch
        )
        schedulers = [warmup_scheduler, multistep_scheduler]
        super().__init__(optimizer, schedulers=schedulers, milestones=[warmup_iters])


def register_optimizers() -> None:
    OPTIMIZER_REGISTRY.register()(torch.optim.Adam)
    OPTIMIZER_REGISTRY.register()(torch.optim.SGD)


def register_schedulers() -> None:
    SCHEDULER_REGISTRY.register()(torch.optim.lr_scheduler.MultiStepLR)
    SCHEDULER_REGISTRY.register()(LinearWarmupScheduler)
    SCHEDULER_REGISTRY.register()(LinearWarmupMultiStepLR)


register_optimizers()
register_schedulers()


def get_params_list(parameters_dict, paramwise_opt, base_lr):
    params = []
    all_keys = set(list(parameters_dict.keys()))
    sorted_keys = sorted(list(parameters_dict.keys()))

    for _opt in paramwise_opt:
        queries = _opt["keys"]
        lr_mult = _opt["lr_mult"]  # _opt.get('lr_mult', 1.)
        _params = []
        for k in sorted_keys:
            hit = [(q in k) for q in queries]
            if any(hit):
                all_keys.remove(k)
                v = parameters_dict[k]
                if v.requires_grad:
                    _params.append(v)
        params.append({"params": _params.copy(), "lr": lr_mult * base_lr})

    _params = []
    for k in sorted_keys:
        if k not in all_keys:
            continue
        v = parameters_dict[k]
        if v.requires_grad:
            _params.append(v)
    params.append({"params": _params.copy()})
    return params


def build_optimizer(parameters_dict: Dict, optimizer_opt: Dict):
    optimizer_opt = deepcopy(optimizer_opt)
    _optimizer_opt = deepcopy(optimizer_opt)  ## for logging
    optimizer_type = optimizer_opt.pop("type")
    paramwise_opt = optimizer_opt.pop("paramwise_opt", [])
    if paramwise_opt:
        params = get_params_list(
            parameters_dict, paramwise_opt, optimizer_opt["lr"]
        )
    else:
        params = (v for v in parameters_dict.values() if v.requires_grad)

    log_dict_items(_optimizer_opt, level="DEBUG", indent=True)
    optimizer = OPTIMIZER_REGISTRY.get(optimizer_type)(
        params=params, **optimizer_opt
    )
    return optimizer


def build_scheduler(optimizer, scheduler_opt: Dict):
    scheduler_opt = deepcopy(scheduler_opt)
    scheduler_type = scheduler_opt.pop("type")
    log_dict_items(scheduler_opt, level="DEBUG", indent=True)
    scheduler = SCHEDULER_REGISTRY.get(scheduler_type)(
        optimizer, **scheduler_opt
    )
    return scheduler
