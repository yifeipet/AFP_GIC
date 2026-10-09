import json
import os
from typing import Dict, Union, Optional, Iterable

import torch
import wandb
from tqdm import tqdm

from src.models import build_comp_model
from src.dataset import build_dataset
from src.utils.logger import AvgMeter, get_root_logger, CSVLogger, bolded_log, log_dict_items
from src.utils.timer import Timer
from src.utils.model_saver import Saver
from src.utils.path import PathHandler


class BaseTrainer(object):
    def __init__(self, opt) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.opt = opt
        self.device = opt.device
        self.logger = get_root_logger()
        self.set_models()
        self.set_optimizer_scheduler()
        self.set_losses()
        self.set_dataloader()
        self.set_csv_loggers()
        self.use_wandb = opt.get('use_wandb', False)
        if self.use_wandb:
            self.init_wandb()
        self.loss_recorder = AvgMeter()
        self.time_recorder = Timer(start_iter=opt.start_iter, end_iter=opt.total_iter)
        self.path_handler = PathHandler(self.opt.ckpt_root, self.opt.exp)
        self.model_saver = Saver(
            self.opt.path.ckpt_root,
            self.opt.exp,
            self.opt.save_step,
            self.opt.keep_step,
            staging_root=self.opt.get("checkpoint_staging_root", None),
            keep_previous=self.opt.get("checkpoint_keep_previous", False),
            sync_cuda=self.opt.get("checkpoint_sync_cuda", False),
        )
        self._init_best_checkpoint_state()
        # Resume training from the same experiment directory. When resuming, do not
        # also apply stage-initialization checkpoint loading, otherwise the resumed
        # state would be overwritten by the upstream stage checkpoint.
        if self.opt.start_iter > 0:
            self.load_checkpoint(self.opt.exp, self.opt.start_iter)
        # Start from other job_dir only for fresh runs.
        elif self.opt.get('load_checkpoint', None):
            _exp = self.opt.load_checkpoint.pop('exp')
            _itr = self.opt.load_checkpoint.pop('iter')
            self.load_checkpoint(_exp, _itr, **self.opt.load_checkpoint)
        if self.opt.dry_run:
            self.print_models()
            exit()

    ########## Initialization #####################################################################

    def set_models(self) -> None:
        bolded_log('Model', level="INFO", new_line=True)
        self._set_models()

    def _set_models(self) -> None:
        self.comp_model = build_comp_model(self.opt).to(self.device)
        if self.opt.get('pretrained_weight_path', None):
            self.comp_model.load_learned_weight(self.opt.pretrained_weight_path)
        self.comp_model.train()

    def set_optimizer_scheduler(self) -> None:
        bolded_log('Optimizers & Schedulers', level="INFO", new_line=True)
        self._set_optimizer_scheduler()

    def _set_optimizer_scheduler(self) -> None:
        raise NotImplementedError()
    
    def set_losses(self) -> None:
        bolded_log('Loss functions', level="INFO", new_line=True)
        self._set_losses()

    def _set_losses(self) -> None:
        raise NotImplementedError()

    def set_dataloader(self) -> None:
        bolded_log('Dataloader', level="INFO", new_line=True)
        train_dataset = build_dataset(self.opt.dataset.train_dataset, is_train=True)
        num_workers = self.opt.get('num_workers', 8)
        log_dict_items({'batch_size': self.opt.dataset.batch_size, 'num_workers': num_workers}, level='INFO', indent=True)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.opt.dataset.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )
        eval_dataset = build_dataset(self.opt.dataset.eval_dataset, is_train=False)
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=1)

    def set_csv_loggers(self) -> None:
        self.train_logger = CSVLogger(log_path=self.opt.path.log_loss_path, resume=(self.opt.start_iter > 0))
        self.eval_logger = CSVLogger(log_path=self.opt.path.log_eval_path, resume=(self.opt.start_iter > 0))

    def init_wandb(self) -> None:
        if self.opt.get('wandb_dryrun'):
            os.environ['WANDB_MODE'] = 'dryrun'
        wandb_root = self.opt.wandb_root
        wandb.init(
            dir=wandb_root,
            project=self.opt.project_name, 
            name=self.opt.exp, 
            id=self.opt.exp, 
            config=self.opt._cfg_dict,
            resume=(self.opt.start_iter > 0),
            notes=self.opt.get('comment', None),
            tags=self.opt.get('wandb_tag', []),
            settings=wandb.Settings(start_method="thread"),
        )

    def load_checkpoint(self, exp: str, itr: int, **kwargs) -> None:
        bolded_log('Load checkpoint', level="INFO", new_line=True)
        log_dict_items(dict(exp=exp, iter=itr, **kwargs), level='INFO', indent=False)
        self._load_checkpoint(exp, itr, **kwargs)

    def _load_checkpoint(self, exp: str, itr: int, **kwargs) -> None:
        raise NotImplementedError()

    def print_models(self) -> None:
        bolded_log('Print Models', level="DEBUG", new_line=True)
        self._print_models()

    def _print_models(self) -> None:
        bolded_log('comp_model', level="DEBUG", new_line=True, prefix='', suffix='')
        self.logger.debug(str(self.comp_model))

    ########## Main Loop ######################################################################

    def train_data_generator(self, dataloader: torch.utils.data.DataLoader, start_itr: int, end_itr: int, use_tqdm: bool=True):
        data_iter = iter(dataloader)
        range_ = tqdm(range(start_itr, end_itr), ncols=60) if use_tqdm else range(start_itr, end_itr)
        for i in range_:
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data = next(data_iter)
            yield i+1, data

    def train_loop(self) -> None:
        """Main training loop
        """
        bolded_log('train_loop start', new_line=True)
        self.time_recorder.start()
        for itr, data in self.train_data_generator(
            self.train_loader, self.opt.start_iter, self.opt.total_iter, use_tqdm=not(self.use_wandb)):

            loss_dict = self.optimize_parameters(itr, data)
            if loss_dict is not None:
                self.update_loss_recorder(loss_dict)

            if itr % self.opt.log_step == 0:
                self.log_train_loss(itr)

            if itr % self.opt.eval_step == 0:
                self.validation(itr)

            if itr % self.opt.save_step == 0:
                self.save(itr)

            if itr % self.opt.time_step == 0:
                self.log_time(itr)

    def optimize_parameters(self, itr: int, data: Dict) -> Dict:
        raise NotImplementedError()

    def update_loss_recorder(self, loss_dict: Dict) -> None:
        # TODO: gather loss_dict from all ranks
        mean_dic = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                mean_dic[k] = v.mean().item()
            else:
                mean_dic[k] = v
        self.loss_recorder.update(mean_dic)

    def validation(self, current_iter: int) -> None:
        torch.backends.cudnn.benchmark = False
        self.comp_model.eval()
        eval_dict = self._validation()
        self._log_validation_results(current_iter, eval_dict)
        self._maybe_save_best_checkpoint(current_iter, eval_dict)
        self.comp_model.train()
        torch.backends.cudnn.benchmark = True

    def _validation(self) -> Dict:
        raise NotImplementedError()

    def _log_validation_results(self, current_iter: int, eval_dict: Dict) -> None:
        # print log
        log_str = f'validation iter{current_iter}\n'
        for key, val in eval_dict.items():
            log_str += f'\t {key}: {val:.4f}\n'
        self.logger.debug(log_str)
        # CSV log
        log_dict = {'iter': current_iter}
        log_dict.update(eval_dict)
        self.eval_logger.update(log_dict)
        # wandb log
        if self.use_wandb:
            wandb_dict = {f'eval_{k}':v for k, v in eval_dict.items() if k not in ['iter']}
            wandb_dict['iter'] = current_iter
            wandb.log(wandb_dict)

    def _init_best_checkpoint_state(self) -> None:
        cfg = self.opt.get("best_checkpoint", {})
        self.best_checkpoint_enabled = bool(cfg.get("enabled", False))
        self.best_checkpoint_metric = cfg.get("metric", None)
        self.best_checkpoint_mode = cfg.get("mode", "max")
        self.best_checkpoint_label = cfg.get("label", None)
        self.best_checkpoint_value = None
        self.best_checkpoint_iter = None
        self.best_checkpoint_meta_path = os.path.join(
            self.opt.path.model_dir, "best_checkpoint_state.json"
        )
        if not self.best_checkpoint_enabled:
            return
        if self.best_checkpoint_mode not in {"max", "min"}:
            raise ValueError(
                f'best_checkpoint.mode must be "max" or "min", '
                f'but got "{self.best_checkpoint_mode}"'
            )
        if self.best_checkpoint_label is None:
            metric_name = str(self.best_checkpoint_metric or "metric").replace(".", "_")
            self.best_checkpoint_label = f"best_{metric_name}"
        if os.path.exists(self.best_checkpoint_meta_path):
            with open(self.best_checkpoint_meta_path, "r") as f:
                state = json.load(f)
            self.best_checkpoint_value = state.get("best_value", None)
            self.best_checkpoint_iter = state.get("best_iter", None)
            self.logger.info(
                "loaded best checkpoint state: "
                f"metric={state.get('metric')} "
                f"best_value={self.best_checkpoint_value} "
                f"best_iter={self.best_checkpoint_iter}"
            )

    def _is_better_metric(self, metric_value: float) -> bool:
        if self.best_checkpoint_value is None:
            return True
        if self.best_checkpoint_mode == "max":
            return metric_value > self.best_checkpoint_value
        return metric_value < self.best_checkpoint_value

    def _maybe_save_best_checkpoint(self, current_iter: int, eval_dict: Dict) -> None:
        if not self.best_checkpoint_enabled:
            return
        metric_name = self.best_checkpoint_metric
        if metric_name not in eval_dict:
            self.logger.warning(
                "best checkpoint metric is not found in eval_dict: "
                f"{metric_name}"
            )
            return
        metric_value = float(eval_dict[metric_name])
        if not self._is_better_metric(metric_value):
            return

        self.best_checkpoint_value = metric_value
        self.best_checkpoint_iter = current_iter
        payloads = self._get_best_checkpoint_payloads()
        for save_label, network_dict in payloads.items():
            self._save_best_checkpoint_payload(
                save_label,
                network_dict,
                current_iter,
                metric_name,
                metric_value,
            )
        state = {
            "metric": metric_name,
            "mode": self.best_checkpoint_mode,
            "label": self.best_checkpoint_label,
            "best_value": metric_value,
            "best_iter": current_iter,
            "eval_dict": {},
        }
        for key, val in eval_dict.items():
            scalar = self._to_log_scalar(val)
            if scalar is not None:
                state["eval_dict"][key] = scalar
                continue
            try:
                state["eval_dict"][key] = float(val)
            except (TypeError, ValueError):
                state["eval_dict"][key] = val
        with open(self.best_checkpoint_meta_path, "w") as f:
            json.dump(state, f, indent=2)
        self.logger.info(
            "saved best checkpoint: "
            f"metric={metric_name} value={metric_value:.6f} iter={current_iter}"
        )

    def _get_best_checkpoint_payloads(self) -> Dict[str, Dict]:
        return {"comp_model": {"comp_model": self.comp_model}}

    def _save_best_checkpoint_payload(
        self,
        save_label: str,
        network_dict: Dict,
        current_iter: int,
        metric_name: str,
        metric_value: float,
    ) -> None:
        state_dict = {
            "iter": current_iter,
            "best_metric": metric_name,
            "best_metric_value": metric_value,
        }
        for key, network in network_dict.items():
            state_dict[key] = network.state_dict()
        save_path = os.path.join(
            self.opt.path.model_dir, f"{save_label}_{self.best_checkpoint_label}.pth.tar"
        )
        self.logger.info(f'saving best checkpoint "{save_path}"')
        torch.save(state_dict, save_path)

    def log_train_loss(self, current_iter: int) -> None:
        avg_dic = self.loss_recorder.get_avg_values()
        # print log
        if self.opt.get('debug'):
            log_str = f'loss iter{current_iter}: '
            for k, v in avg_dic.items():
                log_str += f'{k}: {v:.3f} '
            self.logger.debug(log_str)
        # CSV log
        log_dict = {'iter': current_iter}
        log_dict.update(avg_dic)
        self.train_logger.update(log_dict)

        if self.use_wandb and (current_iter % self.opt.wandb_loss_step) == 0:
            wandb.log(log_dict)
        self.loss_recorder.reset()

    def log_time(self, current_iter: int) -> None:
        if self.use_wandb:
            time_dict = self.time_recorder.get_time_stat(current_iter)
            time_dict['iter'] = current_iter
            wandb.log(time_dict)

    def save(self, current_iter: int) -> None:
        raise NotImplementedError()

    ########## Utility functions ######################################################################

    @staticmethod
    def update_learning_rate(optimizer, new_lr: float):
        """update learning rate manually  
        TODO: update learning rate for each param_group

        Args:
            optimizer (Optimizer): _description_
            new_lr (float): _description_
        """
        assert isinstance(new_lr, float), f'only float lr is supported, but got {type(new_lr)}'
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    @staticmethod
    def check_loss_nan_inf(loss_total) -> Union[str, bool]:
        """Return False if loss is normal, otherwise return the state of loss
        """
        if loss_total.isnan().any():
            return 'nan'
        elif loss_total.isinf().any():
            return 'inf'
        elif loss_total > 10000:
            return 'huge'
        return False

    @staticmethod
    def _to_log_scalar(val) -> Optional[float]:
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return None
            if torch.isnan(val).all() or torch.isinf(val).all():
                return None
            return val.detach().float().mean().item()
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _get_batch_paths(self, data_dict: Optional[Dict], max_items: int = 8):
        if not data_dict:
            return []
        img_paths = data_dict.get("img_path")
        if img_paths is None:
            return []
        if isinstance(img_paths, str):
            return [img_paths]
        if isinstance(img_paths, Iterable):
            paths = []
            for item in img_paths:
                if isinstance(item, str):
                    paths.append(item)
                else:
                    paths.append(str(item))
                if len(paths) >= max_items:
                    break
            return paths
        return [str(img_paths)]

    def log_loss_anomaly(
        self,
        current_iter: int,
        loss_anomaly: str,
        loss_total,
        log_dict: Optional[Dict] = None,
        data_dict: Optional[Dict] = None,
    ) -> None:
        loss_total_scalar = self._to_log_scalar(loss_total)
        if loss_total_scalar is None:
            self.logger.warning(
                f"iter{current_iter}: skipped because loss is {loss_anomaly}"
            )
        else:
            self.logger.warning(
                f"iter{current_iter}: skipped because loss is {loss_anomaly} "
                f"(loss_total={loss_total_scalar:.6f})"
            )
        if log_dict:
            parts = []
            for key, val in log_dict.items():
                scalar = self._to_log_scalar(val)
                if scalar is not None:
                    parts.append(f"{key}={scalar:.6f}")
            if parts:
                self.logger.warning(
                    f"iter{current_iter}: anomaly loss breakdown: " + ", ".join(parts)
                )
        batch_paths = self._get_batch_paths(data_dict)
        if batch_paths:
            self.logger.warning(
                f"iter{current_iter}: anomaly batch paths: " + " | ".join(batch_paths)
            )
