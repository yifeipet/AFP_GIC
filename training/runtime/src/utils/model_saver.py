from typing import Dict, List, Optional, Union

import os
import torch

from .logger import get_root_logger
from .path import PathHandler

class Saver(object):
    def __init__(
        self,
        ckpt_root: str,
        exp: str,
        save_step: int,
        keep_step: Union[int, List],
        staging_root: Optional[str] = None,
        keep_previous: bool = False,
        sync_cuda: bool = False,
    ):
        self.path_hander = PathHandler(ckpt_root, exp)
        self.save_step = save_step
        self.keep_step = keep_step
        self.staging_dir = (
            os.path.join(os.path.abspath(staging_root), exp)
            if staging_root
            else None
        )
        self.keep_previous = keep_previous
        self.sync_cuda = sync_cuda
        if isinstance(keep_step, list):
            self.keep_step = set(keep_step)

    def _should_keep(self, itr: int):
        if isinstance(self.keep_step, int):
            return (itr % self.keep_step == 0)
        return (itr in self.keep_step)

    def save(self, network_dict: Dict, save_label: str, current_iter: int, keep: bool) -> None:
        """
        Save model to `{save_dir}/{save_label}_iter{current_iter}.pth.tar`

        Args:
            network_dict (Dict): {"key": model} e.g., {"comp_model": comp_model}
            save_label (str): "comp_model", "discriminator", "training_state", etc.
            current_iter (int):
            keep (bool): if False, delete the model saved at `current_iter - save_step`
        """
        self._save_network(network_dict, save_label, current_iter)
        fallback_steps = 2 if self.keep_previous else 1
        delete_iter = current_iter - fallback_steps * self.save_step
        if delete_iter <= 0:
            return
        if not(keep) or not(self._should_keep(delete_iter)):
            self._delete_network(save_label, delete_iter)

    def _save_network(self, network_dict: Dict, save_label: str, current_iter: int) -> None:
        state_dict = {'iter': current_iter}
        for key, network in network_dict.items():
            state_dict[key] = network.state_dict()
        save_path = self._get_save_path(save_label, current_iter)
        logger = get_root_logger()
        logger.debug(f'saving {save_label} iter{current_iter} ("{save_path}")')
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.staging_dir is None:
            torch.save(state_dict, save_path)
            return
        self._staged_atomic_save(state_dict, save_path, logger)

    def _staged_atomic_save(self, state_dict: Dict, save_path: str, logger) -> None:
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        file_name = os.path.basename(save_path)
        staging_path = os.path.join(self.staging_dir, f".{file_name}.tmp")
        partial_path = f"{save_path}.part"
        for stale_path in (staging_path, partial_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

        try:
            logger.debug(f'staging checkpoint locally ("{staging_path}")')
            torch.save(state_dict, staging_path)
            self._fsync_file(staging_path)
            staging_size = os.path.getsize(staging_path)
            if staging_size <= 0:
                raise RuntimeError(f"empty staged checkpoint: {staging_path}")

            with open(staging_path, "rb") as source, open(partial_path, "wb") as target:
                while True:
                    chunk = source.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())

            partial_size = os.path.getsize(partial_path)
            if partial_size != staging_size:
                raise RuntimeError(
                    f"checkpoint copy size mismatch: staged={staging_size}, "
                    f"partial={partial_size}"
                )
            os.replace(partial_path, save_path)
            logger.debug(
                f'committed checkpoint atomically ({staging_size} bytes, "{save_path}")'
            )
        finally:
            for temporary_path in (staging_path, partial_path):
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

    @staticmethod
    def _fsync_file(path: str) -> None:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())

    def _delete_network(self, save_label: str, delete_iter: int) -> None:
        delete_path = self._get_save_path(save_label, delete_iter)
        logger = get_root_logger()
        logger.debug(f'deleting {save_label} iter{delete_iter} ("{delete_path}")')
        if os.path.exists(delete_path):
            os.remove(delete_path)
        else:
            logger.warning(f'Tried to delete checkpoint "{save_label}" iter{delete_iter}, but "{delete_path}" does not exist.')

    @staticmethod
    def get_save_path(save_dir: str, save_label: str, itr: int) -> str:
        itr_str = PathHandler.iter2str(itr)
        return os.path.join(save_dir, f'{save_label}_iter{itr_str}.pth.tar')

    def _get_save_path(self, save_label: str, itr: int) -> str:
        return self.path_hander.get_ckpt_path(save_label, itr)
