import argparse
import logging
import os
from datetime import datetime
from glob import glob
from itertools import product
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

import src  # noqa: F401
from src.models import build_comp_model
from src.utils.logger import get_root_logger
from src.utils.options import BaseConfig

MEMO_DICT = {}


class CustomConfig(BaseConfig):
    @classmethod
    def get_opt(cls) -> "CustomConfig":
        arg_dict = cls.arg_parse()
        filename = arg_dict["config_path"]
        cfg_dict, cfg_text, loaded_yamls = cls._file2dict_yaml(filename)
        cfg_dict["loaded_yamls"] = loaded_yamls
        arg_dict = cls._merge_a_into_b(arg_dict, cfg_dict)
        return cls(arg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def arg_parse() -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str)
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--save_dir", type=str, required=True)
        parser.add_argument("--dataset_root", type=str, required=True)
        parser.add_argument("--beta_vq", type=float, nargs="+", required=True)
        parser.add_argument("--target_rate", type=float, nargs="+", required=True)
        parser.add_argument("--max_beta_rate", type=float, required=True)
        parser.add_argument("--error_delta", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=None)
        parser.add_argument("--max_run_cnt", type=int, default=10)
        parser.add_argument("-d", "--device", type=str, default="cuda:0")
        return vars(parser.parse_args())


class ImageMaybeTokenDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.img_list = sorted(glob(os.path.join(root_dir, "*.png")))
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int) -> Dict:
        img_path = self.img_list[index]
        img = Image.open(img_path).convert("RGB")
        out = {"real_images": self.transform(img)}
        npy_path = img_path.replace(".png", ".npy")
        if os.path.exists(npy_path):
            vq_indices = np.load(npy_path).astype(np.int32)
            out["vq_indices"] = torch.from_numpy(vq_indices).long()
        else:
            out["vq_indices"] = None
        return out


def collate_img_token_batch(batch):
    collated = {"real_images": default_collate([b["real_images"] for b in batch])}
    vq_list = [b["vq_indices"] for b in batch]
    collated["vq_indices"] = None if any(v is None for v in vq_list) else default_collate(vq_list)
    return collated


def build_pretrained_model(opt: CustomConfig):
    model = build_comp_model(opt).to(opt.device)
    model.load_learned_weight(ckpt_path=opt.model_path)
    model.eval()
    return model


def build_dataloader(opt: CustomConfig):
    logger = get_root_logger()
    dataset_root = opt.dataset_root
    assert os.path.exists(dataset_root), f'dataset_root "{dataset_root}" does not exist.'
    logger.info(f"dataset_root: {dataset_root}")
    dataset = ImageMaybeTokenDataset(dataset_root)
    num_workers = opt.num_workers
    if num_workers is None:
        num_workers = min(8, opt.batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_img_token_batch,
    )
    return dataloader


@torch.no_grad()
def run_one_search(model, dataloader, beta_rate: float, beta_vq: float) -> float:
    total_bpp, image_count = 0.0, 0
    for data_dict in tqdm(dataloader, ncols=80, leave=False):
        vq_indices: Optional[torch.Tensor] = data_dict.get("vq_indices", None)
        out_dict = model.run_model(
            real_images=data_dict["real_images"],
            vq_indices=vq_indices,
            beta_rate=beta_rate,
            beta_vq=beta_vq,
            is_train=False,
        )
        batch_size = data_dict["real_images"].shape[0]
        total_bpp += out_dict["bpp"].item() * batch_size
        image_count += batch_size
    if image_count == 0:
        raise ValueError("The selection dataset is empty.")
    return total_bpp / image_count


def memo_dict_key(beta_vq: float, beta_rate: float) -> str:
    return f"{beta_vq:.4f}-{beta_rate:.4f}".replace(".", "_")


def run(opt, model, dataloader, target_rate, beta_vq, logger):
    data_list = []
    beta_rate_min = 0.0
    beta_rate_max = opt.max_beta_rate
    run_cnt = 0

    while True:
        run_cnt += 1
        beta_rate = round((beta_rate_min + beta_rate_max) / 2.0, 3)
        memo_key = memo_dict_key(beta_vq, beta_rate)
        logger.info(
            f"run_cnt {run_cnt:2} | min: {beta_rate_min}, max: {beta_rate_max}, beta_rate: {beta_rate}"
        )

        if memo_key in MEMO_DICT:
            logger.info(f"run_cnt {run_cnt:2} |   beta_rate: {beta_rate} is already searched")
            avg_bpp = MEMO_DICT[memo_key]
        else:
            logger.info(f"run_cnt {run_cnt:2} |   beta_rate: {beta_rate} is not searched yet. Start running...")
            avg_bpp = run_one_search(model, dataloader, beta_rate, beta_vq)
            MEMO_DICT[memo_key] = avg_bpp

        diff = abs(avg_bpp - target_rate)
        data_list.append(
            {
                "run_cnt": run_cnt,
                "beta_vq": beta_vq,
                "beta_rate": beta_rate,
                "avg_bpp": avg_bpp,
                "diff": diff,
            }
        )
        logger.info(f"run_cnt {run_cnt:2} |   avg_bpp: {avg_bpp:.5f}, diff: {diff:.5f}")

        if diff <= opt.error_delta:
            break
        if avg_bpp > target_rate:
            beta_rate_min = beta_rate
        else:
            beta_rate_max = beta_rate

        if run_cnt >= opt.max_run_cnt:
            logger.warning(f"reached max run count: {opt.max_run_cnt}")
            break

    df = pd.json_normalize(data_list)
    df = df.sort_values("diff").reset_index(drop=True)
    return df


def main() -> None:
    opt = CustomConfig.get_opt()
    os.makedirs(opt.save_dir, exist_ok=True)
    log_file = os.path.join(opt.save_dir, f'run_{datetime.now().isoformat(timespec="seconds")}.log')
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)

    model = build_pretrained_model(opt)
    dataloader = build_dataloader(opt)

    total = len(opt.beta_vq) * len(opt.target_rate)
    for i, (beta_vq, target_rate) in enumerate(product(opt.beta_vq, opt.target_rate), start=1):
        logger.info(f"**** {i}/{total} ****")
        logger.info(f"beta_vq: {beta_vq}, target_rate: {target_rate}")
        csv_path = os.path.join(
            opt.save_dir,
            f"result_beta_vq_{beta_vq:.2f}_target_rate_{target_rate:.3f}.csv",
        )
        df = run(opt, model, dataloader, target_rate, beta_vq, logger)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
