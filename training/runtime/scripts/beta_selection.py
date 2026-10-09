import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from glob import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT_DIR, "scripts"))
from calc_metrics import FIDMetric, PSNRMetric  # noqa: E402

import src  # noqa: F401,E402
from src.models import build_comp_model  # noqa: E402
from src.utils import img_utils  # noqa: E402
from src.utils.logger import get_root_logger  # noqa: E402
from src.utils.options import BaseConfig  # noqa: E402

SEARCH_ERROR_THRESHOLD = 0.001


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
        parser.add_argument("--search_dir", type=str, required=True)
        parser.add_argument("--save_dir", type=str, required=True)
        parser.add_argument("--dataset_root", type=str, required=True)
        parser.add_argument("--beta_vq", type=float, nargs="+", required=True)
        parser.add_argument("--target_rate", type=float, nargs="+", required=True)
        parser.add_argument("--alpha", type=float, default=2.0)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=None)
        parser.add_argument("--keep_recon", action="store_true")
        parser.add_argument("--skip_fid", action="store_true")
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
        out = {
            "real_images": self.transform(img),
            "img_name": os.path.basename(img_path),
        }
        npy_path = img_path.replace(".png", ".npy")
        if os.path.exists(npy_path):
            vq_indices = np.load(npy_path).astype(np.int32)
            out["vq_indices"] = torch.from_numpy(vq_indices).long()
        else:
            out["vq_indices"] = None
        return out


def collate_img_token_batch(batch):
    collated = {
        "real_images": default_collate([b["real_images"] for b in batch]),
        "img_name": [b["img_name"] for b in batch],
    }
    vq_list = [b["vq_indices"] for b in batch]
    collated["vq_indices"] = None if any(v is None for v in vq_list) else default_collate(vq_list)
    return collated


def calc_batch_bpp(y_likelihood: Tensor, z_likelihood: Tensor, num_pixel: int) -> Tensor:
    bit_y = -torch.log(y_likelihood) / np.log(2)
    bit_z = -torch.log(z_likelihood) / np.log(2)
    bit_y = torch.sum(bit_y, dim=tuple(range(1, bit_y.ndim)))
    bit_z = torch.sum(bit_z, dim=tuple(range(1, bit_z.ndim)))
    return (bit_y + bit_z) / num_pixel


def get_rate_summary(img_name: str, bpp: Tensor, num_pixel: int) -> Dict:
    bpp_float = bpp.item()
    return {
        "img_name": img_name.split(".")[0],
        "num_pixel": num_pixel,
        "total_bit": bpp_float * num_pixel,
        "bitrate": bpp_float,
    }


@torch.no_grad()
def save_reconstructions(model, dataloader, save_dir, beta_vq, beta_rate):
    rate_summary_list = []
    for data_dict in tqdm(dataloader, ncols=80):
        out_dict = model.run_model(
            real_images=data_dict["real_images"],
            vq_indices=data_dict.get("vq_indices", None),
            beta_vq=beta_vq,
            beta_rate=beta_rate,
            is_train=False,
        )
        h, w = data_dict["real_images"].size()[-2:]
        batch_bpp = calc_batch_bpp(out_dict["y_likelihood"], out_dict["z_likelihood"], num_pixel=h * w)
        bs = data_dict["real_images"].size(0)
        for i in range(bs):
            img_name = data_dict["img_name"][i]
            img_utils.imwrite(os.path.join(save_dir, img_name), out_dict["fake_images"][i])
            rate_summary_list.append(get_rate_summary(img_name, batch_bpp[i], num_pixel=h * w))

    df = pd.json_normalize(rate_summary_list)
    df.to_csv(os.path.join(save_dir, "_rate_summary.csv"), index=False)
    avg_bpp = float(df["bitrate"].mean())
    with open(os.path.join(save_dir, "_avg_bitrate.json"), "w") as f:
        json.dump({"avg_bpp": avg_bpp}, f)
    return avg_bpp


def main(opt):
    os.makedirs(opt.save_dir, exist_ok=True)
    log_file = os.path.join(opt.save_dir, f'run_{datetime.now().isoformat(timespec="seconds")}.log')
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)

    assert os.path.exists(opt.dataset_root), f'dataset_root "{opt.dataset_root}" does not exist.'
    logger.info(f"dataset_root: {opt.dataset_root}")
    dataset = ImageMaybeTokenDataset(opt.dataset_root)
    num_workers = opt.num_workers
    if num_workers is None:
        num_workers = max(1, min(8, opt.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_img_token_batch,
    )

    model = build_comp_model(opt).to(opt.device)
    model.load_learned_weight(ckpt_path=opt.model_path)
    model.eval()

    psnr_func = PSNRMetric()
    fid_func = None if opt.skip_fid else FIDMetric(opt.device)
    selection_results = []

    for target_rate in opt.target_rate:
        data_list = []
        save_dir = os.path.join(opt.save_dir, f"target_rate_{target_rate}")
        os.makedirs(save_dir, exist_ok=True)

        for beta_vq in opt.beta_vq:
            beta_vq_str = f"{beta_vq:.2f}"
            target_rate_str = f"{target_rate:.3f}"
            bin_search_csv = os.path.join(
                opt.search_dir,
                f"result_beta_vq_{beta_vq_str}_target_rate_{target_rate_str}.csv",
            )
            bin_search_df = pd.read_csv(bin_search_csv).sort_values(by="diff")
            search_result = bin_search_df.iloc[0]
            if search_result["diff"] > SEARCH_ERROR_THRESHOLD:
                logger.warning(
                    f'bpp difference is larger than threshold: {search_result["diff"]} > {SEARCH_ERROR_THRESHOLD}. Skip.'
                )
                continue

            beta_rate = float(search_result["beta_rate"])
            recon_dir = os.path.join(save_dir, f"beta_vq_{beta_vq_str}")
            os.makedirs(recon_dir, exist_ok=True)
            avg_bpp = save_reconstructions(model, dataloader, recon_dir, beta_vq, beta_rate)

            fake_images = sorted(glob(os.path.join(recon_dir, "*.png")))
            real_images = sorted(glob(os.path.join(opt.dataset_root, "*.png")))
            psnr_val = psnr_func.calc_metric(real_images, fake_images)
            fid_val = 0.0 if opt.skip_fid else fid_func.calc_metric(real_images, fake_images)
            score = psnr_val if opt.skip_fid else opt.alpha * psnr_val - fid_val
            data_list.append(
                {
                    "beta_vq": beta_vq,
                    "beta_rate": beta_rate,
                    "bpp": avg_bpp,
                    "psnr": psnr_val,
                    "fid": fid_val,
                    "score": score,
                }
            )

            if not opt.keep_recon:
                shutil.rmtree(recon_dir)

        result_df = pd.json_normalize(data_list)
        result_csv = os.path.join(save_dir, "result.csv")
        if result_df.empty:
            logger.warning(f"no valid beta pair found for target_rate={target_rate}. skip.")
            result_df.to_csv(result_csv, index=False)
            continue
        result_df = result_df.sort_values(by="score", ascending=False)
        result_df.to_csv(result_csv, index=False)
        best_result = result_df.iloc[0]
        logger.info(
            f'target_rate: {target_rate}, selected beta_vq: {best_result["beta_vq"]}, selected beta_rate: {best_result["beta_rate"]}'
        )
        selection_results.append(
            {
                "target_rate": target_rate,
                "selected_beta_vq": best_result["beta_vq"],
                "selected_beta_rate": best_result["beta_rate"],
            }
        )

    pd.json_normalize(selection_results).to_csv(
        os.path.join(opt.save_dir, "beta_selection_results.csv"), index=False
    )


if __name__ == "__main__":
    main(CustomConfig.get_opt())
