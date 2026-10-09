"""
Create an OpenImage validation dataset for Stage 2 beta-selection.

For VQGAN-style flows, this script can optionally also save offline VQ indices.
For AdaCode flows, omit --vqgan_type and it will save only cropped PNGs.
"""

import argparse
import os
import random
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from tqdm import tqdm


def get_vqgan_config(vqgan_type: str):
    if vqgan_type == "f8-n256":
        return dict(
            embed_dim=4,
            n_embed=256,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=4,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[32],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    if vqgan_type == "f8":
        return dict(
            embed_dim=4,
            n_embed=16384,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=4,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[32],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    if vqgan_type == "f16":
        return dict(
            embed_dim=8,
            n_embed=16384,
            monitor=None,
            ddconfig=dict(
                double_z=False,
                z_channels=8,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[1, 1, 2, 2, 4],
                num_res_blocks=2,
                attn_resolutions=[16],
                dropout=0.0,
            ),
            lossconfig=dict(target="torch.nn.Identity"),
        )
    raise ValueError(vqgan_type)


def build_vqgan(vq_config: dict, weight_path: str, device: str):
    import ldm
    from ldm.models.autoencoder import VQModelInterface

    _ = ldm
    vae = VQModelInterface(**vq_config)
    state_dict = torch.load(weight_path, map_location=device)["state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss.")}
    vae.load_state_dict(state_dict)
    vae.quantize.sane_index_shape = True
    vae.eval()
    return vae.to(device)


def read_img(img_path: str) -> Image.Image:
    return Image.open(img_path).convert("RGB")


@torch.no_grad()
def vqgan_encode(vae, img_batch: Tensor) -> Tensor:
    z = vae.encode(img_batch)
    qz, _, (_, _, indices) = vae.quantize(z)
    _ = qz
    return indices


def _resolve_image_dir(openimage_root: str) -> str:
    candidates = [
        os.path.join(openimage_root, "train_0"),
        os.path.join(openimage_root, "validation"),
        openimage_root,
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"cannot find image dir under {openimage_root}")


def _save_dir_name(vqgan_type: Optional[str]) -> str:
    if vqgan_type is None:
        return "img_only"
    return {"f8-n256": "vq_f8_n256", "f16": "vq_f16", "f8": "vq_f8"}[vqgan_type]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_dir = _resolve_image_dir(args.openimage_root)
    save_root = args.save_root

    patch_size = 256
    min_short_length = 256
    num_images = args.num_img

    img_path_list = glob(os.path.join(img_dir, "*.jpg"))
    if not img_path_list:
        img_path_list = glob(os.path.join(img_dir, "*.png"))
    img_path_list.sort()
    np.random.shuffle(img_path_list)

    save_dir = os.path.join(
        save_root,
        f"{_save_dir_name(args.vqgan_type)}/crop_{patch_size}_{num_images}_seed_{args.seed}",
    )
    os.makedirs(save_dir, exist_ok=True)
    print("Image dir:", img_dir)
    print("Save to:", save_dir)
    print("Source images:", len(img_path_list))

    vae = None
    transform = None
    if args.vqgan_type is not None:
        vqgan_ckpt_dir = "checkpoint/pretrained_vq_model"
        weight_path = os.path.join(vqgan_ckpt_dir, f"vq-{args.vqgan_type}.ckpt")
        assert os.path.exists(weight_path), f"missing VQGAN ckpt: {weight_path}"
        vq_config = get_vqgan_config(args.vqgan_type)
        vae = build_vqgan(vq_config, weight_path, device=args.device)
        transform = T.Compose(
            (
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            )
        )

    cnt = 0
    qbar = tqdm(total=num_images, ncols=80)
    for img_path in img_path_list:
        img = read_img(img_path)
        w, h = img.size
        if min(w, h) < min_short_length:
            continue

        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        img = img.crop((left, top, left + patch_size, top + patch_size))

        img_name = os.path.basename(img_path)
        stem, _ = os.path.splitext(img_name)
        save_path = os.path.join(save_dir, f"{stem}.png")

        img_np = np.array(img, dtype=np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

        if vae is not None:
            assert transform is not None
            img_torch = transform(img.copy()).unsqueeze(0).to(args.device)
            indices = vqgan_encode(vae, img_torch).cpu().numpy()
            if args.vqgan_type == "f8-n256":
                indices = indices.astype(np.uint8)
            else:
                indices = indices.astype(np.uint16)
            np.save(save_path.replace(".png", ".npy"), indices[0])

        cnt += 1
        qbar.update(1)
        if cnt >= num_images:
            break

    print(f"Saved {cnt} samples to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqgan_type", type=str, default=None)
    parser.add_argument("--openimage_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="./datasets/openimage_validation")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--num_img", type=int, default=2000)
    args = parser.parse_args()
    if args.vqgan_type not in {None, "f8-n256", "f16", "f8"}:
        raise ValueError(f"unsupported vqgan_type: {args.vqgan_type}")
    main(args)
