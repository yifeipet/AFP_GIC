#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List

import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


RUNTIME_DIR = Path(__file__).resolve().parent
PUBLIC_RELEASE_DIR = RUNTIME_DIR.parent
ROOT_DIR = PUBLIC_RELEASE_DIR.parent
sys.path.insert(0, str(RUNTIME_DIR))

from src.models import build_comp_model  # noqa: E402
from src.utils import img_utils  # noqa: E402
from src.utils.options import BaseConfig  # noqa: E402
import src  # noqa: F401,E402  # register classes


TARGET_BPPS = [0.050, 0.075, 0.100, 0.125, 0.150]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_path: str
    model_path: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    img_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument(
        "--results-root",
        default=str(ROOT_DIR / "results" / "stage3_selected_beta_sweep"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing summary.json if present.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Save reconstructed png files in addition to bitstreams and metrics.",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_false",
        dest="save_images",
        help="Do not save reconstructed png files.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to evaluate.",
    )
    parser.add_argument(
        "--qualities",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of quality indices to evaluate.",
    )
    return parser.parse_args()


def load_infer_config(config_path: str, device: str) -> BaseConfig:
    cfg_dict, cfg_text, _ = BaseConfig._file2dict_yaml(config_path)
    cfg_dict["is_train"] = False
    cfg_dict["device"] = device
    cfg_dict["config_path"] = config_path
    prior_cfg = cfg_dict.get("subnet", {}).get("adacode_prior", {})
    prior_ckpt = prior_cfg.get("ckpt_path")
    if prior_ckpt:
        prior_ckpt_path = Path(prior_ckpt)
        if not prior_ckpt_path.exists():
            candidates = [
                RUNTIME_DIR
                / "external_libs"
                / "adacode"
                / "weights_release_v0"
                / prior_ckpt_path.name,
                ROOT_DIR
                / "dc_vic_adacode_merge"
                / "external_libs"
                / "adacode"
                / "weights_release_v0"
                / prior_ckpt_path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    cfg_dict["subnet"]["adacode_prior"]["ckpt_path"] = str(candidate)
                    break
            else:
                raise FileNotFoundError(
                    "AdaCode prior weights not found. Expected one of: "
                    + ", ".join(str(path) for path in [prior_ckpt_path, *candidates])
                )
    return BaseConfig(cfg_dict, cfg_text=cfg_text, filename=config_path)


def resolve_model_path(*candidates: Path) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "None of the expected model checkpoints were found: "
        + ", ".join(str(path) for path in candidates)
    )


def get_model_specs() -> List[ModelSpec]:
    public_release_ckpt = (
        ROOT_DIR / "checkpoint" / "afp_gic_release" / "model" / "afp_gic_release.pth.tar"
    )
    legacy_public_release_ckpt = (
        ROOT_DIR / "checkpoint" / "afp_gic_release" / "model" / "comp_model.pth.tar"
    )
    staged_public_release_ckpt = (
        ROOT_DIR
        / "public_release_assets"
        / "afp_gic_release"
        / "model"
        / "afp_gic_release.pth.tar"
    )
    return [
        ModelSpec(
            name="afp_gic_release",
            config_path=str(
                RUNTIME_DIR
                / "config"
                / "afp_gic_release.yaml"
            ),
            model_path=resolve_model_path(
                public_release_ckpt,
                legacy_public_release_ckpt,
                staged_public_release_ckpt,
            ),
        ),
    ]


def get_dataset_specs() -> List[DatasetSpec]:
    return [
        DatasetSpec(name="kodak", img_dir=str(ROOT_DIR / "datasets" / "kodak")),
        DatasetSpec(name="clic2020_test", img_dir=str(ROOT_DIR / "datasets" / "CLIC" / "clic_test_images")),
        DatasetSpec(
            name="div2k_valid_hr",
            img_dir=str(ROOT_DIR / "datasets" / "DIV2K_valid_HR" / "DIV2K_valid_HR"),
        ),
    ]


def get_image_paths(img_dir: str) -> List[str]:
    path_list = sorted(glob(os.path.join(img_dir, "*.png")))
    if not path_list:
        raise FileNotFoundError(f"No png files found under {img_dir}")
    return path_list


def read_real_tensor(img_path: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def load_summary(summary_path: Path) -> Dict:
    with open(summary_path, "r") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_actual_bit_count(string_list: List[bytes]) -> int:
    # `save_byte_strings()` stores each payload as [uint32 length][payload bytes].
    return sum((4 + len(byte_string)) * 8 for byte_string in string_list)


def write_bitrate_files(combo_dir: Path, rows: List[Dict]) -> Dict[str, str]:
    bitrate_csv = combo_dir / "_bitrates.csv"
    avg_bitrate_json = combo_dir / "_avg_bitrate.json"
    bitrate_df = pd.DataFrame(
        [{"img_name": row["img_name"], "real_bpp": row["bpp"]} for row in rows]
    )
    bitrate_df.to_csv(bitrate_csv, index=False)
    with open(avg_bitrate_json, "w") as f:
        json.dump({"avg_bpp": float(bitrate_df["real_bpp"].mean())}, f)
    return {
        "bitrate_csv": str(bitrate_csv),
        "avg_bitrate_json": str(avg_bitrate_json),
    }


def run_author_calc_metrics(real_dir: str, fake_dir: str, device: str) -> Dict:
    cmd = [
        sys.executable,
        str(RUNTIME_DIR / "calc_metrics.py"),
        "--real_dir",
        real_dir,
        "--fake_dir",
        fake_dir,
        "-d",
        device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RUNTIME_DIR)
    print(f"[author-metrics] start fake_dir={fake_dir}")
    subprocess.run(cmd, check=True, cwd=str(RUNTIME_DIR), env=env)
    metrics_path = Path(fake_dir) / "_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    print(f"[author-metrics] done fake_dir={fake_dir}")
    return metrics


def evaluate_combo(
    model_spec: ModelSpec,
    dataset_spec: DatasetSpec,
    quality_ind: int,
    target_bpp: float,
    device: str,
    combo_dir: Path,
    save_images: bool,
    combo_index: int,
    combo_total: int,
) -> Dict:
    combo_dir.mkdir(parents=True, exist_ok=True)
    summary_path = combo_dir / "summary.json"
    per_image_csv = combo_dir / "per_image_metrics.csv"
    image_paths = get_image_paths(dataset_spec.img_dir)
    combo_start = time.time()

    print(
        f"[combo {combo_index}/{combo_total}] start "
        f"model={model_spec.name} dataset={dataset_spec.name} "
        f"q={quality_ind} target_bpp={target_bpp:.3f} "
        f"num_images={len(image_paths)}"
    )

    opt = load_infer_config(model_spec.config_path, device=device)
    model = build_comp_model(opt).to(device)
    model.load_learned_weight(ckpt_path=model_spec.model_path)
    model.codec_setup()
    model.eval()

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    rows = []
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["img_name", "bpp", "PSNR", "MS-SSIM", "LPIPS"]
        )
        writer.writeheader()

        with torch.no_grad():
            progress = tqdm(
                image_paths,
                desc=(
                    f"[{combo_index}/{combo_total}] "
                    f"{model_spec.name} {dataset_spec.name} q{quality_ind}"
                ),
                ncols=120,
                dynamic_ncols=True,
                smoothing=0.1,
            )
            for img_path in progress:
                img_name = os.path.basename(img_path)
                real_img = read_real_tensor(img_path)
                _, _, h, w = real_img.shape

                out_dict = model.compress(real_img, quality_ind=quality_ind)
                actual_bpp = get_actual_bit_count(out_dict["string_list"]) / (h * w)
                fake_img, _, _ = model.decompress(out_dict["string_list"])

                if save_images:
                    img_utils.imwrite(str(combo_dir / img_name), fake_img)

                psnr = float(img_utils.calc_psnr(real_img, fake_img, 255))
                ms_ssim = float(img_utils.calc_ms_ssim(real_img, fake_img))
                lpips_val = float(
                    lpips_fn(fake_img.to(device), real_img.to(device)).item()
                )
                row = {
                    "img_name": img_name,
                    "bpp": actual_bpp,
                    "PSNR": psnr,
                    "MS-SSIM": ms_ssim,
                    "LPIPS": lpips_val,
                }
                rows.append(row)
                writer.writerow(row)
                f.flush()
                progress.set_postfix(
                    avg_bpp=f"{np.mean([r['bpp'] for r in rows]):.4f}",
                    avg_psnr=f"{np.mean([r['PSNR'] for r in rows]):.3f}",
                    avg_lpips=f"{np.mean([r['LPIPS'] for r in rows]):.4f}",
                )

    summary = {
        "model": model_spec.name,
        "dataset": dataset_spec.name,
        "quality_ind": quality_ind,
        "target_bpp": target_bpp,
        "num_images": len(rows),
        "bpp": float(np.mean([row["bpp"] for row in rows])),
        "PSNR": float(np.mean([row["PSNR"] for row in rows])),
        "MS-SSIM": float(np.mean([row["MS-SSIM"] for row in rows])),
        "LPIPS": float(np.mean([row["LPIPS"] for row in rows])),
        "config_path": model_spec.config_path,
        "model_path": model_spec.model_path,
        "img_dir": dataset_spec.img_dir,
        "per_image_csv": str(per_image_csv),
        "elapsed_sec": time.time() - combo_start,
    }

    summary.update(write_bitrate_files(combo_dir, rows))
    author_metrics = run_author_calc_metrics(
        real_dir=dataset_spec.img_dir,
        fake_dir=str(combo_dir),
        device=device,
    )
    summary["author_metrics"] = author_metrics
    summary["author_metrics_json"] = str(combo_dir / "_metrics.json")
    save_json(summary_path, summary)
    print(
        f"[combo {combo_index}/{combo_total}] done "
        f"elapsed={summary['elapsed_sec'] / 60.0:.2f} min "
        f"bpp={summary['bpp']:.6f} PSNR={summary['PSNR']:.4f} "
        f"MS-SSIM={summary['MS-SSIM']:.6f} LPIPS={summary['LPIPS']:.6f}"
    )
    return summary


def build_pivot(summary_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    subset = summary_df[summary_df["dataset"] == dataset_name].copy()
    subset["target_bpp_label"] = subset["target_bpp"].map(lambda x: f"{x:.3f}")
    pivot_rows = []
    metric_names = ["bpp", "PSNR", "MS-SSIM", "LPIPS"]
    for target_bpp in [f"{x:.3f}" for x in TARGET_BPPS]:
        target_df = subset[subset["target_bpp_label"] == target_bpp]
        row = {"dataset": dataset_name, "target_bpp": target_bpp}
        for _, rec in target_df.iterrows():
            prefix = rec["model"]
            for metric_name in metric_names:
                row[f"{prefix}_{metric_name}"] = rec[metric_name]
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    model_specs = get_model_specs()
    if args.models is not None:
        model_specs = [spec for spec in model_specs if spec.name in set(args.models)]
        if not model_specs:
            raise ValueError(f"No model matched --models={args.models}")

    dataset_specs = get_dataset_specs()
    if args.datasets is not None:
        dataset_specs = [spec for spec in dataset_specs if spec.name in set(args.datasets)]
        if not dataset_specs:
            raise ValueError(f"No dataset matched --datasets={args.datasets}")

    quality_pairs = list(enumerate(TARGET_BPPS))
    if args.qualities is not None:
        quality_set = set(args.qualities)
        quality_pairs = [pair for pair in quality_pairs if pair[0] in quality_set]
        if not quality_pairs:
            raise ValueError(f"No quality matched --qualities={args.qualities}")

    combo_specs = []
    for dataset_spec in dataset_specs:
        for model_spec in model_specs:
            for quality_ind, target_bpp in quality_pairs:
                combo_specs.append((dataset_spec, model_spec, quality_ind, target_bpp))

    summary_rows = []
    completed_combo_times = []
    combo_total = len(combo_specs)
    for combo_index, (dataset_spec, model_spec, quality_ind, target_bpp) in enumerate(
        combo_specs, start=1
    ):
        combo_dir = results_root / dataset_spec.name / model_spec.name / f"q{quality_ind}"
        summary_path = combo_dir / "summary.json"
        combo_start = time.time()
        if args.skip_existing and summary_path.exists():
            summary = load_summary(summary_path)
            print(
                f"[combo {combo_index}/{combo_total}] skip existing "
                f"model={model_spec.name} dataset={dataset_spec.name} q={quality_ind}"
            )
        else:
            summary = evaluate_combo(
                model_spec=model_spec,
                dataset_spec=dataset_spec,
                quality_ind=quality_ind,
                target_bpp=target_bpp,
                device=args.device,
                combo_dir=combo_dir,
                save_images=args.save_images,
                combo_index=combo_index,
                combo_total=combo_total,
            )
        summary_rows.append(summary)
        combo_elapsed = time.time() - combo_start
        completed_combo_times.append(combo_elapsed)
        remaining_combo = combo_total - combo_index
        if remaining_combo > 0:
            avg_combo_sec = sum(completed_combo_times) / len(completed_combo_times)
            eta_sec = avg_combo_sec * remaining_combo
            print(
                f"[overall] completed={combo_index}/{combo_total} "
                f"last_combo={combo_elapsed / 60.0:.2f} min "
                f"avg_combo={avg_combo_sec / 60.0:.2f} min "
                f"eta_remaining={eta_sec / 3600.0:.2f} h"
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "quality_ind", "model"])
    summary_csv = results_root / "summary_all.csv"
    summary_df.to_csv(summary_csv, index=False)

    pivot_frames = [build_pivot(summary_df, dataset_name=ds.name) for ds in dataset_specs]
    comparison_csv = results_root / "comparison_pivot.csv"
    if pivot_frames:
        comparison_df = pd.concat(pivot_frames, ignore_index=True)
        comparison_df.to_csv(comparison_csv, index=False)
    else:
        pd.DataFrame().to_csv(comparison_csv, index=False)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {comparison_csv}")


if __name__ == "__main__":
    main()
