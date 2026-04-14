#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / "public_release" / "runtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Public AFP-GIC evaluation entry point."
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--dataset",
        default="kodak",
        choices=["kodak", "clic2020_test", "div2k_valid_hr"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--qualities",
        nargs="*",
        type=int,
        default=None,
        help="Optional quality indices, e.g. --qualities 0 or --qualities 0 1 2 3 4.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional custom output root. Defaults to results/public_release_eval/<dataset>.",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Optional Python interpreter used for the underlying evaluation script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    if results_root is None:
        results_root = str(ROOT_DIR / "results" / "public_release_eval" / args.dataset)
    python_bin = args.python_bin or os.environ.get("PYTHON_BIN") or sys.executable

    cmd = [
        python_bin,
        str(RUNTIME_DIR / "eval_public_release.py"),
        "-d",
        args.device,
        "--models",
        "afp_gic_release",
        "--datasets",
        args.dataset,
        "--results-root",
        results_root,
        "--save-images",
    ]
    if args.qualities:
        cmd.extend(["--qualities", *[str(q) for q in args.qualities]])

    print(f"repo_root={ROOT_DIR}")
    print(f"runtime_dir={RUNTIME_DIR}")
    print(f"dataset={args.dataset}")
    print("model=afp_gic_release")
    print(f"python_bin={python_bin}")
    print(f"results_root={results_root}")
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))


if __name__ == "__main__":
    main()
